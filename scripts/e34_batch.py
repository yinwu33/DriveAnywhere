"""E34 (docs/E34_DATA.md): run the pair pipeline over the selected segments, resumable, within the disk and GPU limits.

Every stage is its own CLI script with disk inputs and outputs; this only schedules them and records what happened.
A stage is done when its output exists, so a restarted batch continues where it stopped.

Phase "points" (pair type a), per segment in the order of --segments:
    fetch      scripts/e34_fetch.py              network, one at a time, at most --max_raw raw segments on disk
    masks      scripts/e34_masks.py               GPU lane (masks venv)
    calib      scripts/e34_calib.py               CPU pool of --calib_jobs x --calib_threads (sfm venv)
    undistort  scripts/e34_undistort.py           CPU pool, accepted segments only
    diag       scripts/e34_diag_calib.py          CPU pool, evaluation-side GT comparison (reported, never used)
    pose       scripts/run_pose.py --camera_dir   GPU lane (mapanything venv)
    points     scripts/e34_points_pairs.py        GPU lane (main venv)
    finalize   scripts/e34_finalize.py --cleanup  per-segment meta.json; segments at positions < --keep_first keep
                                                  their reconstruction inputs for phase "gs"
Phase "gs" (pair type b), for the first --gs_count accepted segments: scripts/run_fusion.py (main venv), run_mesh.py
(nksr venv), e34_train_gs.py, e34_gs_pairs.py, e34_finalize.py --cleanup; then the other kept segments are cleaned.

One GPU job at a time; a GPU job starts only when nvidia-smi reports at least its memory need free (other users share
the GPU) and no other E34 process holds the GPU (a stage started by hand, recognised by "/e34/" or "scripts/e34_" in
its command line). New work stops when /home has less than --min_free_gb free. A stage that exits non-zero marks the segment
failed (stage, log) and the batch goes on; --max_failures failures stop it. Stage output goes to
<log_root>/e34_<scene_id>.log, the schedule to <log_root>/e34_batch_<phase>.log, the state to <root>/batch_<phase>.json.

Example (any venv; the stages use the venvs under --venvs):
    python scripts/e34_batch.py --phase points --segments data/e34/segments.json --root data/e34 \
        --venvs .venvs --cuda_home $HOME/cuda-12.6 --log_root results/_logs \
        --remote onedrive:/Projects/P05_DriveAnywhere/waymo_processed/training --tpslimit 8 --max_raw 12 \
        --calib_jobs 4 --calib_threads 16 --keep_first 30 --gs_count 20 --min_free_gb 150 --max_failures 8
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashrecon.provenance import git_commit  # noqa: E402

MASK_ARGS = ["--target_cams", "1", "2", "3", "4", "--target_stride", "2", "--detector_id", "IDEA-Research/grounding-dino-base",
             "--sam_id", "facebook/sam2.1-hiera-large", "--box_threshold", "0.25", "--text_threshold", "0.25",
             "--dilate_px", "5", "--seg_model_id", "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
             "--seg_input_width", "1536", "--tf32"]
CALIB_ARGS = ["--target_cams", "1", "2", "3", "4", "--overlap_front", "20", "--overlap_target", "10",
              "--cross", "0-1", "0-2", "1-3", "2-4", "--cross_window", "10", "--cross_key_stride", "2",
              "--max_features", "8192", "--seed", "0", "--refine_pp_cams", "3", "4", "--min_front_frac", "0.9",
              "--min_target_frac", "0.8"]
PAIR_HW = ["576", "1024"]
POSE_ARGS = ["--backend", "mapanything", "--model_id", "facebook/map-anything", "--scale", "model", "--max_views", "300",
             "--minibatch_size", "16"]
POINTS_ARGS = ["--path_back", "40", "--path_fwd", "10", "--max_window_frames", "40", "--conf_percentile", "30",
               "--max_depth", "60", "--max_splat_px", "5"]
FUSION_ARGS = ["--test_stride", "10", "--conf_percentile", "30", "--max_depth", "60", "--consistency_k", "4",
               "--consistency_rel", "0.05", "--consistency_min", "2", "--voxel", "0.05", "--outlier_nb", "20",
               "--outlier_std", "2.0", "--road_cell", "0.5", "--road_sigma", "2.0", "--road_min_points", "5",
               "--road_max_dz", "0.3", "--normal_knn", "30", "--rejected_sample", "200000", "--seed", "0"]
MESH_ARGS = ["--detail_level", "0.5", "--mise_iter", "1", "--solver_tol", "1e-4", "--coverage_radius", "0.15"]
GPU_NEED_GB = {"masks": 10, "pose": 30, "points": 10, "mesh": 8, "train": 12, "render": 12}
CALIB, POSE = "calib-glomap-cams", "pose-glomap-cams_depth-mapanything"
FRONT_MASKS = "mask-gsam2_sky-segformer_img-glomap-cams"


def remove(path: str) -> None:
    """Delete the partial output of an interrupted stage before it is run again."""
    if os.path.exists(path):
        shutil.rmtree(path)


class Batch:
    def __init__(self, args):
        self.a = args
        self.root = os.path.abspath(args.root)
        self.segs = json.load(open(args.segments))["selected"]
        self.lock = threading.Lock()
        self.claimed, self.failed, self.stop = set(), {}, None
        self.gpu_busy = False
        self.log_path = os.path.join(args.log_root, f"e34_batch_{args.phase}.log")
        self.state_path = os.path.join(self.root, f"batch_{args.phase}.json")
        self.timing = {}
        self.env = dict(os.environ, CUDA_HOME=args.cuda_home, TORCH_CUDA_ARCH_LIST="9.0", HF_HUB_OFFLINE="1",
                        PATH=f"{os.path.join(args.venvs, 'main', 'bin')}:{os.path.join(args.cuda_home, 'bin')}:{os.environ['PATH']}")

    # paths
    def p(self, *parts):
        return os.path.join(self.root, *parts)

    def raw(self, s):
        return self.p("raw", f"{s['idx']:03d}")

    def work(self, s):
        return self.p("work", s["scene_id"])

    def pairs(self, s):
        return self.p("pairs", s["scene_id"])

    def py(self, venv):
        return os.path.join(self.a.venvs, venv, "bin", "python")

    def log(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        with self.lock:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")
        print(line, flush=True)

    def save_state(self):
        with self.lock:
            state = {"phase": self.a.phase, "failed": self.failed, "stop": self.stop, "timing_s": self.timing,
                     "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "dashrecon_commit": self.commit}
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def free_gb(self):
        return shutil.disk_usage(self.root).free / 1e9

    def gpu_free_gb(self):
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], text=True)
        used, total = (float(x) for x in out.strip().split(","))
        return (total - used) / 1024

    def run(self, s, stage, argv):
        """Run one stage; False (and the segment marked failed) on a non-zero exit."""
        t0 = time.time()
        self.log(f"{s['scene_id']} {stage} start")
        with open(os.path.join(self.a.log_root, f"e34_{s['scene_id']}.log"), "a") as f:
            f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {stage}: {' '.join(argv)}\n")
            f.flush()
            rc = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, env=self.env).returncode
        dt = time.time() - t0
        with self.lock:
            self.timing.setdefault(s["scene_id"], {})[stage] = dt
        if rc != 0:
            with self.lock:
                self.failed[s["scene_id"]] = {"stage": stage, "returncode": rc,
                                              "log": os.path.join(self.a.log_root, f"e34_{s['scene_id']}.log")}
                if len(self.failed) >= self.a.max_failures:
                    self.stop = f"{len(self.failed)} failures"
            self.log(f"{s['scene_id']} {stage} FAILED rc={rc} after {dt:.0f} s")
        else:
            self.log(f"{s['scene_id']} {stage} done in {dt:.0f} s")
        self.save_state()
        return rc == 0

    def check_disk(self):
        free = self.free_gb()
        if free < self.a.min_free_gb:
            with self.lock:
                self.stop = f"free disk {free:.0f} GB < {self.a.min_free_gb} GB"
        return self.stop is None

    def other_e34_gpu_jobs(self):
        """GPU processes whose command line marks them as E34 work (e.g. a pilot stage started by hand)."""
        out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
        busy = []
        for pid in (int(x) for x in out.split()):
            cmd_file = f"/proc/{pid}/cmdline"
            if os.path.exists(cmd_file):
                cmd = open(cmd_file, "rb").read().replace(b"\0", b" ").decode()
                if "/e34/" in cmd or "scripts/e34_" in cmd:
                    busy.append(pid)
        return busy

    def wait_gpu(self, need):
        while self.gpu_free_gb() < need or self.other_e34_gpu_jobs():
            time.sleep(20)

    # state tests
    def finished(self, s):
        return os.path.exists(os.path.join(self.pairs(s), "meta.json"))

    def fetched(self, s):
        return os.path.exists(os.path.join(self.raw(s), "fetch.json"))

    def masked(self, s):
        return os.path.exists(os.path.join(self.work(s), "masks_orig", "meta.json"))

    def calib_meta(self, s):
        f = os.path.join(self.work(s), CALIB, "meta.json")
        return json.load(open(f)) if os.path.exists(f) else None

    def undistorted(self, s):
        return os.path.exists(os.path.join(self.pairs(s), "targets.json"))

    def diag_done(self, s):
        return os.path.exists(self.p("diag", f"{s['scene_id']}_calib_gt.json"))

    def stage_args(self, s, stage):
        sid, w, pr = s["scene_id"], self.work(s), self.pairs(s)
        raw_root, front_root = self.p("raw"), self.p("front")
        if stage == "fetch":
            return [self.py("main"), "scripts/e34_fetch.py", "--idx", str(s["idx"]), "--remote", self.a.remote,
                    "--raw_root", raw_root, "--gt_root", self.p("gt"), "--tpslimit", str(self.a.tpslimit)]
        if stage == "masks":
            return [self.py("masks"), "scripts/e34_masks.py", "--scene_id", sid, "--raw_root", raw_root,
                    "--out_dir", os.path.join(w, "masks_orig"), *MASK_ARGS]
        if stage == "calib":
            return [self.py("sfm"), "scripts/e34_calib.py", "--scene_id", sid, "--raw_root", raw_root,
                    "--mask_dir", os.path.join(w, "masks_orig"), "--out_dir", os.path.join(w, CALIB),
                    "--front_root", front_root, "--num_threads", str(self.a.calib_threads), *CALIB_ARGS]
        if stage == "undistort":
            return [self.py("sfm"), "scripts/e34_undistort.py", "--scene_id", sid, "--raw_root", raw_root,
                    "--mask_dir", os.path.join(w, "masks_orig"), "--calib_dir", os.path.join(w, CALIB),
                    "--front_root", front_root, "--front_mask_dir", os.path.join(w, FRONT_MASKS), "--pairs_dir", pr,
                    "--pair_hw", *PAIR_HW]
        if stage == "diag":
            return [self.py("sfm"), "scripts/e34_diag_calib.py", "--calib_dir", os.path.join(w, CALIB), "--gt_root",
                    self.p("gt"), "--scene_idx", str(s["idx"]), "--out", self.p("diag", f"{sid}_calib_gt.json")]
        if stage == "pose":
            return [self.py("mapanything"), "scripts/run_pose.py", "--scene_id", sid, "--processed_root", front_root,
                    "--camera_dir", os.path.join(w, CALIB), "--out_root", self.p("work"), *POSE_ARGS]
        if stage == "points":
            return [self.py("main"), "scripts/e34_points_pairs.py", "--scene_id", sid, "--calib_dir", os.path.join(w, CALIB),
                    "--pose_dir", os.path.join(w, POSE), "--mask_dir", os.path.join(w, FRONT_MASKS),
                    "--front_root", front_root, "--pairs_dir", pr, *POINTS_ARGS]
        if stage == "fusion":
            return [self.py("main"), "scripts/run_fusion.py", "--scene_id", sid, "--pose_dir", os.path.join(w, POSE),
                    "--mask_dir", os.path.join(w, FRONT_MASKS), "--processed_root", front_root, "--out_root", self.p("work"),
                    *FUSION_ARGS]
        if stage == "mesh":
            return [self.py("nksr"), "scripts/run_mesh.py", "--fusion_dir", os.path.join(w, f"{POSE}__{FRONT_MASKS}"),
                    *MESH_ARGS]
        if stage == "train":
            return [self.py("main"), "scripts/e34_train_gs.py", "--scene_id", sid, "--front_root", front_root,
                    "--pose_dir", os.path.join(w, POSE), "--mask_dir", os.path.join(w, FRONT_MASKS),
                    "--fusion_dir", os.path.join(w, f"{POSE}__{FRONT_MASKS}"), "--product_root", self.p("work"),
                    "--output_root", os.path.join(w, "gs"), *self.a.train_opts]
        if stage == "render":
            return [self.py("main"), "scripts/e34_gs_pairs.py", "--scene_id", sid,
                    "--log_dir", os.path.join(w, "gs", "E5f", sid), "--calib_dir", os.path.join(w, CALIB), "--pairs_dir", pr]
        raise ValueError(stage)

    def finalize(self, s, keep):
        argv = [self.py("main"), "scripts/e34_finalize.py", "--scene_id", s["scene_id"], "--raw_root", self.p("raw"),
                "--work_dir", self.work(s), "--front_root", self.p("front"), "--pairs_dir", self.pairs(s), "--cleanup"]
        return self.run(s, "finalize", argv + (["--keep_reconstruction"] if keep else []))

    def active(self, s):
        with self.lock:
            return s["scene_id"] not in self.failed

    # phase "points" lanes
    def fetch_lane(self):
        for pos, s in enumerate(self.segs[: self.a.limit]):
            if self.finished(s) or self.fetched(s) or not self.active(s):
                continue
            while True:
                if self.stop is not None:
                    return
                n_raw = sum(1 for x in self.segs if os.path.exists(self.raw(x)) and self.active(x))
                if n_raw < self.a.max_raw and self.check_disk():
                    break
                time.sleep(10)
            self.run(s, "fetch", self.stage_args(s, "fetch"))

    def gpu_lane(self):
        todo = self.segs[: self.a.limit]
        while self.stop is None:
            if all(self.finished(s) or not self.active(s) for s in todo):
                return
            job = None
            for pos, s in enumerate(todo):
                if self.active(s) and not self.finished(s) and self.undistorted(s) and self.diag_done(s):
                    job = ("pairs", pos, s)
                    break
            if job is None:
                waiting = sum(1 for s in todo if self.active(s) and self.masked(s) and self.calib_meta(s) is None)
                if waiting < self.a.calib_jobs + 2:
                    for pos, s in enumerate(todo):
                        if self.active(s) and self.fetched(s) and not self.masked(s) and not self.finished(s):
                            job = ("masks", pos, s)
                            break
            if job is None:
                time.sleep(10)
                continue
            kind, pos, s = job
            if kind == "masks":
                remove(os.path.join(self.work(s), "masks_orig"))  # a half-written earlier try
                self.wait_gpu(GPU_NEED_GB["masks"])
                self.run(s, "masks", self.stage_args(s, "masks"))
                continue
            if not os.path.exists(os.path.join(self.work(s), POSE, "meta.json")):
                remove(os.path.join(self.work(s), POSE))
                self.wait_gpu(GPU_NEED_GB["pose"])
                if not self.run(s, "pose", self.stage_args(s, "pose")):
                    continue
            remove(os.path.join(self.pairs(s), "points"))
            self.wait_gpu(GPU_NEED_GB["points"])
            if not self.run(s, "points", self.stage_args(s, "points")):
                continue
            self.finalize(s, keep=pos < self.a.keep_first)

    def cpu_worker(self):
        todo = self.segs[: self.a.limit]
        while self.stop is None:
            job = None
            with self.lock:
                for s in todo:
                    if s["scene_id"] in self.claimed or s["scene_id"] in self.failed or not self.masked(s) or self.finished(s):
                        continue
                    meta = self.calib_meta(s)
                    if meta is None or not meta["accepted"] or not (self.undistorted(s) and self.diag_done(s)):
                        job = s
                        self.claimed.add(s["scene_id"])
                        break
            if job is None:
                if all(self.finished(s) or not self.active(s) for s in todo):
                    return
                time.sleep(10)
                continue
            s = job
            if self.calib_meta(s) is None:
                remove(os.path.join(self.work(s), CALIB))
                if not self.run(s, "calib", self.stage_args(s, "calib")):
                    continue
            meta = self.calib_meta(s)
            if not meta["accepted"]:
                self.log(f"{s['scene_id']} REJECTED: {'; '.join(meta['reasons'])}")
                self.finalize(s, keep=False)
                continue
            if not self.undistorted(s):
                remove(os.path.join(self.p("front"), f"{s['idx']:03d}"))
                remove(os.path.join(self.work(s), FRONT_MASKS))
                remove(self.pairs(s))
                if not self.run(s, "undistort", self.stage_args(s, "undistort")):
                    continue
            if not self.diag_done(s):
                self.run(s, "diag", self.stage_args(s, "diag"))

    # phase "gs"
    def gs_lane(self):
        done = 0
        for pos, s in enumerate(self.segs[: self.a.keep_first]):
            if self.stop is not None:
                return
            meta_f = os.path.join(self.pairs(s), "meta.json")
            if not os.path.exists(meta_f) or not json.load(open(meta_f))["accepted"]:
                continue
            if done >= self.a.gs_count:
                if os.path.exists(os.path.join(self.work(s), POSE)):
                    self.finalize(s, keep=False)
                continue
            done += 1
            if os.path.exists(os.path.join(self.pairs(s), "gs.json")):
                continue
            if not self.check_disk():
                return
            w = self.work(s)
            fusion = os.path.join(w, f"{POSE}__{FRONT_MASKS}")
            steps = [("fusion", os.path.join(fusion, "meta.json")), ("mesh", os.path.join(fusion, "mesh_nksr.json")),
                     ("train", os.path.join(w, "gs", "E5f", s["scene_id"], "meta.json")),
                     ("render", os.path.join(self.pairs(s), "gs.json"))]
            ok = True
            for stage, marker in steps:
                if os.path.exists(marker):
                    continue
                if stage == "train":
                    remove(os.path.join(w, "gs"))
                if stage == "render":
                    remove(os.path.join(self.pairs(s), "gs"))
                if stage in GPU_NEED_GB:
                    self.wait_gpu(GPU_NEED_GB[stage])
                if not self.run(s, stage, self.stage_args(s, stage)):
                    ok = False
                    break
            if ok:
                self.finalize(s, keep=False)

    def main(self):
        self.commit = git_commit()
        assert not self.commit.endswith("-dirty"), "commit the code before a batch run (AGENTS.md section 0.8)"
        os.makedirs(self.p("diag"), exist_ok=True)
        self.log(f"start phase {self.a.phase}, commit {self.commit}, {len(self.segs[: self.a.limit])} segments, "
                 f"free disk {self.free_gb():.0f} GB")
        if self.a.phase == "points":
            threads = [threading.Thread(target=self.fetch_lane), threading.Thread(target=self.gpu_lane)]
            threads += [threading.Thread(target=self.cpu_worker) for _ in range(self.a.calib_jobs)]
        else:
            threads = [threading.Thread(target=self.gs_lane)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.save_state()
        self.log(f"end phase {self.a.phase}: stop={self.stop}, failed={sorted(self.failed)}, free disk {self.free_gb():.0f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True, choices=["points", "gs"])
    parser.add_argument("--segments", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--venvs", required=True)
    parser.add_argument("--cuda_home", required=True)
    parser.add_argument("--log_root", required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--tpslimit", type=int, required=True)
    parser.add_argument("--max_raw", type=int, required=True)
    parser.add_argument("--calib_jobs", type=int, required=True)
    parser.add_argument("--calib_threads", type=int, required=True)
    parser.add_argument("--keep_first", type=int, required=True, help="positions whose reconstruction phase gs may use")
    parser.add_argument("--gs_count", type=int, required=True)
    parser.add_argument("--min_free_gb", type=float, required=True)
    parser.add_argument("--max_failures", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None, help="only the first N selected segments")
    parser.add_argument("--train_opts", nargs="*", default=[], help="OmegaConf overrides for e34_train_gs.py")
    args = parser.parse_args()
    args.venvs = os.path.abspath(args.venvs)
    Batch(args).main()


if __name__ == "__main__":
    main()
