"""Make a new views directory whose inputs are links into an existing one (nothing of the source is written).

Example:
    .venvs/main/bin/python scripts/link_views.py --src results/E14/val056/views/r2 --dst results/_diagnostics/DB2/val056/r2/views \
        --names cams.json filled mask depth count
"""
import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    args = parser.parse_args()
    os.makedirs(args.dst, exist_ok=False)
    for name in args.names:
        src = os.path.abspath(os.path.join(args.src, name))
        assert os.path.exists(src), src
        os.symlink(src, os.path.join(args.dst, name))
    print(f"[link_views] {args.dst}: {', '.join(args.names)} -> {args.src}", flush=True)


if __name__ == "__main__":
    main()
