#!/usr/bin/env bash
# Download the PDFs listed in papers.tsv from arXiv into this directory as <arxiv_id>_<short_name>.pdf.
# The export mirror (export.arxiv.org, meant for scripts) is used first, 3 s apart; it cuts transfers at 32 MiB
# (curl error 18; 2604.06113 is 40 MB), and for those the file comes from arxiv.org instead, with a message.
# A file counts as fetched only if it starts with %PDF- and ends with %%EOF.
# The PDFs are not in git (size; each paper keeps its own arXiv license).
set -uo pipefail
cd "$(dirname "$0")"
UA="DashRecon literature collection"
complete_pdf() { head -c 5 "$1" | grep -q "%PDF-" && tail -c 1024 "$1" | grep -aq "%%EOF"; }
grep -v '^#' papers.tsv | while IFS=$'\t' read -r id name _; do
  out="${id}_${name}.pdf"
  if [ -s "$out" ]; then continue; fi
  curl -sSfL -m 300 -A "$UA" -o "$out.part" "https://export.arxiv.org/pdf/${id}"
  status=$?
  if [ $status -eq 18 ]; then
    echo "export mirror truncated $id (curl 18); fetching from arxiv.org" >&2
    curl -sSfL -m 300 -A "$UA" -o "$out.part" "https://arxiv.org/pdf/${id}"
    status=$?
  fi
  if [ $status -ne 0 ] || ! complete_pdf "$out.part"; then
    echo "download failed or incomplete: $id (curl $status)" >&2
    rm -f "$out.part"
    exit 1
  fi
  mv "$out.part" "$out"
  echo "fetched $out ($(du -h "$out" | cut -f1))"
  sleep 3
done
