#!/usr/bin/env bash
# Download the PDFs listed in papers.tsv from arXiv (export mirror, 3 s apart) into this directory as
# <arxiv_id>_<short_name>.pdf. The PDFs are not in git (size; each paper keeps its own arXiv license).
set -euo pipefail
cd "$(dirname "$0")"
grep -v '^#' papers.tsv | while IFS=$'\t' read -r id name _; do
  out="${id}_${name}.pdf"
  if [ -s "$out" ]; then continue; fi
  curl -sSfL -m 180 -A "DashRecon literature collection" -o "$out" "https://export.arxiv.org/pdf/${id}"
  head -c 5 "$out" | grep -q "%PDF-" || { echo "not a PDF: $out" >&2; rm -f "$out"; exit 1; }
  echo "fetched $out ($(du -h "$out" | cut -f1))"
  sleep 3
done
