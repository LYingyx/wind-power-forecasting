#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p build
# Standard TeX Live is preferred. The fallback is local to this workstation.
if command -v xelatex >/dev/null 2>&1 && kpsewhich ctexart.cls >/dev/null; then
  for pass in 1 2 3; do
    xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error \
      -file-line-error -output-directory=build main.tex
  done
else
  tex_deps="${WIND_PAPER_TEX_DEPS:-/home/liying/codex/.deps/tex-cn}"
  if [[ ! -f "${tex_deps}/web2c/xelatex.fmt" ]]; then
    echo 'Install XeLaTeX with ctex/Fandol, or set WIND_PAPER_TEX_DEPS to the local TeX dependency directory.' >&2
    exit 1
  fi
  export TEXMFHOME="${tex_deps}/root/usr/share/texlive/texmf-dist"
  export TEXMFVAR="${tex_deps}/var"
  export TEXMFCONFIG="${tex_deps}/config"
  for pass in 1 2 3; do
    xetex -fmt="${tex_deps}/web2c/xelatex.fmt" -progname=xelatex -no-shell-escape \
      -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex
  done
fi
cp build/main.pdf manuscript.pdf
cp build/main.pdf manuscript_cn.pdf
