#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p build
tex_deps="$(pwd)/../.deps/tex-cn"
if command -v xelatex >/dev/null 2>&1 && kpsewhich ctexart.cls >/dev/null; then
  latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
else
  export TEXMFHOME="${tex_deps}/root/usr/share/texlive/texmf-dist"
  export TEXMFVAR="${tex_deps}/var"
  export TEXMFCONFIG="${tex_deps}/config"
  test -f "${tex_deps}/web2c/xelatex.fmt"
  for pass in 1 2 3; do
    xetex -fmt="${tex_deps}/web2c/xelatex.fmt" -progname=xelatex -no-shell-escape \
      -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex
  done
fi
cp build/main.pdf manuscript.pdf
cp build/main.pdf manuscript_cn.pdf
