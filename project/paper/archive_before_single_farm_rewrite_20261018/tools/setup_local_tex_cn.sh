#!/usr/bin/env bash
# Project-local TeX packages. Does not install or modify system packages.
set -euo pipefail
paper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tex_deps="${paper_dir}/../.deps/tex-cn"
mkdir -p "${tex_deps}/packages" "${tex_deps}/root" "${tex_deps}/web2c"
cd "${tex_deps}/packages"
apt-get download texlive-xetex texlive-lang-chinese texlive-lang-cjk
for package in texlive-*.deb; do
  dpkg-deb -x "$package" "${tex_deps}/root"
done
export TEXMFHOME="${tex_deps}/root/usr/share/texlive/texmf-dist"
export TEXMFVAR="${tex_deps}/var"
export TEXMFCONFIG="${tex_deps}/config"
cd "${tex_deps}/web2c"
xetex -ini -etex -interaction=nonstopmode -halt-on-error -jobname=xelatex '\input xelatex.ini' > format_setup.log
printf '%s\n' "Local Chinese TeX prepared at ${tex_deps}"
