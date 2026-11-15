# Compile Instructions

This project uses XeLaTeX to compile the IEEE LaTeX manuscripts. Compilation must use the shared Windows-standard font library so that the English manuscript uses Times New Roman rather than substitute fonts.

## Compiler and Font Environment

```bash
/storage/luoao/texlive/2025/bin/x86_64-linux/latexmk
/storage/luoao/fonts/ustc-fontconfig.conf
```

## Environment Variables

Use the shared fontconfig and TeX Live paths when compiling:

```bash
FONTCONFIG_FILE=/storage/luoao/fonts/ustc-fontconfig.conf
PATH=/storage/luoao/texlive/2025/bin/x86_64-linux:$PATH
```

## Compile Chinese Version

```bash
cd /storage/luoao/GNN/DP_NEW/dayahead/paper/ieee_bigdia_zh

FONTCONFIG_FILE=/storage/luoao/fonts/ustc-fontconfig.conf \
PATH=/storage/luoao/texlive/2025/bin/x86_64-linux:$PATH \
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -file-line-error -outdir=build manuscript_zh_short.tex
```

Output:

```text
/storage/luoao/GNN/DP_NEW/dayahead/paper/ieee_bigdia_zh/build/manuscript_zh_short.pdf
```

## Compile English Version

```bash
cd /storage/luoao/GNN/DP_NEW/dayahead/paper/ieee_bigdia_zh_en

FONTCONFIG_FILE=/storage/luoao/fonts/ustc-fontconfig.conf \
PATH=/storage/luoao/texlive/2025/bin/x86_64-linux:$PATH \
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -file-line-error -outdir=build manuscript_en_short.tex
```

Output:

```text
/storage/luoao/GNN/DP_NEW/dayahead/paper/ieee_bigdia_zh_en/build/manuscript_en_short.pdf
```

## Check Page Count

```bash
pdfinfo /storage/luoao/GNN/DP_NEW/dayahead/paper/ieee_bigdia_zh/build/manuscript_zh_short.pdf
pdfinfo /storage/luoao/GNN/DP_NEW/dayahead/paper/ieee_bigdia_zh_en/build/manuscript_en_short.pdf
```

Current compiled page counts:

```text
Chinese short version: 7 pages
English short version: 7 pages
```

## Check Embedded Fonts

```bash
pdffonts build/manuscript_en_short.pdf
```

The English manuscript text should use `TimesNewRoman`, with `Arial` and `CourierNew` available for sans-serif and monospaced text. `Noto`, `Fandol`, and `Liberation` must not be used as manuscript font substitutes.

## Expected Warnings

The current build may report `Underfull \hbox` or `Underfull \vbox` warnings, mainly from long URLs, references, and narrow IEEE columns. These warnings do not indicate missing references or LaTeX compilation failure.
