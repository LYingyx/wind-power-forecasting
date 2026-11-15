# Paper artifacts

This directory is the canonical location for the paper source and generated PDF.

- `make` or `make pdf`: incrementally build `main.pdf` with `latexmk`.
- `make force`: rebuild from scratch and converge all cross-references.
- `make check`: verify that the PDF is non-empty, newer than `main.tex`, has at least one page, and has no undefined references or citations.
- `make watch`: keep rebuilding while `main.tex` changes.
- `make clean`: remove intermediate LaTeX files while preserving `main.pdf`.
- `make distclean`: remove intermediate files and `main.pdf`.

The paper currently reports 10-seed results on the 2023 chronological validation split. The previously inspected 2024 split is intentionally not presented as an independent test set.
