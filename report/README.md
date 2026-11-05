# Thesis preview and build

Open `report/main.tex` in VS Code with LaTeX Workshop installed, then run the
**latexmk (pdflatex)** recipe. Workspace settings enable automatic builds on file
changes and the PDF viewer in a tab. The preamble, front matter and numbered
sections identify `main.tex` as their root, so they can also be edited directly.
The PDF and SyncTeX data are generated in `report/build/`; use LaTeX Workshop's
forward and reverse SyncTeX navigation between the source and preview.

For a terminal build, run from `report/`:

```sh
latexmk -pdf -interaction=nonstopmode -file-line-error main.tex
```

The build uses pdfLaTeX, BibTeX, `IEEEtran.bst` and `references.bib`. Install
latexmk, the TeX commands and the LaTeX packages used by the document. The local
`IEEEtran.cls` is retained in this directory. `latexmk -C main.tex` removes the
generated PDF as well as auxiliary files; rebuild afterwards to restore the
tracked `build/main.pdf`. If migrating from an older Biber build, also remove
any stale `build/main.bbl` before rebuilding.
