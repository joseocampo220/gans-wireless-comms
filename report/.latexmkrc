# Keep all generated files out of the source directory.
# This affects manual `latexmk` runs from within the report/ folder.
$out_dir = 'build';
$aux_dir = 'build';

# Use pdfLaTeX with SyncTeX; latexmk invokes BibTeX for the IEEE bibliography.
$pdflatex = 'pdflatex -synctex=1 %O %S';
