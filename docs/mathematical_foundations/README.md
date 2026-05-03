# LLMs for Fun

Complete LaTeX source for **LLMs for Fun: From Zero to a State-of-the-Art LLM's Inference Engine**.

The project includes a conceptual guide for aligning the mathematical manuscript with the repository design and shared specifications.

## Build

```sh
make
```

The Makefile runs:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The extra final LaTeX pass is intentional: it leaves the table of contents, bibliography, long tables, and cross-references settled with a warning-free build log in this environment.

If your TeX installation requires `bibtex8`, use:

```sh
make BIBTEX=bibtex8
```

## Manual build

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Clean

```sh
make clean
```

## Contents

- `main.tex` - top-level book file.
- `llmbook.sty` - custom book style, theorem boxes, code listings, notation, and typography.
- `bibliography.bib` - BibTeX bibliography.
- `chapters/` - front matter, all parts, and appendices.
- `diagrams/` - TikZ diagrams.
- `Makefile` and `latexmkrc` - build helpers.
- Conceptual guide - design rationale and alignment notes for the repository.
