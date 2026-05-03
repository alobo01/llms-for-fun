PDF=main.pdf
LATEX=pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX?=bibtex

all: $(PDF)

$(PDF): main.tex llmbook.sty bibliography.bib chapters/*.tex diagrams/*.tex
	$(LATEX) main.tex
	$(BIBTEX) main
	$(LATEX) main.tex
	$(LATEX) main.tex
	$(LATEX) main.tex

clean:
	rm -f *.aux *.bbl *.blg *.brf *.idx *.ilg *.ind *.lof *.log *.lot *.out *.toc *.fls *.fdb_latexmk *.synctex.gz $(PDF)

.PHONY: all clean
