first: 
	pdflatex -shell-escape main.tex

main: first
	bibtex main && pdflatex -shell-escape main.tex

omain: main
	open main.pdf

clean:
	rm -f *.aux main.log main.out main.blg main.bbl
