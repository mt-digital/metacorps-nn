first: 
	xelatex -shell-escape ms.tex

main: first
	bibtex ms && xelatex -shell-escape ms.tex

omain: main
	open ms.pdf

clean:
	rm -f *.aux ms.log ms.out ms.blg ms.bbl
