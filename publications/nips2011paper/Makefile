all: draft1.pdf poster.pdf

clean:
	rm *.bbl *.aux *.blg *.log
	rm draft1.pdf

draft1.pdf: draft1.tex figures/* *.bib
	pdflatex draft1.tex
	bibtex -min-crossrefs=999 draft1
	pdflatex draft1.tex
	pdflatex draft1.tex

copy2articles: draft1.tex *.bib *.sty
	svn up
	cp Makefile *.tex.tex *.bib *.sty ~/articles/2011/hyper_opt/
	cp figures/*.* ~/articles/2011/hyper_opt/figures
	cp figures/dbn_efficiency/*.* ~/articles/2011/hyper_opt/figures/dbn_efficiency
	cd ~/articles/2011/hyper_opt; svn commit -m "new version copied";

poster.pdf: poster.tex lriposter.sty
	pdflatex poster
