## Makefile

PROG_NAME = small-summit-demo

dist_src = README.md LICENSE Makefile


install: requirements.txt
	pip install -r $<


dist-gzip: $(dist_src)
	tar cJvf $(PROG_NAME).tar.gz $<
	@echo Created archive $(PROG_NAME).tar.gz
