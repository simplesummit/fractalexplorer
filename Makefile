## Makefile

dist_src = README.md LICENSE Makefile

PROG_NAME = small-summit-demo


obj:
	$(MAKE) -C src


clean:
	$(MAKE) -C src clean

dist-gzip: $(dist_src)
	tar cJvf $(PROG_NAME).tar.gz $<
	@echo Created archive $(PROG_NAME).tar.gz


