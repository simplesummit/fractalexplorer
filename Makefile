## Makefile

dist_src = README.md LICENSE Makefile src/Makefile src/*.c src/*.h src/*.cu

PROG_NAME = small-summit-demo


obj:
	$(MAKE) -C src

clean:
	$(MAKE) -C src clean

dist-gzip: $(dist_src)
	tar cJf $(PROG_NAME).tar.gz $^
	@echo Created archive $(PROG_NAME).tar.gz


