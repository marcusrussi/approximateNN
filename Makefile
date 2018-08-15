TEST_EFILES := time_results test_correctness
EFILES := $(TEST_EFILES) compare_results
LIB_OFILES := algc.o rand_pr.o ann.o gpu_comp.o algg.o
TEST_OFILES := time_results.o test_correctness.o compare_results.o
OFILES := $(LIB_OFILES) randNorm.o $(TEST_OFILES)
FAKE_FILES := time_results.h test_correctness.h compare_results.h algg.c
WARNS := -Wall -Wextra -Wpedantic -Wno-unused-parameter
OS := $(shell uname -s)
# Toss the end of this next line if using OpenCL 2.x.
OCL_OPT := 

ifeq ("$(OS)", "Darwin")
	OSOPT := -DOSX
else
	OSOPT := -DLINUX
endif

.PHONY: clean

all: $(EFILES)
clean: 
	rm -rf $(EFILES) $(OFILES) *.dSYM

algc.o: ocl2c.h compute.cl rand_pr.h ann.h alg.c

algg.c: alggp.c compute.cl
	sed <compute.cl >foo -e 's/.*/"&\\n"/' -e '$$!s/$$/,/'
	sed <alggp.c >algg.c -e '/INSERT_COMP_HERE/r foo' \
			     -e '/INSERT_COMP_HERE/d' \
			     -e "s/LINE_COUNT_OCL/$(shell wc -l <compute.cl)/"
	rm foo

algg.o: alg.c ann.h rand_pr.h gpu_comp.h

ann.o: algc.h algg.h

algc.h algg.h: ann.h
	touch $@

$(TEST_OFILES): ann.h randNorm.h gpu_comp.h

time_results.o: timing.h

time_results: time_results.o $(LIB_OFILES) randNorm.o
	cc -o $@ $^ -lOpenCL -lm

test_correctness: test_correctness.o $(LIB_OFILES) randNorm.o
	cc -o $@ $^ -lOpenCL -lm

compare_results: compare_results.o $(LIB_OFILES) randNorm.o
	cc -o $@ $^ -lOpenCL -lm

.INTERMEDIATE: $(FAKE_FILES)

$(filter %.h,$(FAKE_FILES)): %.h:
	touch $@

$(OFILES): %.o: %.c %.h ftype.h
	clang -c -g $(OSOPT) $(OCL_OPT) $(WARNS) $<
