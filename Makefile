TEST_EFILES := time_results test_correctness
EFILES := $(TEST_EFILES) compare_results
LIB_OFILES := algc.o rand_pr.o algg.o ann.o gpu_comp.o
TEST_OFILES := time_results.o test_correctness.o compare_results.o
OFILES := $(LIB_OFILES) randNorm.o $(TEST_OFILES)
FAKE_HFILES := time_results.h test_correctness.h compare_results.h
WARNS := -Wall -Wextra -Wpedantic
OS := $(shell uname -s)

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
algg.o: ann.h rand_pr.h gpu_comp.h alg.c

ann.o: algc.h algg.h

algc.h algg.h: ann.h
	touch $@

$(TEST_OFILES): ann.h randNorm.h gpu_comp.h

time_results.o: timing.h

time_results: time_results.o $(LIB_OFILES) randNorm.o
	cc -o $@ -lm -lOpenCL $^

test_correctness: test_correctness.o $(LIB_OFILES) randNorm.o
	cc -o $@ -lm -lOpenCL $^

compare_results: compare_results.o $(LIB_OFILES) randNorm.o
	cc -o $@ -lm -lOpenCL $^

.INTERMEDIATE: $(FAKE_HFILES)

$(FAKE_HFILES): %.h:
	touch $@

$(OFILES): %.o: %.c %.h
	cc -c -g $(OSOPT) $(WARNS) $<
