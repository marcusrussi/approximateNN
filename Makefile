TEST_EFILES := time_results test_correctness
EFILES := $(TEST_EFILES)
LIB_OFILES := algc.o rand_pr.o ann.o
TEST_OFILES := time_results.o test_correctness.o
OFILES := $(LIB_OFILES) randNorm.o $(TEST_OFILES)
FAKE_HFILES := time_results.h test_correctness.h
WARNS := -Wall -Wextra -Wpedantic
OS := $(shell uname -s)
# Toss the end of this next line if using OpenCL 2.x.
OCL_OPT := -DSUPPORT_OPENCL_V1_2

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

ann.o: algc.h

algc.h: ann.h
	touch $@

$(TEST_OFILES): ann.h randNorm.h

time_results.o: timing.h

time_results: time_results.o $(LIB_OFILES) randNorm.o
	cc -o $@ $^

test_correctness: test_correctness.o $(LIB_OFILES) randNorm.o
	cc -o $@ $^

.INTERMEDIATE: $(FAKE_HFILES)

$(FAKE_HFILES): %.h:
	touch $@

$(OFILES): %.o: %.c %.h ftype.h
	cc -c -g $(OSOPT) $(OCL_OPT) $(WARNS) $<
