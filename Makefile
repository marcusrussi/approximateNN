TEST_EFILES := time_results test_correctness
EFILES := $(TEST_EFILES) compare_results
LIB_OFILES := algc.o rand_pr.o algg.o ann.o
TEST_OFILES := time_results.o test_correctness.o compare_results.o
OFILES := $(LIB_OFILES) randNorm.o $(TEST_OFILES)
FAKE_HFILES := time_results.h test_correctness.h compare_results.h
OSOPT := -DOSX
WARNS := -Wall -Wextra -Wpedantic

.PHONY: clean

all: $(EFILES)
clean: 
	rm -rf $(EFILES) $(OFILES)

algc.o: ocl2c.h compute.cl rand_pr.h ann.h
algg.o: ann.h rand_pr.h

ann.o: algc.h algg.h

algc.h algg.h: ann.h
	touch $@

time_results.o test_correctness.o compare_results.o: ann.h randNorm.h

time_results.o: timing.h

time_results: time_results.o $(LIB_OFILES) randNorm.o
	cc -o $@ -lm time_results.o $(LIB_OFILES) randNorm.o

test_correctness: test_correctness.o $(LIB_OFILES) randNorm.o
	cc -o $@ -lm $^

compare_results: compare_results.o $(LIB_OFILES) randNorm.o
	cc -o $@ -lm $^

.INTERMEDIATE: $(FAKE_HFILES)

$(FAKE_HFILES): %.h:
	touch $@

$(OFILES): %.o: %.c %.h
	cc -c $(OSOPT) $(WARNS) $<
