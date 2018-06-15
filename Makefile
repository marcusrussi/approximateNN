EFILES := time_results
OFILES := algc.o randNorm.o rand_pr.o time_results.o algg.o ann.o
FAKE_HFILES := time_results.h
OSOPT := -DOSX
WARNS := -Wall -Wextra -Wpedantic

all: $(EFILES)
clean: 
	rm -rf $(EFILES) $(OFILES)

algc.o: ocl2c.h compute.cl rand_pr.h ann.h
algg.o: ann.h rand_pr.h

ann.c: algc.h algg.h

algc.h algg.h: ann.h
	touch $@

time_results.o: ann.h randNorm.h timing.h

time_results: $(OFILES)
	cc -o $@ -lm $(OFILES)

.INTERMEDIATE: $(FAKE_HFILES)

$(FAKE_HFILES): %.h:
	touch $@

$(OFILES): %.o: %.c %.h
	cc -c $(OSOPT) $(WARNS) $<
