EFILES := test
OFILES := alg.o randNorm.o rand_pr.o test.o
FAKE_HFILES := test.h
OSOPT := -DOSX
WARNS := -Wall -Wextra -Wpedantic

all: $(EFILES)
clean: 
	rm -rf $(EFILES) $(OFILES)

alg.o: ocl2c.h compute.cl rand_pr.h

test.o: alg.h randNorm.h timing.h

test: $(OFILES)
	cc -o test -lm $(OFILES)

.INTERMEDIATE: $(FAKE_HFILES)

$(FAKE_HFILES): %.h:
	touch $@

$(OFILES): %.o: %.c %.h
	cc -c $(OSOPT) $(WARNS) $<
