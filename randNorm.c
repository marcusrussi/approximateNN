#include "randNorm.h"
#include <stdlib.h>
#include <limits.h>
#include <math.h>
static const double max_ulong_p1 = (double)ULONG_MAX + 1;

#define rand_dbl() ((double)(unsigned long)random() / max_ulong_p1)

static double next = NAN;

double rand_norm(void) {
  if(isnan(next)) {
    double u1 = sqrt(log(rand_dbl()) * -2), u2 = rand_dbl() * M_PI * 2;
    next = u1 * sin(u2);
    return(u1 * cos(u2));
  } else {
    double d = next;
    next = NAN;
    return(d);
  }
}
