#include "rand_pr.h"
#include <stdlib.h>
#include <limits.h>

static const double max_long_p1 = (double)ULONG_MAX + 1;

#define rand_dbl() ((double)(unsigned long)random() / max_long_p1)

void rand_rot(size_t rotlen, size_t d,
	      size_t *is, size_t *js, double *as) {
  size_t *arr = rand_perm(rotlen * 2, d);
  for(size_t i = 0; i < rotlen; i++)
    is[i] = arr[2 * i], js[i] = arr[2 * i + 1], as[i] = rand_dbl();
  free(arr);
}
size_t *rand_perm(size_t d_pre, size_t d_post) {
  size_t *perm = malloc(sizeof(size_t) * d_post);
  for(size_t i = 0; i < d_post; i++)
    perm[i] = i;
  for(size_t i = 0; i < d_pre; i++) {
    size_t j = (unsigned long)random() % (d_post - i) + i;
    if(j != i) {
      size_t t = perm[i];
      perm[i] = perm[j];
      perm[j] = t;
    }
  }
  return(perm);
}
