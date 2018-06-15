#ifndef ALGGPU
#define ALGGPU
#include "ann.h"

extern size_t *query_gpu(const save_t *save, const double *points,
			 size_t ycnt, double *y);
extern size_t *precomp_gpu(size_t n, size_t k, size_t d, double *points,
			   int tries, size_t rots_before,
			   size_t rot_len_before, size_t rots_after,
			   size_t rot_len_after, save_t *save);
#endif
