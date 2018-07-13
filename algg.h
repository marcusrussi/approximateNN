#ifndef ALGGPU
#define ALGGPU
#include "ann.h"

extern size_t *query_gpu(const save_t *save, const ftype *points,
			 size_t ycnt, const ftype *y);
extern size_t *precomp_gpu(size_t n, size_t k, size_t d, const ftype *points,
			   int tries, size_t rots_before,
			   size_t rot_len_before, size_t rots_after,
			   size_t rot_len_after, save_t *save);
#endif
