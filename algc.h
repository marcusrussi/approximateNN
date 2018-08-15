#ifndef ALGCPU
#define ALGCPU
#include "ann.h"

extern size_t *query_cpu(const save_t *save, const ftype *points,
			 size_t ycnt, const ftype *y, ftype **dists_o);
extern size_t *precomp_cpu(size_t n, size_t k, size_t d, const ftype *points,
			   int tries, size_t rots_before,
			   size_t rot_len_before, size_t rots_after,
			   size_t rot_len_after, save_t *save,
			   ftype **dists_o);

#endif
