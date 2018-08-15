#ifndef ANN
#define ANN
#include <stddef.h>
#include "ftype.h"

typedef struct {
  int tries;
  size_t n, k, d_short, d_long, **which_par, *par_maxes, *graph;
  ftype *row_means, *bases;
} save_t;

extern size_t *query(const save_t *save, const ftype *points,
		     size_t ycnt, const ftype *y, ftype **dists, char use_cpu);
extern size_t *precomp(size_t n, size_t k, size_t d, const ftype *points,
		       int tries, size_t rots_before, size_t rot_len_before,
		       size_t rots_after, size_t rot_len_after, save_t *save,
		       ftype **dists, char use_cpu);
extern void free_save(save_t *save);
#endif
