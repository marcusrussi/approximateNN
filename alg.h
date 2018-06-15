#ifndef ALG
#define ALG
#include <stddef.h>

typedef struct {
  int tries;
  size_t n, k, d_short, d_long, **which_par, *par_maxes, *graph;
  double *row_means, *bases;
} save_t;

extern size_t *query(const save_t *save, const double *points,
		     size_t ycnt, double *y);
extern size_t *precomp(size_t n, size_t k, size_t d, double *points,
		       int tries, size_t rots_before, size_t rot_len_before,
		       size_t rots_after, size_t rot_len_after, save_t *save);
extern void free_save(save_t *save);
#endif
