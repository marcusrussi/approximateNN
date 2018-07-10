#ifndef ANN
#define ANN
#include <stddef.h>

typedef struct {
  int tries;
  size_t n, k, d_short, d_long, **which_par, *par_maxes, *graph;
  float *row_means, *bases;
} save_t;

extern size_t *query(const save_t *save, const float *points,
		     size_t ycnt, const float *y, char use_cpu);
extern size_t *precomp(size_t n, size_t k, size_t d, const float *points,
		       int tries, size_t rots_before, size_t rot_len_before,
		       size_t rots_after, size_t rot_len_after, save_t *save,
		       char use_cpu);
extern void free_save(save_t *save);
#endif
