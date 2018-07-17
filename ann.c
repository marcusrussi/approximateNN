#include "ann.h"
#include "algc.h"
#include <assert.h>
#include <stdlib.h>

size_t *query(const save_t *save, const ftype *points,
	      size_t ycnt, const ftype *y, char use_cpu) {
  assert(use_cpu && "CPU usage required if compiled for CPU only");
  return(query_cpu(save, points, ycnt, y));
}
size_t *precomp(size_t n, size_t k, size_t d, const ftype *points,
		int tries, size_t rots_before, size_t rot_len_before,
		size_t rots_after, size_t rot_len_after, save_t *save,
		char use_cpu) {
  assert(use_cpu && "CPU usage required if compiled for CPU only");
  return(precomp_cpu(n, k, d, points, tries, rots_before, rot_len_before,
		     rots_after, rot_len_after, save));
}

void free_save(save_t *save) {
  for(int i = 0; i < save->tries; i++) {
    free(save->which_par[i]);
  }
  free(save->which_par);
  free(save->par_maxes);
  free(save->graph);
  free(save->row_means);
  free(save->bases);
}
