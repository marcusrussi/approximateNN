#include <math.h>
#include <stdlib.h>
#include "ocl2c.h"
#include "compute.cl"
#include "rand_pr.h"
#include "alg.h"
#include <string.h>

#define max(a, b) ((a) < (b)?b:(a))

unsigned lg(size_t d) {
  unsigned r = (d > 0xFFFFFFFF) << 5;
  d >>= r;
  unsigned s = (d > 0xFFFF) << 4;
  d >>= s, r |= s;
  d >>= s = (d > 0xFF) << 3, r |= s;
  d >>= s = (d > 0xF) << 2, r |= s;
  d >>= s = (d > 3) << 1;
  return(r | s | d >> 1);
}

void add_up_rows(size_t d, size_t n, double *points, double *sums) {
  LOOP2(add_rows_step_0(d, n, points, sums), n/2, d);
  for(size_t m = n >> 1; m >> 1; m >>= 1)
    LOOP2(add_rows_step_n(d, m, sums), m/2, d);
}

void walsh(size_t d, size_t n, double *a) {
  if(d == 1)
    return;
  int l = lg(d);
  size_t nth = max(d / 16, 1);
  LOOP2(apply_walsh_step(l, 0, a), n, nth);
  for(int i = 1; i < l; i++)
    LOOP2(apply_walsh_step(l, i, a), n, nth);
}

void add_up_cols(size_t d, size_t k, size_t skip, size_t n,
		 double *mat, double *out) {
  LOOP3(add_cols_step(d, d, k - skip, mat), n, k - skip, d / 2);
  for(size_t l = d / 2; l >> 1; l >>= 1)
    LOOP3(add_cols_step(d, l, k - skip, mat), n, k - skip, l / 2);
  LOOP2(add_cols_fin(d, k, skip, mat, out), n, k - skip);
}

void do_sort(size_t k, size_t n, size_t *along, double *order) {
  int lk = lg(k);
  size_t nth = (size_t)1 << max(lk - 4, 0);
  for(int s = 1; s < lk; s++)
    for(int ss = s; ss >= 0; ss--)
      LOOP2(sort_two_step(k, n, s, ss, along, order), n, nth);
}


/* Starting point: */
/* We have an array, points, that is n by d_long. */
/* We also have save, which is a save structure. */

size_t *precomp(size_t n, size_t k, size_t d, double *points,
		int tries, size_t rots_before, size_t rot_len_before,
		size_t rots_after, size_t rot_len_after, save_t *save) {
  size_t d_short = ceil(log2((double)n / k));
  size_t d_max = d - 1;
  d_max |= d_max >> 1;
  d_max |= d_max >> 2;
  d_max |= d_max >> 4;
  d_max |= d_max >> 8;
  d_max |= d_max >> 16;
  d_max |= d_max >> 32;
  d_max++;
  {
    double *pnts = malloc(sizeof(double) * n * d);
    memcpy(pnts, points, sizeof(double) * n * d);
    points = pnts;
  }
  double *row_sums = malloc(sizeof(double) * (n/2) * d);
  add_up_rows(d, n, points, row_sums);
  LOOP1(divide_by_length(n, row_sums), d);
  LOOP2(subtract_off(d, points, row_sums), n, d);
  if(save != NULL) {
    save->tries = tries;
    save->n = n;
    save->k = k;
    save->d_short = d_short;
    save->d_long = d;
    save->points = points;
    save->row_means = malloc(sizeof(double) * d);
    memcpy(save->row_means, row_sums, sizeof(double) * d);
    save->which_par = malloc(sizeof(size_t *) * tries);
    save->par_maxes = malloc(sizeof(size_t) * tries);
    save->bases = malloc(sizeof(double) * tries * d_short * d);
  }
  free(row_sums);
  size_t *pointers_out = malloc(sizeof(size_t) * (n * k * tries + 1));
  double *dists_out = malloc(sizeof(double) * (n * k * tries + 1));
  for(int i = 0; i < tries; i++) {
    size_t *rot_is_b = malloc(sizeof(size_t) * rots_before * rot_len_before);
    size_t *rot_js_b = malloc(sizeof(size_t) * rots_before * rot_len_before);
    double *rot_as_b = malloc(sizeof(double) * rots_before * rot_len_before);
    size_t *rot_is_a = malloc(sizeof(size_t) * rots_after * rot_len_after);
    size_t *rot_js_a = malloc(sizeof(size_t) * rots_after * rot_len_after);
    double *rot_as_a = malloc(sizeof(double) * rots_after * rot_len_after);
    for(size_t j = 0; j < rots_before * rot_len_before; j += rot_len_before) {
    	  rand_rot(rot_len_before, d,
		   rot_is_b + j, rot_js_b + j, rot_as_b + j);
    }
    for(size_t j = 0; j < rots_after * rot_len_after; j += rot_len_after) {
    	  rand_rot(rot_len_after, d_max, rot_is_a + j,
	           rot_js_a + j, rot_as_a + j);
    }
    size_t *perm_before = rand_perm(d, d_max);
    size_t *perm_after_i = rand_perm(d_short, d_max);
    double *pc = malloc(sizeof(double) * n * d);
    double *pc2 = malloc(sizeof(double) * n * d_max);
    LOOP2(copy_some_floats(d, d, 0, points, pc), n, d);
    for(size_t j = 0; j < rots_before; j++)        
      LOOP2(apply_rotation(d, rot_is_b + j * rot_len_before,
	                   rot_js_b + j * rot_len_before,
			   rot_as_b + j * rot_len_before, pc),
	                n, rot_len_before);
    LOOP2(apply_permutation(d, d_max, perm_before, pc, pc2), n, d_max);
    free(pc);
    walsh(d_max, n, pc2);
    for(size_t j = 0; j < rots_after; j++)
      LOOP2(apply_rotation(d, rot_is_a + j * rot_len_after,
	                   rot_js_a + j * rot_len_after,
			   rot_as_a + j * rot_len_after, pc2),
	    n, rot_len_after);
    pc = malloc(sizeof(double) * n * d_short + 1);
    LOOP2(apply_perm_inv(d_max, d_short, n * d_short, perm_after_i, pc2, pc),
	     n, d_max);
    free(pc2);
    size_t *signs = malloc(sizeof(size_t) * n);
    LOOP1(compute_signs(d_short, (long *)pc, signs), n);
    free(pc);
    if(save) {
      double *vecs = malloc(sizeof(double) * (d_short * d + 1));
      double *vecs2 = malloc(sizeof(double) * d_short * d_max);
      for(size_t j = 0; j < d_short; j++)
	for(size_t l = 0; l < d_short; l++)
	  vecs[j * d_short + l] = l == j;
      LOOP2(apply_permutation(d_short, d_max, perm_after_i, vecs, vecs2),
			d_short, d_max);
      for(long j = rots_after - 1; j >= 0; j--)
	LOOP2(apply_rotation(d_max, rot_js_a + j * rot_len_after,
			     rot_is_a + j * rot_len_after,
			     rot_as_a + j * rot_len_after, vecs2),
	      d_short, rot_len_after);
      walsh(d_max, d_short, vecs2);
      LOOP2(apply_perm_inv(d_max, d, d_short * d, perm_after_i,
			   vecs2, vecs), d_short, d_max);
      free(vecs2);
      for(long j = rots_before - 1; j >= 0; j--)
	LOOP2(apply_rotation(d, rot_js_b + j * rot_len_before,
			     rot_is_b + j * rot_len_before,
			     rot_as_b + j * rot_len_before, vecs2),
	      d_short, rot_len_before);
      memcpy(save->bases + i * d_short * d, vecs,
	     sizeof(double) * d_short * d);
      free(vecs);
    }
    free(rot_is_b);
    free(rot_js_b);
    free(rot_as_b);
    free(rot_is_a);
    free(rot_js_a);
    free(rot_as_a);
    free(perm_before);
    free(perm_after_i);
    size_t *counts = malloc(sizeof(size_t) << d_short);
    for(size_t j = 0; j < 1 << d_short; j++)
        counts[j] = 0;
    for(size_t j = 0; j < n; j++)
	counts[signs[j]]++;
    size_t tmax = counts[0];
    for(size_t j = 1; j < 1 << d_short; j++)
        if(tmax < counts[j])
	    tmax = counts[j];
    size_t *which = malloc(sizeof(size_t) * tmax << d_short);
    if(save != NULL) {
        save->which_par[i] = which;
	save->par_maxes[i] = tmax;
    }
    for(size_t j = 0; j < 1 << d_short; j++)
	for(size_t l = counts[j]; l < tmax; l++)
	    which[j * tmax + l] = n;
    for(size_t j = 0; j < n; j++)
        which[j * tmax + --counts[signs[j]]] = j;
    free(counts);
    size_t *which_d = malloc(sizeof(size_t) * ((d_short + 1) * n * tmax + 1));
    double *dists = malloc(sizeof(double) * (((d_short + 1) * n * tmax + 1)));
    double *diffs = malloc(sizeof(double) * (d_short + 1) * n * d * tmax);
    LOOP3(compute_which(d_short, tmax, signs, which, which_d),
	     n, d_short + 1, tmax);
    LOOP3(compute_diffs_squared(d, (d_short + 1) * tmax, n, 0,
	                           which_d, points, points, diffs),
	     n, (d_short + 1) * tmax, d);
    add_up_cols(d, (d_short + 1) * tmax, 0, n, diffs, dists);
    free(diffs);
    do_sort((d_short + 1) * tmax, n, which_d, dists);
    LOOP2(rdups((d_short + 1) * tmax, which_d, dists),
	       n, (d_short + 1) * tmax - 1);
    do_sort((d_short + 1) * tmax, n, which_d, dists);
    LOOP2(copy_some_ints((d_short + 1) * tmax, k * tries, k * i,
	                      which_d, pointers_out), n, k);
    LOOP2(copy_some_floats((d_short + 1) * tmax, k * tries, k * i,
			   dists, dists_out), n, k);
    if(save != NULL)
        free(which);
    free(signs);
    free(counts);
    free(which_d);
    free(diffs);
  }
  do_sort(k * tries, n, pointers_out, dists_out);
  LOOP2(rdups(k * tries, pointers_out, dists_out), n, k * tries - 1);
  do_sort(k * tries, n, pointers_out, dists_out);
  size_t *nedge = malloc(sizeof(size_t) * (n * k * (k + 1) + 1));
  double *ndists = malloc(sizeof(double) * (n * k * (k + 1) + 1));
  LOOP3(supercharge(n, k * tries, k, pointers_out, pointers_out, nedge),
	n, k, k);
  LOOP2(copy_some_ints(k * tries, k * (k + 1), 0, pointers_out, nedge),
	n, k);
  LOOP2(copy_some_floats(k * tries, k * (k + 1), 0, dists_out, ndists),
	n, k);
  free(pointers_out);
  free(dists_out);
  double *diffs = malloc(sizeof(double) * n * k * k * d);
  LOOP3(compute_diffs_squared(d, k * (k + 1), n, k,
			      nedge, points, points, diffs),
	n, k * k, d);
  add_up_cols(d, k * (k + 1), k, n, diffs, ndists);
  free(diffs);
  do_sort(k * (k + 1), n, nedge, ndists);
  LOOP2(rdups(k * (k + 1), nedge, ndists), n, k * (k + 1) - 1);
  do_sort(k * (k + 1), n, nedge, ndists);
  free(ndists);
  size_t *fedges = malloc(sizeof(size_t) * n * k);
  if(save)
    save->graph = fedges;
  LOOP2(copy_some_ints(k * (k + 1), k, 0, nedge, fedges), n, k);
  free(nedge);
  return(fedges);
}

// We now have save->points (n by d_long), save->graph (n by k),
// save->row_means (d_long), save->par_maxes (tries),
// save->which_par (tries, then 1 << d_short by save->par_maxes[i]),
// save->bases (tries by d_short by d_long), y (ycnt by d_long).

size_t *query(const save_t *save, size_t ycnt, double *y) {
  double *cprds = malloc(sizeof(double) * save->tries *
			 save->d_short * save->d_long);
  double *dprds = malloc(sizeof(double) * save->tries * save->d_short);
  LOOP2(subtract_off(save->d_long, y, save->row_means), ycnt, save->d_long);
  LOOP3(prods(save->d_long, save->n, y, save->bases, cprds),
	ycnt, save->tries * save->d_short, save->d_long);
  add_up_cols(save->d_long, save->d_short, 0, save->tries * ycnt,
	      cprds, dprds);
  free(cprds);
  size_t *pmaxes = malloc(sizeof(size_t) * save->tries);
  size_t msofar = 0;
  for(int i = 0; i < save->tries; i++) {
    pmaxes[i] = msofar;
    msofar += save->par_maxes[i];
  }
  size_t *signs = malloc(sizeof(size_t) * save->tries);
  LOOP1(compute_signs(save->d_short, (long *)dprds, signs), save->tries * ycnt);
  free(dprds);
  size_t *ppts = malloc(sizeof(size_t) *
			(msofar * (save->d_short + 1) * ycnt + 1));
  size_t *ipts = malloc(sizeof(size_t) *
			(msofar * (save->d_short + 1) * ycnt + 1));

  for(int i = 0; i < save->tries; i++) {
    LOOP3(compute_which(save->d_short, save->par_maxes[i],
			signs + i, save->which_par[i],
			ppts + pmaxes[i] * (save->d_short + 1) * ycnt),
	  ycnt, save->d_short + 1, save->par_maxes[i]);
    LOOP2(copy_some_ints(save->par_maxes[i] * (save->d_short + 1),
			 msofar * (save->d_short + 1),
			 pmaxes[i] * (save->d_short + 1),
			 ppts + pmaxes[i] * (save->d_short + 1) * ycnt,
			 ipts),
	  ycnt, save->par_maxes[i] * (save->d_short + 1));
  }
  free(ppts);
  free(pmaxes);
  free(signs);
  double *dpts = malloc(sizeof(double) *
			(msofar * (save->d_short + 1) * ycnt + 1));
  double *diffs = malloc(sizeof(double) *
			 msofar * (save->d_short + 1) * save->d_long * ycnt);
  LOOP3(compute_diffs_squared(save->d_long, msofar * (save->d_short + 1),
			      save->n, 0, ipts, y, save->points, diffs),
	ycnt, msofar * (save->d_short + 1), save->d_long);
  add_up_cols(save->d_long, msofar * (save->d_short + 1), 0, ycnt,
	      diffs, dpts);
  free(diffs);
  do_sort(msofar * (save->d_short + 1), ycnt, ipts, dpts);
  LOOP2(rdups(msofar * (save->d_short + 1), ipts, dpts),
	ycnt, msofar * (save->d_short + 1) - 1);
  do_sort(msofar * (save->d_short + 1), ycnt, ipts, dpts);
  {
    size_t *ipts2 = malloc(sizeof(size_t) *
			   (save->k * (save->k + 1) * ycnt + 1));
    LOOP2(copy_some_ints(save->k, save->k, 0, ipts, ipts2),
	  ycnt, save->k);
    free(ipts);
    ipts = ipts2;
    double *dpts2 = malloc(sizeof(double) *
			   (save->k * (save->k + 1) * ycnt + 1));
    LOOP2(copy_some_floats(save->k, save->k, 0, dpts, dpts2),
	  ycnt, save->k);
    free(dpts);
    dpts = dpts2;
  }
  LOOP3(supercharge(save->n, 0, save->k, ipts, save->graph, ipts),
	ycnt, save->k, save->k);
  diffs = malloc(sizeof(double) * save->k * save->k * save->d_long * ycnt);
  
  LOOP3(compute_diffs_squared(save->d_long, save->k * (save->k + 1), save->n,
			      save->k, ipts, y, save->points, diffs),
	ycnt, save->k * save->k, save->d_long);
  add_up_cols(save->d_long, save->k * (save->k + 1), save->k, ycnt,
	      diffs, dpts);
  free(diffs);
  do_sort(save->k * (save->k + 1), ycnt, ipts, dpts);
  LOOP2(rdups(save->k * (save->k + 1), ipts, dpts),
	ycnt, save->k * (save->k + 1) - 1);
  do_sort(save->k * (save->k + 1), ycnt, ipts, dpts);
  free(dpts);
  size_t *results = malloc(sizeof(size_t) * ycnt * save->k);
  LOOP2(copy_some_ints(save->k * (save->k + 1), save->k, 0, ipts, results),
	ycnt, save->k);
  free(ipts);
  return(results);
}
