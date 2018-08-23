#include <string.h>

#define conc(a, b, c) a ## b ## c
#define concb(a, b, c) conc(a, b, c)
#define MK_NAME(x) concb(x, _, TYPE_OF_COMP)
// hack to allow two versions of a function to be defined in separate files
// with separate names, but only slightly different.

#define rot_info MK_NAME(rot_info_int)
#define ortho_info MK_NAME(ortho_info_int)

typedef struct {
  BUFTYPE(size_t) *is;
  BUFTYPE(size_t) *js;
  BUFTYPE(ftype) *as;
} rot_info;

typedef struct {
  rot_info rb, ra;
  BUFTYPE(size_t) perm_b;
  BUFTYPE(size_t) perm_ai;
} ortho_info;

// Hacks to skip first (two) arguments if they are OpenCL types
#ifndef ocl2c
#define FST_GONLY_INT(x, ...) x(__VA_ARGS__)
#define TWO_GONLY_INT(x, ...) x(__VA_ARGS__)
#else
#define FST_GONLY_INT(x, y, ...) x(__VA_ARGS__)
#define TWO_GONLY_INT(x, y, z, ...) x(__VA_ARGS__)
#endif

#define FST_GONLY(x, ...) FST_GONLY_INT(MK_NAME(x), __VA_ARGS__)
#define TWO_GONLY(x, ...) TWO_GONLY_INT(MK_NAME(x), __VA_ARGS__)

// Make a rot_info and fill it with rotations.
rot_info FST_GONLY(make_rot_info, cl_context c,
		   size_t rot_len, size_t rots, size_t dim) {
  size_t *ri = malloc(sizeof(size_t) * rot_len);
  size_t *rj = malloc(sizeof(size_t) * rot_len);
  ftype *ra = malloc(sizeof(ftype) * rot_len);
  rot_info roti;
  roti.is = malloc(sizeof(BUFTYPE(size_t)) * rots);
  roti.js = malloc(sizeof(BUFTYPE(size_t)) * rots);
  roti.as = malloc(sizeof(BUFTYPE(ftype)) * rots);
  for(size_t j = 0; j < rots; j++) {
    rand_rot(rot_len, dim, ri, rj, ra);
    roti.is[j] = MK_BUF_COPY_RO_NA(c, size_t, rot_len, ri);
    roti.js[j] = MK_BUF_COPY_RO_NA(c, size_t, rot_len, rj);
    roti.as[j] = MK_BUF_COPY_RO_NA(c, ftype, rot_len, ra);
  }
  free(ri);
  free(rj);
  free(ra);
  return(roti);
}

// Make an ortho_info and fill it up with rotations and permutations
ortho_info FST_GONLY(make_ortho_info, cl_context c,
		     size_t rot_len_b, size_t rots_b,
		     size_t rot_len_a, size_t rots_a,
		     size_t dim_low, size_t dim_high, size_t dim_max) {
  ortho_info orti;
  size_t *perm;
  orti.rb = FST_GONLY(make_rot_info, c, rot_len_b, rots_b, dim_high);
  orti.ra = FST_GONLY(make_rot_info, c, rot_len_a, rots_a, dim_low);
  perm = rand_perm(dim_high, dim_max);
  orti.perm_b = MK_BUF_COPY_RO_NA(c, size_t, dim_max, perm);
  free(perm);
  perm = rand_perm(dim_low, dim_max);
  orti.perm_ai = MK_BUF_COPY_RO_NA(c, size_t, dim_max, perm);
  free(perm);
  return(orti);
}

// Free the contents of a rot_info
void MK_NAME(free_rot_info) (size_t rots, rot_info r) {
  for(size_t i = 0; i < rots; i++) {
    relMem(r.is[i]);
    relMem(r.js[i]);
    relMem(r.as[i]);
  }
  free(r.is);
  free(r.js);
  free(r.as);
}

// Free the contents of an ortho_info
void MK_NAME(free_ortho_info) (size_t rots_b, size_t rots_a, ortho_info *o) {
  MK_NAME(free_rot_info)(rots_b, o->rb);
  MK_NAME(free_rot_info)(rots_a, o->ra);
  relMem(o->perm_b);
  relMem(o->perm_ai);
}

#define CLEANUP_DATA MK_NAME(cleanup_data)

typedef struct {
  size_t rots_b, rots_a;
  ortho_info *o;
  int tries;
} CLEANUP_DATA;

// Callback for cleanup of ortho_infos
void MK_NAME(cleanup)(OEVENT e, OINT i, void *stuff) {
  CLEANUP_DATA *trash = (CLEANUP_DATA *)stuff;
  for(int i = 0; i < trash->tries; i++)
    MK_NAME(free_ortho_info)(trash->rots_b, trash->rots_a, trash->o + i);
  free(trash->o);
}

void FST_GONLY(walsh, cl_command_queue q,
	       size_t d, size_t n, BUFTYPE(ftype) a) {
  if(d == 1)
    return;
  int l = lg(d);
  size_t nth = max(d / 16, 1);
  for(int i = 0; i < l; i++)
    LOOP2(q, apply_walsh_step(l, i, a), n, nth);
}

void FST_GONLY(add_up_rows, cl_command_queue q,
	       size_t d, size_t n,
	       BUFTYPE(ftype) points, BUFTYPE(ftype) sums) {
  LOOP2(q, add_rows_step_0(d, n, points, sums), n/2, d);
  for(size_t m = n >> 1; m >> 1; m >>= 1)
    LOOP2(q, add_rows_step_n(d, m, sums), m/2, d);
}

void FST_GONLY(add_up_cols, cl_command_queue q,size_t d, size_t k,
	       size_t skip, size_t n, BUFTYPE(ftype) mat, BUFTYPE(ftype) out) {
  for(size_t l = d; l >> 1; l >>= 1)
    LOOP3(q, add_cols_step(d, l, k - skip, mat), n, k - skip, l / 2);
  fin_add_cols(q, d, k, skip, mat, out, n);
}

void FST_GONLY(do_sort, cl_command_queue q, size_t k, size_t n,
	     BUFTYPE(size_t) along, BUFTYPE(ftype) order) {
  int lk = lg(k);
  size_t nth = (size_t)1 << max(lk - 4, 0);
  for(int s = 0; s < lk; s++)
    for(int ss = s; ss >= 0; ss--)
      LOOP2(q, sort_two_step(k, s, ss, along, order), n, nth);
}


// Performs a random orthogonal operation + projection (supplied),
// and returns the signs of every provided vector after that op,
// shoved into a list of size_ts.
// Returns that list two ways: as a BUFTYPE(size_t) (possibly a cl_mem),
// and as a size_t * (sets *sgns to the address thereof).
// The latter is not guaranteed to have been written to,
// so wait for the queue you passed in to finish before reading it.
BUFTYPE(size_t) TWO_GONLY(run_initial, cl_context c, cl_command_queue q,
			  size_t n, size_t d_low,
			  size_t d_high, size_t d_max,
			  size_t rots_b, size_t rot_len_b,
			  size_t rots_a, size_t rot_len_a,
			  const ortho_info *o, BUFTYPE(ftype) points,
			  size_t **sgns) {
  BUFTYPE(ftype) pc = MK_BUF_RW_NA(c, ftype, n * d_high);
  enqueueCopyBuf(q, sizeof(ftype) * n * d_high, points, pc);
  for(size_t i = 0; i < rots_b; i++)   
    LOOP2(q, apply_rotation(d_high, o->rb.is[i], o->rb.js[i], o->rb.as[i], pc),
	  n, rot_len_b);
  BUFTYPE(ftype) pc2 = MK_BUF_RW_NA(c, ftype, n * d_max);
  LOOP2(q, apply_permutation(d_high, d_max, o->perm_b, pc, pc2), n, d_max);
  relMem(pc);
  FST_GONLY(walsh, q, d_max, n, pc2);
  for(size_t i = 0; i < rots_a; i++)
    LOOP2(q, apply_rotation(d_max, o->ra.is[i], o->ra.js[i], o->ra.as[i], pc2),
	  n, rot_len_a);

  pc = MK_BUF_RW_NA(c, ftype, n * d_low);
  LOOP2(q, apply_perm_inv(d_max, d_low, o->perm_ai, pc2, pc), n, d_max);
  relMem(pc2);
  BUFTYPE(size_t) signs = MK_BUF_RW_RO(c, size_t, n);
  LOOP1(q, compute_signs(d_low, pc, signs), n);
  relMem(pc);
  *sgns = malloc(sizeof(size_t) * n);
  enqueueReadBuf(q, sizeof(size_t) * n, signs, *sgns);
  return(signs);
}

// Undoes the random orthogonal operation and projection (supplied),
// on the basis vectors of the output (I.E., the input represents the
// d_high by d_low matrix M, we want M^T)
// and saves the results to the location given in loc.
void TWO_GONLY(save_vecs, cl_context c, cl_command_queue q,
	       size_t d_low, size_t d_high, size_t d_max,
	       size_t rots_b, size_t rot_len_b,
	       size_t rots_a, size_t rot_len_a,
	       const ortho_info *o, ftype *loc) {
  ftype *vcs = malloc(sizeof(ftype) * d_low * d_low);
  for(size_t i = 0; i < d_low; i++)
    for(size_t j = 0; j < d_low; j++)
      vcs[i * d_low + j] = i == j;
  BUFTYPE(ftype) vecs = MK_BUF_COPY_RO_NA(c, ftype, d_low * d_low, vcs);
  free(vcs);
  BUFTYPE(ftype) vecs2 = MK_BUF_RW_NA(c, ftype, d_low * d_max);
  LOOP2(q, apply_permutation(d_low, d_max, o->perm_ai, vecs, vecs2),
	    d_low, d_max);
  relMem(vecs);
  for(long i = rots_a - 1; i >= 0; i--)
    LOOP2(q, apply_rotation(d_max, o->ra.js[i], o->ra.is[i], o->ra.as[i],
			    vecs2), d_low, rot_len_a);
  FST_GONLY(walsh, q, d_max, d_low, vecs2);
  vecs = MK_BUF_RW_RO(c, ftype, d_low * d_high);
  LOOP2(q, apply_perm_inv(d_max, d_high, o->perm_b, vecs2, vecs),
	d_low, d_max);
  relMem(vecs2);
  for(long i = rots_b - 1; i >= 0; i--)
    LOOP2(q, apply_rotation(d_high, o->rb.js[i], o->rb.is[i], o->rb.as[i],
			     vecs), d_low, rot_len_b);
  enqueueReadBuf(q, sizeof(ftype) * d_low * d_high, vecs, loc);
  relMem(vecs);
}

// Pass it a pair of matrices, both n by k, one of ftypes,
// the other of size_ts.
// Sorts both by the ftypes along the columns,
// but of all entries with the same size_t,
// at most one will remain (the rest have the ftype set to infinity).
void FST_GONLY(sort_and_uniq, cl_command_queue q, size_t n,
		 size_t k, BUFTYPE(size_t) along,
		 BUFTYPE(ftype) order) {
  FST_GONLY(do_sort, q, k, n, along, order);
  LOOP2(q, rdups(k, along, order), n, k - 1);
  FST_GONLY(do_sort, q, k, n, along, order);
}

// Ugh, I dunno how to describe this. It computes certain distances.
void TWO_GONLY(compdists, cl_context c, cl_command_queue q,
	       size_t n, size_t k, size_t d, size_t ycnt, size_t s,
	       BUFTYPE(const ftype) y, BUFTYPE(const ftype) points,
	       BUFTYPE(size_t) pointers, BUFTYPE(ftype) dists) {
  BUFTYPE(ftype) diffs = MK_BUF_RW_NA(c, ftype, (k - s) * d * ycnt);
  LOOP3(q, compute_diffs_squared(d, k, n, s, pointers, y, points, diffs),
	ycnt, k - s, d);
  FST_GONLY(add_up_cols, q, d, k, s, ycnt, diffs, dists);
  relMem(diffs);
}
// This (a) figures out what the candidates for near neighbors are,
// and (b) computes the distances and sorts, then returns the k nearest.
void TWO_GONLY(second_half, cl_context c, cl_command_queue q,
	       size_t n, size_t k, size_t d_low, size_t d_high,
	       save_t *save, int i, int tries,
	       const size_t *sgns, BUFTYPE(size_t) signs,
	       BUFTYPE(const ftype) points,
	       BUFTYPE(size_t) pointers_out,
	       BUFTYPE(ftype) dists_out) {
  size_t *counts = malloc(sizeof(size_t) << d_low);
  for(size_t j = 0; j < 1 << d_low; j++)
    counts[j] = 0;
  for(size_t j = 0; j < n; j++)
    counts[sgns[j]]++;
  size_t tmax = counts[0];
  for(size_t j = 1; j < 1 << d_low; j++)
    if(tmax < counts[j])
      tmax = counts[j];
  size_t *wh = malloc(sizeof(size_t) * tmax << d_low);
  for(size_t j = 0; j < 1 << d_low; j++)
    for(size_t l = counts[j]; l < tmax; l++)
      wh[j * tmax + l] = n;
  for(size_t j = 0; j < n; j++)
    wh[sgns[j] * tmax + --counts[sgns[j]]] = j;
  free(counts);
  BUFTYPE(size_t) which = MK_BUF_COPY_RO_NA(c, size_t, tmax << d_low, wh);
  if(save != NULL) {
    save->which_par[i] = wh;
    save->par_maxes[i] = tmax;
  } else
    free(wh);
  BUFTYPE(size_t) which_d =
    MK_BUF_RW_NA(c, size_t, (d_low + 1) * n * tmax);
  LOOP3(q, compute_which(d_low, tmax, signs, which, which_d),
	n, d_low + 1, tmax);
  relMem(signs);
  relMem(which);
  BUFTYPE(ftype) dists = MK_BUF_RW_NA(c, ftype, (d_low + 1) * n * tmax);
  TWO_GONLY(compdists, c, q, n, (d_low + 1) * tmax, d_high, n, 0,
	    points, points, which_d, dists);
  FST_GONLY(sort_and_uniq, q, n, (d_low + 1) * tmax, which_d, dists);
  enqueueCopy2D(q, size_t, (d_low + 1) * tmax, k * tries, k * i, which_d,
		pointers_out, n, k);
  relMem(which_d);
  enqueueCopy2D(q, ftype, (d_low + 1) * tmax, k * tries, k * i, dists,
		dists_out, n, k);
  relMem(dists);
}

// Dists is null if not precalculated.
// Result is not guaranteed to be filled in, so wait on q.
// pointers, dists should be ycnt by len;
// graph should be n by k or equal to pointers;
// y should be ycnt by d,
// points should be n by d.
// Releases pointers and dists, but nothing else.
// Computes distances if necessary,
// then sorts, tosses all but top k,
// supercharges, recomputes distances, sorts,
// tosses all but top k, returns new guesses.
size_t *TWO_GONLY(det_results, cl_context c, cl_command_queue q,
		  size_t n, size_t k, size_t d, size_t ycnt, size_t len,
		  BUFTYPE(size_t) pointers, BUFTYPE(ftype) dists,
		  BUFTYPE(const size_t) graph, BUFTYPE(const ftype) y,
		  BUFTYPE(const ftype) points, ftype **dists_o) {
  if(dists == NULL) {
    dists = MK_BUF_RW_NA(c, ftype, len * ycnt);
    TWO_GONLY(compdists, c, q, n, len, d, ycnt, 0, y, points, pointers, dists);
  }
  FST_GONLY(sort_and_uniq, q, ycnt, len, pointers, dists);
  {
    BUFTYPE(size_t) ipts = MK_BUF_RW_RO(c, size_t, k * (k + 1) * ycnt);
    enqueueCopy2D(q, size_t, len, k * (k + 1), 0, pointers, ipts, ycnt, k);
    LOOP3(q, supercharge(n, len, pointers == graph? len : k, k,
			 pointers, graph, ipts), ycnt, k, k);
    relMem(pointers);
    pointers = ipts;
    BUFTYPE(ftype) dpts = MK_BUF_RW_RO(c, ftype, k * (k + 1) * ycnt);
    enqueueCopy2D(q, ftype, len, k * (k + 1), 0, dists, dpts, ycnt, k);
    relMem(dists);
    dists = dpts;
  }
  TWO_GONLY(compdists, c, q, n, k * (k + 1), d, ycnt, k,
	    y, points, pointers, dists);
  FST_GONLY(sort_and_uniq, q, ycnt, k * (k + 1), pointers, dists);
  if(dists_o != NULL) {
    *dists_o = malloc(sizeof(ftype) * ycnt * k);
    enqueueRead2D(q, ftype, k * (k + 1), k, 0, dists, *dists_o, ycnt, k);
  }
  relMem(dists);
  size_t *results = malloc(sizeof(size_t) * ycnt * k);
  enqueueRead2D(q, size_t, k * (k + 1), k, 0, pointers, results, ycnt, k);
  relMem(pointers);
  return(results);
}

/* Starting point: */
/* We have an array, points, that is n by d_long. */
/* We also have save, which is a save structure. */
size_t *MK_NAME(precomp) (size_t n, size_t k, size_t d, const ftype *points,
			  int tries, size_t rots_before, size_t rot_len_before,
			  size_t rots_after, size_t rot_len_after,
			  save_t *save, ftype **dists_o) {
  setup();
  size_t d_short = ceil(log2((ftype)n / k));
  size_t d_max = d - 1;
  d_max |= d_max >> 1;
  d_max |= d_max >> 2;
  d_max |= d_max >> 4;
  d_max |= d_max >> 8;
  d_max |= d_max >> 16;
  d_max |= d_max >> 32;
  d_max++;
  if(d_short > d_max)
    d_short = d_max;
  MAKE_COMMAND_QUEUE(gpu_context, the_gpu, NULL, NULL, q);
  MAKE_COMMAND_QUEUE(gpu_context, the_gpu, NULL, NULL, sq);
  BUFTYPE(ftype) pnts =
    MK_BUF_COPY_RW_NA(gpu_context, ftype, n * d, points);
  BUFTYPE(ftype) row_sums;
  if(save != NULL)
    row_sums = MK_BUF_RW_RO(gpu_context, ftype, (n/2) * d);  
  else
    row_sums = MK_BUF_RW_NA(gpu_context, ftype, (n/2) * d);
  FST_GONLY(add_up_rows, q, d, n, pnts, row_sums);  
  LOOP1(q, divide_by_length(n, row_sums), d);
  LOOP2(q, subtract_off(d, pnts, row_sums), n, d);
  if(save != NULL) {
    save->tries = tries;
    save->n = n;
    save->k = k;
    save->d_short = d_short;
    save->d_long = d;
    save->row_means = malloc(sizeof(ftype) * d);
    enqueueReadBuf(q, sizeof(ftype) * d, row_sums, save->row_means);
    save->which_par = malloc(sizeof(size_t *) * tries);
    save->par_maxes = malloc(sizeof(size_t) * tries);
    save->bases = malloc(sizeof(ftype) * tries * d_short * d);
  }
  relMem(row_sums);
  BUFTYPE(size_t) pointers_out =
    MK_BUF_RW_NA(gpu_context, size_t, n * k * tries);
  BUFTYPE(ftype) dists_out =
    MK_BUF_RW_NA(gpu_context, ftype, n * k * tries);
  ortho_info *inf = malloc(sizeof(ortho_info) * tries);
  for(int i = 0; i < tries; i++)
    inf[i] = FST_GONLY(make_ortho_info, gpu_context,
		       rot_len_before, rots_before,
		       rot_len_after, rots_after,
		       d_short, d, d_max);
  size_t **sgns = malloc(sizeof(size_t *) * tries);
  BUFTYPE(size_t) *signs = malloc(sizeof(BUFTYPE(size_t)) * tries);
  for(int i = 0; i < tries; i++) {
    signs[i] = TWO_GONLY(run_initial, gpu_context, q,
			 n, d_short, d, d_max,
			 rots_before, rot_len_before,
			 rots_after, rot_len_after,
			 inf + i, pnts, sgns + i);
    if(save != NULL)
      TWO_GONLY(save_vecs, gpu_context, sq, d_short, d, d_max,
		rots_before, rot_len_before, rots_after, rot_len_after,
		inf + i, save->bases + i * d_short * d);
  }
  relMem(pnts);
  clFinish(q);
  CLEANUP_DATA cup = {rots_before, rots_after, inf, tries};
  waitForQueueThenCall(sq, MK_NAME(cleanup), (void *)&cup);
  BUFTYPE(const ftype) pnts2 =
    MK_BUF_USE_RO_NA(gpu_context, ftype, n * d, points);
  for(int i = 0; i < tries; i++) {
    TWO_GONLY(second_half, gpu_context, q, n, k, d_short, d, save, i, tries,
	      sgns[i], signs[i], pnts2, pointers_out, dists_out);
    free(sgns[i]);
  }
  free(sgns);
  free(signs);
  size_t *fedges = TWO_GONLY(det_results, gpu_context, q,
			     n, k, d, n, k * tries,
			     pointers_out, dists_out,
			     pointers_out, pnts2, pnts2, dists_o);
  relMemU(pnts2);
  clFinish(q);
  clReleaseCommandQueue(q);
  clFinish(sq);
  clReleaseCommandQueue(sq);
  if(save != NULL) {
    save->graph = fedges;
    fedges = malloc(sizeof(size_t) * n * k);
    memcpy(fedges, save->graph, sizeof(size_t) * n * k);
  }
  return(fedges);
}

// Frees subsigns (soft).
// Computes the candidates and puts them in the right places.
void TWO_GONLY(shufcomp, cl_context c, cl_command_queue q, size_t d,
	       size_t ycnt, size_t offset, size_t len, size_t flen,
	       BUFTYPE(const size_t) subsigns,
	       const size_t *which,
	       BUFTYPE(size_t) ipts) {	       
  BUFTYPE(size_t) ppts = MK_BUF_RW_NA(c, size_t, len * (d + 1) * ycnt);
  BUFTYPE(const size_t) wp =
    MK_BUF_USE_RO_NA(c, size_t, len << d, which);
  LOOP3(q, compute_which(d, len, subsigns, wp, ppts), ycnt, d + 1, len);
  relMemU(subsigns);
  relMemU(wp);
  enqueueCopy2D(q, size_t, len * (d + 1), flen * (d + 1), offset * (d + 1),
		ppts, ipts, ycnt, len * (d + 1));
  relMem(ppts);
}

// We now have points (n by d_long), save->graph (n by k),
// save->row_means (d_long), save->par_maxes (tries),
// save->which_par (tries, then 1 << d_short by save->par_maxes[i]),
// save->bases (tries by d_short by d_long), y (ycnt by d_long).
size_t *MK_NAME(query) (const save_t *save, const ftype *points,
			size_t ycnt, const ftype *y, ftype **dists_o) {
  setup();
  MAKE_COMMAND_QUEUE(gpu_context, the_gpu, NULL, NULL, q);
  BUFTYPE(ftype) y2 = MK_BUF_COPY_RW_NA(gpu_context, ftype,
					 save->d_long * ycnt, y);
  BUFTYPE(const ftype) rm =
    MK_BUF_USE_RO_NA(gpu_context, ftype, save->d_long, save->row_means);
  LOOP2(q, subtract_off(save->d_long, y2, rm), ycnt, save->d_long);
  relMemU(rm);
  BUFTYPE(const ftype) bases = MK_BUF_USE_RO_NA(gpu_context, ftype,
						 save->tries * save->d_short *
						 save->d_long, save->bases);
  BUFTYPE(ftype) cprds = MK_BUF_RW_NA(gpu_context, ftype,
				       save->tries * ycnt *
				       save->d_short * save->d_long);
  LOOP3(q, prods(save->d_long, save->tries * save->d_short, y2, bases, cprds),
	ycnt, save->tries * save->d_short, save->d_long);
  relMemU(bases);
  relMem(y2);
  BUFTYPE(ftype) dprds = MK_BUF_RW_NA(gpu_context, ftype,
				       save->tries * ycnt * save->d_short);
  FST_GONLY(add_up_cols, q, save->d_long, save->d_short, 0, save->tries * ycnt,
	      cprds, dprds);
  relMem(cprds);
  size_t *pmaxes = malloc(sizeof(size_t) * save->tries);
  size_t msofar = 0;
  for(int i = 0; i < save->tries; i++) {
    pmaxes[i] = msofar;
    msofar += save->par_maxes[i];
  }
  BUFTYPE(size_t) signs = MK_BUF_RW_NA(gpu_context, size_t,
				       save->tries * ycnt);
  LOOP1(q, compute_signs(save->d_short, dprds, signs), save->tries * ycnt);
  relMem(dprds);
  BUFTYPE(size_t) ipts = MK_BUF_RW_NA(gpu_context, size_t,
				      msofar * (save->d_short + 1) * ycnt);
  for(int i = 0; i < save->tries; i++) {
    BUFTYPE(size_t) subsgns =
      MK_SUBBUF_RO_NA_REG(size_t, signs, i * ycnt, ycnt);
    TWO_GONLY(shufcomp, gpu_context, q, save->d_short, ycnt, pmaxes[i],
	      save->par_maxes[i], msofar, subsgns, save->which_par[i], ipts);
  }
  free(pmaxes);
  relMem(signs);
  BUFTYPE(const ftype) y3 =
    MK_BUF_USE_RO_NA(gpu_context, ftype, save->d_long * ycnt, y);
  BUFTYPE(const ftype) pnts =
    MK_BUF_USE_RO_NA(gpu_context, ftype, save->n * save->d_long, points);
  BUFTYPE(const size_t) graph =
    MK_BUF_USE_RO_NA(gpu_context, size_t, save->n * save->k, save->graph);
  size_t *ans = TWO_GONLY(det_results, gpu_context, q,
			  save->n, save->k, save->d_long, ycnt,
			  msofar * (save->d_short + 1), ipts, NULL,
			  graph, y3, pnts, dists_o);
  relMemU(y3);
  relMemU(pnts);
  relMemU(graph);
  clFinish(q);
  clReleaseCommandQueue(q);
  return(ans);
}
