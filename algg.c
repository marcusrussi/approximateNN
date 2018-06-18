#include "ann.h"
#include "rand_pr.h"
#include "gpu_comp.h"
#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>

#define max(a, b) ((a) < (b)?b:(a))

static unsigned lg(size_t d) {
  unsigned r = (d > 0xFFFFFFFF) << 5;
  d >>= r;
  unsigned s = (d > 0xFFFF) << 4;
  d >>= s, r |= s;
  d >>= s = (d > 0xFF) << 3, r |= s;
  d >>= s = (d > 0xF) << 2, r |= s;
  d >>= s = (d > 3) << 1;
  return(r | s | d >> 1);
}

cl_program compute;
static cl_kernel add_rows, add_rows_step_0, add_rows_step_n, divide_by_length,
  subtract_off, apply_rotation, apply_permutation, apply_perm_inv, apply_walsh,
  apply_walsh_step, compute_diffs_squared, add_cols, add_cols_step,
  add_cols_fin, sort_two_step, sort_two, rdups, copy_some_ints,
  copy_some_floats, compute_signs, compute_which, supercharge, prods;

#define create_kernel(p, x) make_kernel(p, &x, #x)
static void make_kernel(cl_program p, cl_kernel *x, const char *xn) {
  cl_int err;
  *x = clCreateKernel(p, xn, &err);
  if(err != CL_SUCCESS) {
    fprintf(stderr, "Error creating kernel for %s.\n", xn);
    exit(1);
  }
}

static void clone_kernel(cl_kernel *dst, cl_kernel src) {
  cl_int err;
  *dst = clCloneKernel(src, &err);
  if(err != CL_SUCCESS) {
    fprintf(stderr, "Error cloning kernel.\n");
    exit(1);
  }
}

static void setup(void) {
  static char c = 0;
  if(c)
    return;
  FILE *ocl_src = fopen("compute.cl");
  struct stat ocl_stt;
  stat("compute.cl", &ocl_stt);
  size_t len = ocl_stt.st_size;
  char *src_full = malloc(len);
  fread(src_full, 1, len, ocl_src);
  fclose(ocl_src);
  cl_int error;
  compute = clCreateProgramWithSource(gpu_context, 1, &src_full, &len, &error);
  if(error != CL_SUCCESS) {
    fprintf(stderr, "Error loading OpenCL code.\n");
    exit(1);
  }
  if(clBuildProgram(compute, 0, NULL, "-cl-std=CL2.0",
		    NULL, NULL) != CL_SUCCESS) {
    fprintf(stderr, "Error building program.\n");
    exit(1);
  }
  create_kernel(compute, add_rows);
  create_kernel(compute, add_rows_step_0);
  create_kernel(compute, add_rows_step_n);
  create_kernel(compute, divide_by_length);
  create_kernel(compute, subtract_off);
  create_kernel(compute, apply_rotation);
  create_kernel(compute, apply_permutation);
  create_kernel(compute, apply_perm_inv);
  create_kernel(compute, apply_walsh);
  create_kernel(compute, apply_walsh_step);
  create_kernel(compute, compute_diffs_squared);
  create_kernel(compute, add_cols);
  create_kernel(compute, add_cols_step);
  create_kernel(compute, add_cols_fin);
  create_kernel(compute, sort_two_step);
  create_kernel(compute, sort_two);
  create_kernel(compute, rdups);
  create_kernel(compute, copy_some_ints);
  create_kernel(compute, copy_some_floats);
  create_kernel(compute, compute_signs);
  create_kernel(compute, compute_which);
  create_kernel(compute, supercharge);
  create_kernel(compute, prods);
}

static void setKerArg(cl_kernel k, cl_uint ai, size_t as, const void *av) {
  if(clSetKernelArg(k, ai, as, av) != CL_SUCCESS)
    fprintf(stderr, "Error setting kernel args.\n"), exit(1);
}

static void add_up_rows(size_t d, size_t n,
			      cl_mem points, cl_mem sums,
			      cl_command_queue q) {
  size_t max_wgs;
  if(clGetKernelWorkgroupInfo(add_rows, NULL, CL_KERNEL_WORK_GROUP_SIZE,
			      sizeof(size_t), &max_wgs, NULL) != CL_SUCCESS)
    fprintf(stderr, "Error reading info.\n"), exit(1);
  if(n/2 <= max_wgs) {
    cl_kernel tk;
    size_t foo[2] = {n/2, d};
    size_t bar[2] = {n/2, 1};
    clone_kernel(&tk, add_rows);
    setKerArg(tk, 0, sizeof(size_t), &d);
    setKerArg(tk, 1, sizeof(size_t), &n);
    setKerArg(tk, 2, sizeof(cl_mem), &points);
    setKerArg(tk, 3, sizeof(cl_mem), &sums);
    clEnqueueNDRangeKernel(q, tk, 2, NULL, foo, bar, 0, NULL, NULL);
    clReleaseKernel(tk);
  } else {
    cl_kernel t0;
    size_t foo[2] = {n/2, d}
    clone_kernel(&t0, add_rows_step_0);
    setKerArg(t0, 0, sizeof(size_t), &d);
    setKerArg(t0, 1, sizeof(size_t), &n);
    setKerArg(t0, 2, sizeof(cl_mem), &points);
    setKerArg(t0, 3, sizeof(cl_mem), &sums);
    clEnqueueNDRangeKernel(q, tk, 2, NULL, foo, NULL, 0, NULL, NULL);
    clReleaseKernel(t0);
    clone_kernel(&t0, add_rows_step_n);
    setKerArg(t0, 0, sizeof(size_t), &d);
    setKerArg(t0, 2, sizeof(cl_mem), &sums);
    for(size_t m = n >> 1; m >> 1; m >>= 1) {
      cl_kernel tk;
      *foo = m/2;
      clone_kernel(&tk, t0);
      setKerArg(tk, 1, sizeof(size_t), &m);
      clEnqueueNDRangeKernel(q, tk, 2, NULL, foo, NULL, 0, NULL, NULL);
      clReleaseKernel(tk);
    }
    clReleaseKernel(t0);
  }
}

static void walsh(size_t d, size_t n, cl_mem a, cl_command_queue q) {
  if(d == 1)
    return;
  size_t max_wgs;
  if(clGetKernelWorkgroupInfo(apply_walsh, NULL, CL_KERNEL_WORK_GROUP_SIZE,
			      sizeof(size_t), &max_wgs, NULL) != CL_SUCCESS)
    fprintf(stderr, "Error reading info.\n"), exit(1);
  size_t l = lg(d);
  size_t nth = max(d / 16, 1);
  if(nth <= max_wgs) {
    cl_kernel tk;
    size_t foo[2] = {n, nth}, bar[2] = {1, nth};
    clone_kernel(&tk, apply_walsh);
    setKerArg(tk, 0, sizeof(size_t), &l);
    setKerArg(tk, 1, sizeof(cl_mem), &a);
    clEnqueueNDRangeKernel(q, tk, 2, NULL, foo, bar, 0, NULL, NULL);
    clReleaseKernel(tk);
  } else {
    cl_kernel t0;
    size_t foo[2] = {n, nth};
    clone_kernel(&t0, apply_walsh_step);
    setKerArg(t0, 0, sizeof(size_t), &l);
    setKerArg(t0, 2, sizeof(cl_mem), &a);
    for(size_t i = 0; i < l; i++) {
      cl_kernel tk;
      clone_kernel(&tk, t0);
      setKerArg(tk, 1, sizeof(size_t), &i);
      clEnqueueNDRangeKernel(q, tk, 2, NULL, foo, NULL, 0, NULL, NULL);
      clReleaseKernel(tk);
    }
    clReleaseKernel(t0);
  }
}

static void add_up_cols(size_t d, size_t k, size_t skip, size_t n,
			cl_mem mat, cl_mem out, cl_command_queue q) {
  size_t max_wgs;
  if(clGetKernelWorkgroupInfo(add_cols, NULL, CL_KERNEL_WORK_GROUP_SIZE,
			      sizeof(size_t), &max_wgs, NULL) != CL_SUCCESS)
    fprintf(stderr, "Error reading info.\n"), exit(1);
  if(d/2 <= max_wgs) {
    cl_kernel t0;
    size_t foo[3] = {n, k - skip, d/2}, bar[3] = {1, 1, d/2};
    clone_kernel(&t0, add_cols);
    setKerArg(t0, 0, sizeof(size_t), &d);
    setKerArg(t0, 1, sizeof(size_t), &k);
    setKerArg(t0, 2, sizeof(size_t), &skip);
    setKerArg(t0, 3, sizeof(cl_mem), &mat);
    setKerArg(t0, 4, sizeof(cl_mem), &out);
    clEnqueueNDRangeKernel(q, t0, 3, NULL, foo, bar, 0, NULL, NULL);
    clReleaseKernel(t0);
  } else {
    cl_kernel t0;
    size_t kms = k - skip, foo[3] = {n, kms, 0};
    clone_kernel(&t0, add_cols_step);
    setKerArg(t0, 0, sizeof(size_t), &d);
    setKerArg(t0, 2, sizeof(size_t), &kms);
    setKerArg(t0, 3, sizeof(cl_mem), &mat);
    for(size_t l = d; l >> 1; l >>= 1) {
      cl_kernel tk;
      clone_kernel(&tk, t0);
      setKerArg(tk, 1, sizeof(size_t), &l);
      clEnqueueNDRangeKernel(q, tk, 3, NULL, foo, NULL, 0, NULL, NULL);
      clReleaseKernel(tk);
    }
    clReleaseKernel(t0);
    clone_kernel(&t0, add_cols_fin);
    setKerArg(t0, 0, sizeof(size_t), &d);
    setKerArg(t0, 1, sizeof(size_t), &k);
    setKerArg(t0, 2, sizeof(size_t), &skip);
    setKerArg(t0, 3, sizeof(cl_mem), &mat);
    setKerArg(t0, 4, sizeof(cl_mem), &out);
    clEnqueueNDRangeKernel(q, t0, 2, NULL, foo, NULL, 0, NULL, NULL);
    clReleaseKernel(t0);
  }
}

static void do_sort(size_t k, size_t n, cl_mem along, cl_mem order,
		    cl_command_queue q) {
  size_t max_wgs;
  if(clGetKernelWorkgroupInfo(sort_two, NULL, CL_KERNEL_WORK_GROUP_SIZE,
			      sizeof(size_t), &max_wgs, NULL) != CL_SUCCESS)
    fprintf(stderr, "Error reading info.\n"), exit(1);
  int lk = lg(k);
  size_t nth = (size_t)1 << max(lk - 4, 0);
  if(nth <= max_wgs) {
    size_t foo[2] = {n, nth}, bar[2] = {1, nth};
    cl_kernel tk;
    clone_kernel(&tk, sort_two);
    setKerArg(tk, 0, sizeof(size_t), &k);
    setKerArg(tk, 1, sizeof(size_t), &n);
    setKerArg(tk, 2, sizeof(cl_mem), &along);
    setKerArg(tk, 3, sizeof(cl_mem), &order);
    clEnqueueNDRangeKernel(q, tk, 2, NULL, foo, bar, 0, NULL, NULL);
    clReleaseKernel(tk);
  } else {
    size_t foo[2] = {n, nth};
    cl_kernel t0;
    clone_kernel(&t0, sort_two_step);
    setKerArg(t0, 0, sizeof(size_t), &k);
    setKerArg(t0, 1, sizeof(size_t), &n);
    setKerArg(t0, 4, sizeof(cl_mem), &along);
    setKerArg(t0, 5, sizeof(cl_mem), &order);
    for(int s = 0; s < lk; s++) {
      cl_kernel t1;
      clone_kernel(&t1, t0);
      setKerArg(t1, 2, sizeof(int), s);
      for(int ss = s; ss >= 0; ss--) {
	cl_kernel tk;
	clone_kernel(&tk, t1);
	setKerArg(tk, 3, sizeof(int), ss);
	clEnqueueNDRangeKernel(q, tk, 2, NULL, foo, NULL, 0, NULL, NULL);
	clReleaseKernel(tk);
      }
      clReleaseKernel(t1);
    }
    clReleaseKernel(t0);
  }
}
/* Starting point: */
/* We have an array, points, that is n by d_long. */
/* We also have save, which is a save structure. */
size_t *precomp_gpu(size_t n, size_t k, size_t d, double *points,
		    int tries, size_t rots_before, size_t rot_len_before,
		    size_t rots_after, size_t rot_len_after, save_t *save) {
  fprintf(stderr, "Sorry, gpu unimplemented.\n");
  exit(1);
#if 0
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
    pc = malloc(sizeof(double) * (n * d_short + 1));
    LOOP2(apply_perm_inv(d_max, d_short, n * d_short, perm_after_i, pc2, pc),
	     n, d_max);
    free(pc2);
    size_t *signs = malloc(sizeof(size_t) * n);
    LOOP1(compute_signs(d_short, (unsigned long *)pc, signs), n);
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
			     rot_as_b + j * rot_len_before, vecs),
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
        which[signs[j] * tmax + --counts[signs[j]]] = j;
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
    if(save == NULL)
        free(which);
    free(signs);
    free(which_d);
  }
  do_sort(k * tries, n, pointers_out, dists_out);
  LOOP2(rdups(k * tries, pointers_out, dists_out), n, k * tries - 1);
  do_sort(k * tries, n, pointers_out, dists_out);
  size_t *nedge = malloc(sizeof(size_t) * (n * k * (k + 1) + 1));
  double *ndists = malloc(sizeof(double) * (n * k * (k + 1) + 1));
  LOOP3(supercharge(n, k * tries, k * tries, k,
		    pointers_out, pointers_out, nedge),
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
  free(points);
  add_up_cols(d, k * (k + 1), k, n, diffs, ndists);
  free(diffs);
  do_sort(k * (k + 1), n, nedge, ndists);
  LOOP2(rdups(k * (k + 1), nedge, ndists), n, k * (k + 1) - 1);
  do_sort(k * (k + 1), n, nedge, ndists);
  free(ndists);
  size_t *fedges = malloc(sizeof(size_t) * n * k);
  if(save != NULL)
    save->graph = fedges;
  LOOP2(copy_some_ints(k * (k + 1), k, 0, nedge, fedges), n, k);
  free(nedge);
  return(fedges);
#endif
  return(NULL);
}

// We now have points (n by d_long), save->graph (n by k),
// save->row_means (d_long), save->par_maxes (tries),
// save->which_par (tries, then 1 << d_short by save->par_maxes[i]),
// save->bases (tries by d_short by d_long), y (ycnt by d_long).

size_t *query_gpu(const save_t *save, const double *points,
	      size_t ycnt, double *y) {
  fprintf(stderr, "Sorry, gpu unimplemented.\n");
  exit(1);
  return(NULL);
#if 0
  double *cprds = malloc(sizeof(double) * save->tries * ycnt *
			 save->d_short * save->d_long);
  double *dprds = malloc(sizeof(double) * save->tries * ycnt * save->d_short);
  double *y2 = malloc(sizeof(double) * save->d_long * ycnt);
  memcpy(y2, y, sizeof(double) * save->d_long * ycnt);
  LOOP2(subtract_off(save->d_long, y2, save->row_means), ycnt, save->d_long);
  LOOP3(prods(save->d_long, save->tries * save->d_short, y2,
	      save->bases, cprds),
	ycnt, save->tries * save->d_short, save->d_long);
  free(y2);
  add_up_cols(save->d_long, save->d_short, 0, save->tries * ycnt,
	      cprds, dprds);
  free(cprds);
  size_t *pmaxes = malloc(sizeof(size_t) * save->tries);
  size_t msofar = 0;
  for(int i = 0; i < save->tries; i++) {
    pmaxes[i] = msofar;
    msofar += save->par_maxes[i];
  }
  size_t *signs = malloc(sizeof(size_t) * save->tries * ycnt);
  LOOP1(compute_signs(save->d_short, (unsigned long *)dprds, signs),
      save->tries * ycnt);
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
			      save->n, 0, ipts, y, points, diffs),
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
    LOOP2(copy_some_ints(msofar * (save->d_short + 1),
			 save->k * (save->k + 1), 0, ipts, ipts2),
	  ycnt, save->k);
    LOOP3(supercharge(save->n, msofar * (save->d_short + 1), save->k, save->k,
		      ipts, save->graph, ipts2),
	  ycnt, save->k, save->k);

    free(ipts);
    (ipts = ipts2)[save->k * (save->k + 1) * ycnt] = save->n;
    double *dpts2 = malloc(sizeof(double) *
			   (save->k * (save->k + 1) * ycnt + 1));
    LOOP2(copy_some_floats(msofar * (save->d_short + 1),
			   save->k * (save->k + 1), 0, dpts, dpts2),
	  ycnt, save->k);
    free(dpts);
    (dpts = dpts2)[save->k * (save->k + 1) * ycnt] = 1.0/0;
  }
  diffs = malloc(sizeof(double) * save->k * save->k * save->d_long * ycnt);
  
  LOOP3(compute_diffs_squared(save->d_long, save->k * (save->k + 1), save->n,
			      save->k, ipts, y, points, diffs),
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
#endif
}
