#include "ann.h"
#include "rand_pr.h"
#include "gpu_comp.h"
#include <OpenCL/opencl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <math.h>

#ifdef OSX
#define clCreateCommandQueueWithProperties(a, b, c, d) \
  clCreateCommandQueue(a, b, 0, d)
#endif

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
static cl_kernel add_rows, apply_walsh, add_cols, sort_two;
#ifdef OSX
#define clone_kernel(src) make_kernel(compute, #src);
#else
static cl_kernel add_rows_step_0, add_rows_step_n, divide_by_length,
  subtract_off, apply_rotation, apply_permutation, apply_perm_inv,
  apply_walsh_step, compute_diffs_squared, add_cols_step, add_cols_fin,
  sort_two_step, rdups, compute_signs, compute_which, supercharge, prods;

static cl_kernel clone_kernel(cl_kernel src) {
  cl_int err;
  cl_kernel dst = clCloneKernel(src, &err);
  if(err != CL_SUCCESS)
    fprintf(stderr, "Error cloning kernel.\n"), exit(1);
  return(dst);
}
#endif

#define create_kernel(p, x) x = make_kernel(p, #x)

static cl_kernel make_kernel(cl_program p, const char *xn) {
  cl_int err;
  cl_kernel x = clCreateKernel(p, xn, &err);
  if(err != CL_SUCCESS) {
    fprintf(stderr, "Error creating kernel for %s.\n", xn);
    exit(1);
  }
  return(x);
}

static void setup(void) {
  static char c = 0;
  if(c)
    return;
  FILE *ocl_src = fopen("compute.cl", "r");
  struct stat ocl_stt;
  stat("compute.cl", &ocl_stt);
  size_t len = ocl_stt.st_size;
  char *src_full = malloc(len);
  fread(src_full, 1, len, ocl_src);
  fclose(ocl_src);
  cl_int error;
  compute = clCreateProgramWithSource(gpu_context, 1, (const char **)&src_full,
				      &len, &error);
  if(error != CL_SUCCESS) {
    fprintf(stderr, "Error loading OpenCL code.\n");
    exit(1);
  }
  if(clBuildProgram(compute, 0, NULL,
#ifdef OSX
		    "",
#else
		    "-cl-std=CL2.0",
#endif		    
		    NULL, NULL) != CL_SUCCESS) {
    fprintf(stderr, "Error building program.\n");
    exit(1);
  }
  create_kernel(compute, add_rows);
  create_kernel(compute, apply_walsh);
  create_kernel(compute, add_cols);
  create_kernel(compute, sort_two);
#ifndef OSX
  create_kernel(compute, add_rows_step_0);
  create_kernel(compute, add_rows_step_n);
  create_kernel(compute, divide_by_length);
  create_kernel(compute, subtract_off);
  create_kernel(compute, apply_rotation);
  create_kernel(compute, apply_permutation);
  create_kernel(compute, apply_perm_inv);
  create_kernel(compute, apply_walsh_step);
  create_kernel(compute, compute_diffs_squared);
  create_kernel(compute, add_cols_step);
  create_kernel(compute, add_cols_fin);
  create_kernel(compute, sort_two_step);
  create_kernel(compute, rdups);
  create_kernel(compute, compute_signs);
  create_kernel(compute, compute_which);
  create_kernel(compute, supercharge);
  create_kernel(compute, prods);
#endif
}

static void enqueue1D(cl_command_queue q, cl_kernel k, size_t x) {
  if(clEnqueueNDRangeKernel(q, k, 1, NULL, &x, NULL, 0, NULL, NULL)
     != CL_SUCCESS)
    fprintf(stderr, "Failed enqueue.\n"), exit(1);
  clReleaseKernel(k);
}

static void enqueue2D(cl_command_queue q, cl_kernel k, size_t x, size_t y) {
  size_t foo[2] = {x, y};
  if(clEnqueueNDRangeKernel(q, k, 2, NULL, foo, NULL, 0, NULL, NULL)
     != CL_SUCCESS)
    fprintf(stderr, "Failed enqueue.\n"), exit(1);
  clReleaseKernel(k);
}

static void enqueue3D(cl_command_queue q, cl_kernel k,
		      size_t x, size_t y, size_t z) {
  size_t foo[3] = {x, y, z};
  if(clEnqueueNDRangeKernel(q, k, 3, NULL, foo, NULL, 0, NULL, NULL)
     != CL_SUCCESS)
    fprintf(stderr, "Failed enqueue.\n"), exit(1);
  clReleaseKernel(k);
}

static void enqueueCopy2D(cl_command_queue q,
			  size_t height_pre,
			  size_t height_post, size_t start_post,
			  cl_mem from, cl_mem to, size_t n, size_t k) {
  size_t src_ori[3] = {0, 0, 0};
  size_t dst_ori[3] = {0, start_post, 0};
  size_t reg[3] = {n, k, sizeof(double)};
  if(clEnqueueCopyBufferRect(q, from, to, src_ori, dst_ori, reg,
			     height_pre, sizeof(double),
			     height_post, sizeof(double),
			     0, NULL, NULL) != CL_SUCCESS)
    fprintf(stderr, "Failed enqueue of copy.\n"), exit(1);
}

static void enqueueCopyBuf(cl_command_queue q,
			   size_t sz, cl_mem from, cl_mem to) {
  if(clEnqueueCopyBuffer(q, from, to, 0, 0, sz,
			 0, NULL, NULL) != CL_SUCCESS)
    fprintf(stderr, "Failed enqueue of copy.\n"), exit(1);
}

static void enqueueFinAC(cl_command_queue q, size_t height, size_t k,
			 size_t skip, cl_mem from, cl_mem to, size_t n) {
  size_t src_ori[3] = {0, 0, 0};
  size_t dst_ori[3] = {0, 0, 0};
  size_t reg[3] = {n, k - skip, sizeof(double)};
  if(clEnqueueCopyBufferRect(q, from, to, src_ori, dst_ori, reg,
			     k - skip, height * sizeof(double),
			     k - skip, sizeof(double),
			     0, NULL, NULL) != CL_SUCCESS)
    fprintf(stderr, "Failed enqueue of copy.\n"), exit(1);
}

#define ska(k, i, o) setKerArg(k, i, sizeof(o), &o)

static void setKerArg(cl_kernel k, cl_uint ai, size_t as, const void *av) {
  if(clSetKernelArg(k, ai, as, av) != CL_SUCCESS)
    fprintf(stderr, "Error setting kernel args.\n"), exit(1);
}

static void add_up_rows(size_t d, size_t n,
			cl_mem points, cl_mem sums,
			cl_command_queue q) {
  size_t max_wgs;
  if(clGetKernelWorkGroupInfo(add_rows, NULL, CL_KERNEL_WORK_GROUP_SIZE,
			      sizeof(size_t), &max_wgs, NULL) != CL_SUCCESS)
    fprintf(stderr, "Error reading info.\n"), exit(1);
  if(n/2 <= max_wgs) {
    size_t foo[2] = {n/2, d};
    size_t bar[2] = {n/2, 1};
    cl_kernel tk = clone_kernel(add_rows);
    ska(tk, 0, d);
    ska(tk, 1, n);
    ska(tk, 2, points);
    ska(tk, 3, sums);
    clEnqueueNDRangeKernel(q, tk, 2, NULL, foo, bar, 0, NULL, NULL);
    clReleaseKernel(tk);
  } else {
    cl_kernel t0 = clone_kernel(add_rows_step_0);
    ska(t0, 0, d);
    ska(t0, 1, n);
    ska(t0, 2, points);
    ska(t0, 3, sums);
    enqueue2D(q, t0, n/2, d);
#ifndef OSX
    t0 = clone_kernel(add_rows_step_n);
    ska(t0, 0, d);
    ska(t0, 2, sums);
#endif
    for(size_t m = n >> 1; m >> 1; m >>= 1) {
#ifdef OSX
      cl_kernel tk = clone_kernel(add_rows_step_n);
      ska(tk, 0, d);
      ska(tk, 2, sums);
#else
      cl_kernel tk = clone_kernel(t0);
#endif
      ska(tk, 1, m);
      enqueue2D(q, tk, m/2, d);
    }
#ifndef OSX
    clReleaseKernel(t0);
#endif
  }
}

static void walsh(size_t d, size_t n, cl_mem a, cl_command_queue q) {
  if(d == 1)
    return;
  size_t max_wgs;
  if(clGetKernelWorkGroupInfo(apply_walsh, NULL, CL_KERNEL_WORK_GROUP_SIZE,
			      sizeof(size_t), &max_wgs, NULL) != CL_SUCCESS)
    fprintf(stderr, "Error reading info.\n"), exit(1);
  size_t l = lg(d);
  size_t nth = max(d / 16, 1);
  if(nth <= max_wgs) {
    size_t foo[2] = {n, nth}, bar[2] = {1, nth};
    cl_kernel tk = clone_kernel(apply_walsh);
    ska(tk, 0, l);
    ska(tk, 1, a);
    clEnqueueNDRangeKernel(q, tk, 2, NULL, foo, bar, 0, NULL, NULL);
    clReleaseKernel(tk);
  } else {
#ifndef OSX
    cl_kernel t0 = clone_kernel(apply_walsh_step);
    ska(t0, 0, l);
    ska(t0, 2, a);
#endif
    for(size_t i = 0; i < l; i++) {
#ifdef OSX
      cl_kernel tk = clone_kernel(apply_walsh_step);
      ska(tk, 0, l);
      ska(tk, 2, a);
#else
      cl_kernel tk = clone_kernel(&tk, t0);
#endif
      ska(tk, 1, i);
      enqueue2D(q, tk, n, nth);
    }
#ifndef OSX
    clReleaseKernel(t0);
#endif
  }
}

static void add_up_cols(size_t d, size_t k, size_t skip, size_t n,
			cl_mem mat, cl_mem out, cl_command_queue q) {
  size_t max_wgs;
  if(clGetKernelWorkGroupInfo(add_cols, NULL, CL_KERNEL_WORK_GROUP_SIZE,
			      sizeof(size_t), &max_wgs, NULL) != CL_SUCCESS)
    fprintf(stderr, "Error reading info.\n"), exit(1);
  if(d/2 <= max_wgs) {
    size_t foo[3] = {n, k - skip, d/2}, bar[3] = {1, 1, d/2};
    cl_kernel tk = clone_kernel(add_cols);
    ska(tk, 0, d);
    ska(tk, 1, k);
    ska(tk, 2, skip);
    ska(tk, 3, mat);
    ska(tk, 4, out);
    clEnqueueNDRangeKernel(q, tk, 3, NULL, foo, bar, 0, NULL, NULL);
    clReleaseKernel(tk);
  } else {
    size_t kms = k - skip;
#ifndef OSX
    cl_kernel t0 = clone_kernel(add_cols_step);
    ska(t0, 0, d);
    ska(t0, 2, kms);
    ska(t0, 3, mat);
#endif
    for(size_t l = d; l >> 1; l >>= 1) {
#ifdef OSX
      cl_kernel tk = clone_kernel(add_cols_step);
      ska(tk, 0, d);
      ska(tk, 2, kms);
      ska(tk, 3, mat);
#else
      cl_kernel tk = clone_kernel(t0);
#endif
      ska(tk, 1, l);
      enqueue3D(q, tk, n, k - skip, l / 2);
    }
#ifndef OSX
    clReleaseKernel(t0);
#endif
    enqueueFinAC(q, d, k, skip, mat, out, n);
  }
}

static void do_sort(size_t k, size_t n, cl_mem along, cl_mem order,
		    cl_command_queue q) {
  size_t max_wgs;
  if(clGetKernelWorkGroupInfo(sort_two, NULL, CL_KERNEL_WORK_GROUP_SIZE,
			      sizeof(size_t), &max_wgs, NULL) != CL_SUCCESS)
    fprintf(stderr, "Error reading info.\n"), exit(1);
  int lk = lg(k);
  size_t nth = (size_t)1 << max(lk - 4, 0);
  if(nth <= max_wgs) {
    size_t foo[2] = {n, nth}, bar[2] = {1, nth};
    cl_kernel tk = clone_kernel(sort_two);
    ska(tk, 0, k);
    ska(tk, 1, n);
    ska(tk, 2, along);
    ska(tk, 3, order);
    clEnqueueNDRangeKernel(q, tk, 2, NULL, foo, bar, 0, NULL, NULL);
    clReleaseKernel(tk);
  } else {
#ifndef OSX
    cl_kernel t0 = clone_kernel(sort_two_step);
    ska(t0, 0, k);
    ska(t0, 1, n);
    ska(t0, 4, along);
    ska(t0, 5, order);
    for(int s = 0; s < lk; s++) {
      cl_kernel t1 = clone_kernel(t0);
      ska(t1, 2, s);
      for(int ss = s; ss >= 0; ss--) {
	cl_kernel tk = clone_kernel(t1);
	ska(tk, 3, ss);
	enqueue2D(q, tk, n, nth);
      }
      clReleaseKernel(t1);
    }
    clReleaseKernel(t0);
#else
  for(int s = 0; s < lk; s++)
    for(int ss = s; ss >= 0; ss--) {
          cl_kernel tk = clone_kernel(sort_two_step);
	  ska(tk, 0, k);
	  ska(tk, 1, n);
	  ska(tk, 2, s);
	  ska(tk, 3, ss);
	  ska(tk, 4, along);
	  ska(tk, 5, order);
	  enqueue2D(q, tk, n, nth);
    }
#endif
  }
}

		      
static cl_kernel cr_divide_by_length(size_t len, cl_mem r) {
  cl_kernel k = clone_kernel(divide_by_length);
  ska(k, 0, len);
  ska(k, 1, r);
  return(k);
}
static cl_kernel cr_subtract_off(size_t h, cl_mem a, cl_mem r) {
  cl_kernel k = clone_kernel(subtract_off);
  ska(k, 0, h);
  ska(k, 1, a);
  ska(k, 2, r);
  return(k);
}
static cl_kernel cr_apply_rotation(size_t h, cl_mem i, cl_mem j,
				   cl_mem a, cl_mem s) {
  cl_kernel k = clone_kernel(apply_rotation);
  ska(k, 0, h);
  ska(k, 1, i);
  ska(k, 2, j);
  ska(k, 3, a);
  ska(k, 4, s);
  return(k);
}

static cl_kernel cr_apply_permutation(size_t hb, size_t ha,
				      cl_mem perm, cl_mem a, cl_mem r) {
  cl_kernel k = clone_kernel(apply_permutation);
  ska(k, 0, hb);
  ska(k, 1, ha);
  ska(k, 2, perm);
  ska(k, 3, a);
  ska(k, 4, r);
  return(k);
}

static cl_kernel cr_apply_perm_inv(size_t hb, size_t ha, size_t g,
				   cl_mem p, cl_mem a, cl_mem r) {
  cl_kernel k = clone_kernel(apply_perm_inv);
  ska(k, 0, hb);
  ska(k, 1, ha);
  ska(k, 2, g);
  ska(k, 3, p);
  ska(k, 4, a);
  ska(k, 5, r);
  return(k);
}
static cl_kernel cr_compute_diffs_squared(size_t d, size_t c,
					  size_t n, size_t s,
					  cl_mem w, cl_mem pa,
					  cl_mem p, cl_mem dr) {
  cl_kernel k = clone_kernel(compute_diffs_squared);
  ska(k, 0, d);
  ska(k, 1, c);
  ska(k, 2, n);
  ska(k, 3, s);
  ska(k, 4, w);
  ska(k, 5, pa);
  ska(k, 6, p);
  ska(k, 7, dr);
  return(k);
}
static cl_kernel cr_rdups(size_t c, cl_mem a, cl_mem o) {
  cl_kernel k = clone_kernel(rdups);
  ska(k, 0, c);
  ska(k, 1, a);
  ska(k, 2, o);
  return(k);
}

static cl_kernel cr_compute_signs(size_t d, cl_mem p, cl_mem r) {
  cl_kernel k = clone_kernel(compute_signs);
  ska(k, 0, d);
  ska(k, 1, p);
  ska(k, 2, r);
  return(k);
}

static cl_kernel cr_compute_which(size_t d, size_t max, cl_mem wir, cl_mem wi,
				  cl_mem w) {
  cl_kernel k = clone_kernel(compute_which);
  ska(k, 0, d);
  ska(k, 1, max);
  ska(k, 2, wir);
  ska(k, 3, wi);
  ska(k, 4, w);
  return(k);
}

static cl_kernel cr_supercharge(size_t n, size_t la, size_t l, size_t k,
				cl_mem na, cl_mem nb, cl_mem a) {
  cl_kernel kr = clone_kernel(supercharge);
  ska(kr, 0, n);
  ska(kr, 1, la);
  ska(kr, 2, l);
  ska(kr, 3, k);
  ska(kr, 4, na);
  ska(kr, 5, nb);
  ska(kr, 6, a);
  return(kr);
}

static cl_kernel cr_prods(size_t d, size_t n, cl_mem v, cl_mem p, cl_mem o) {
  cl_kernel k = clone_kernel(prods);
  ska(k, 0, d);
  ska(k, 1, n);
  ska(k, 2, v);
  ska(k, 3, p);
  ska(k, 4, o);
  return(k);
}

#define LOOP1(q, a, x) enqueue1D(q, cr_ ## a, x)
#define LOOP2(q, a, x, y) enqueue2D(q, cr_ ## a, x, y)
#define LOOP3(q, a, x, y, z) enqueue3D(q, cr_ ## a, x, y, z)


/* Starting point: */
/* We have an array, points, that is n by d_long. */
/* We also have save, which is a save structure. */
size_t *precomp_gpu(size_t n, size_t k, size_t d, double *points,
		    int tries, size_t rots_before, size_t rot_len_before,
		    size_t rots_after, size_t rot_len_after, save_t *save) {
  setup();
  size_t d_short = ceil(log2((double)n / k));
  size_t d_max = d - 1;
  d_max |= d_max >> 1;
  d_max |= d_max >> 2;
  d_max |= d_max >> 4;
  d_max |= d_max >> 8;
  d_max |= d_max >> 16;
  d_max |= d_max >> 32;
  d_max++;
  cl_command_queue q = clCreateCommandQueueWithProperties(gpu_context,
							  the_gpu, NULL, NULL);
  cl_command_queue sq =
    clCreateCommandQueueWithProperties(gpu_context, the_gpu, NULL, NULL);
  cl_mem pnts = clCreateBuffer(gpu_context,
			       CL_MEM_READ_WRITE |
			       CL_MEM_COPY_HOST_PTR |
			       CL_MEM_HOST_NO_ACCESS,
			       sizeof(double) * n * d,
			       points, NULL);
  cl_mem row_sums = clCreateBuffer(gpu_context,
				   CL_MEM_READ_WRITE |
				   (save != NULL? CL_MEM_HOST_READ_ONLY :
				    CL_MEM_HOST_NO_ACCESS),
				   sizeof(double) * (n/2) * d,
				   NULL, NULL);
  add_up_rows(d, n, pnts, row_sums, q);
  
  LOOP1(q, divide_by_length(n, row_sums), d);
  LOOP2(q, subtract_off(d, pnts, row_sums), n, d);
  if(save != NULL) {
    save->tries = tries;
    save->n = n;
    save->k = k;
    save->d_short = d_short;
    save->d_long = d;
    save->row_means = malloc(sizeof(double) * d);
    clEnqueueReadBuffer(q, row_sums, 0, 0, sizeof(double) * d, save->row_means,
			0, NULL, NULL);
    save->which_par = malloc(sizeof(size_t *) * tries);
    save->par_maxes = malloc(sizeof(size_t) * tries);
    save->bases = malloc(sizeof(double) * tries * d_short * d);
  }
  clReleaseMemObject(row_sums);
  cl_mem pointers_out = clCreateBuffer(gpu_context,
				       CL_MEM_READ_WRITE |
				       CL_MEM_HOST_NO_ACCESS,
				       sizeof(size_t) * (n * k * tries + 1),
				       NULL, NULL);
  cl_mem dists_out = clCreateBuffer(gpu_context,
				    CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
				    sizeof(double) * (n * k * tries + 1),
				    NULL, NULL);
  for(int i = 0; i < tries; i++) {

    size_t *ri = malloc(sizeof(size_t) *  rot_len_before);
    size_t *rj = malloc(sizeof(size_t) *  rot_len_before);
    double *ra = malloc(sizeof(double) *  rot_len_before);
    cl_mem *rot_is_b = malloc(sizeof(cl_mem) * rots_before);
    cl_mem *rot_js_b = malloc(sizeof(cl_mem) * rots_before);
    cl_mem *rot_as_b = malloc(sizeof(cl_mem) * rots_before);
    for(size_t j = 0; j < rots_before; j++) {
    	  rand_rot(rot_len_before, d,
		   ri + j, rj + j, ra + j);
	  rot_is_b[j] = clCreateBuffer(gpu_context,
				       CL_MEM_READ_ONLY |
				       CL_MEM_COPY_HOST_PTR |
				       CL_MEM_HOST_NO_ACCESS,
				       sizeof(size_t) * rot_len_before,
				       ri, NULL);
	  rot_js_b[j] = clCreateBuffer(gpu_context,
				       CL_MEM_READ_ONLY |
				       CL_MEM_COPY_HOST_PTR |
				       CL_MEM_HOST_NO_ACCESS,
				       sizeof(size_t) * rot_len_before,
				       rj, NULL);
	  rot_as_b[j] = clCreateBuffer(gpu_context,
				       CL_MEM_READ_ONLY |
				       CL_MEM_COPY_HOST_PTR |
				       CL_MEM_HOST_NO_ACCESS,
				       sizeof(double) * rot_len_before,
				       ra, NULL);
    }
    free(ri);
    free(rj);
    free(ra);
    ri = malloc(sizeof(size_t) *  rot_len_after);
    rj = malloc(sizeof(size_t) *  rot_len_after);
    ra = malloc(sizeof(double) *  rot_len_after);
    cl_mem *rot_is_a = malloc(sizeof(cl_mem) * rots_after);
    cl_mem *rot_js_a = malloc(sizeof(cl_mem) * rots_after);
    cl_mem *rot_as_a = malloc(sizeof(cl_mem) * rots_after);
    for(size_t j = 0; j < rots_after; j++) {
    	  rand_rot(rot_len_after, d,
		   ri + j, rj + j, ra + j);
	  rot_is_a[j] = clCreateBuffer(gpu_context,
				       CL_MEM_READ_ONLY |
				       CL_MEM_COPY_HOST_PTR |
				       CL_MEM_HOST_NO_ACCESS,
				       sizeof(size_t) * rot_len_after,
				       ri, NULL);
	  rot_js_a[j] = clCreateBuffer(gpu_context,
				       CL_MEM_READ_ONLY |
				       CL_MEM_COPY_HOST_PTR |
				       CL_MEM_HOST_NO_ACCESS,
				       sizeof(size_t) * rot_len_after,
				       rj, NULL);
	  rot_as_a[j] = clCreateBuffer(gpu_context,
				       CL_MEM_READ_ONLY |
				       CL_MEM_COPY_HOST_PTR |
				       CL_MEM_HOST_NO_ACCESS,
				       sizeof(double) * rot_len_after,
				       ra, NULL);
    }
    free(ri);
    free(rj);
    free(ra);
    size_t *pb = rand_perm(d, d_max);
    cl_mem perm_before = clCreateBuffer(gpu_context,
					CL_MEM_READ_ONLY |
					CL_MEM_COPY_HOST_PTR |
					CL_MEM_HOST_NO_ACCESS,
					sizeof(size_t) * d_max,
					pb, NULL);
    free(pb);
    size_t *pai = rand_perm(d_short, d_max);
    cl_mem perm_after_i = clCreateBuffer(gpu_context,
					 CL_MEM_READ_ONLY |
					 CL_MEM_COPY_HOST_PTR |
					 CL_MEM_HOST_NO_ACCESS,
					 sizeof(size_t) * d_max,
					 pai, NULL);

    free(pai);
    cl_mem pc = clCreateBuffer(gpu_context,
			       CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
			       sizeof(double) * n * d, NULL, NULL);
    enqueueCopyBuf(q, sizeof(double) * n * d, pnts, pc);
    for(size_t j = 0; j < rots_before; j++)        
      LOOP2(q, apply_rotation(d, rot_is_b[j], rot_js_b[j], rot_as_b[j], pc),
	                n, rot_len_before);
    cl_mem pc2 = clCreateBuffer(gpu_context,
			       CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
			       sizeof(double) * n * d_max, NULL, NULL);

    LOOP2(q, apply_permutation(d, d_max, perm_before, pc, pc2), n, d_max);
    clReleaseMemObject(pc);
    walsh(d_max, n, pc2, q);
    for(size_t j = 0; j < rots_after; j++)
      LOOP2(q, apply_rotation(d, rot_is_a[j], rot_js_a[j], rot_as_a[j], pc2),
	    n, rot_len_after);
    pc = clCreateBuffer(gpu_context,
			CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
			sizeof(double) * (n * d_short + 1), NULL, NULL);
    LOOP2(q, apply_perm_inv(d_max, d_short, n * d_short, perm_after_i,
			    pc2, pc),
	     n, d_max);
    clReleaseMemObject(pc2);
    cl_mem signs = clCreateBuffer(gpu_context,
				  CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
				  sizeof(size_t) * n, NULL, NULL);
    LOOP1(q, compute_signs(d_short, pc, signs), n);
    clReleaseMemObject(pc);
    size_t *sgns = malloc(sizeof(size_t) * n);
    clEnqueueReadBuffer(q, signs, 0, 0, sizeof(size_t) * n, sgns,
			0, NULL, NULL);
    if(save) {
      double *vcs = malloc(sizeof(double) * d_short * d_short);
      for(size_t j = 0; j < d_short; j++)
	for(size_t l = 0; l < d_short; l++)
	  vcs[j * d_short + l] = l == j;
      cl_mem vecs = clCreateBuffer(gpu_context,
				   CL_MEM_READ_ONLY |
				   CL_MEM_COPY_HOST_PTR |
				   CL_MEM_HOST_NO_ACCESS,
				   sizeof(double) * d_short * d_short,
				   vcs, NULL);
      free(vcs);
      cl_mem vecs2 = clCreateBuffer(gpu_context,
				    CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS,
				    sizeof(double) * d_short * d_max,
				    NULL, NULL);
      LOOP2(sq, apply_permutation(d_short, d_max, perm_after_i, vecs, vecs2),
	    d_short, d_max);
      clReleaseMemObject(vecs);
      for(long j = rots_after - 1; j >= 0; j--)
	LOOP2(sq, apply_rotation(d_max, rot_js_a[j], rot_is_a[j], rot_as_a[j],
				 vecs2), d_short, rot_len_after);
      walsh(d_max, d_short, vecs2, sq);
      vecs = clCreateBuffer(gpu_context,
			    CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS,
			    sizeof(double) * (d_short * d + 1),
			    NULL, NULL);			    
      LOOP2(sq, apply_perm_inv(d_max, d, d_short * d, perm_before,
			       vecs2, vecs), d_short, d_max);
      clReleaseMemObject(vecs2);
      for(long j = rots_before - 1; j >= 0; j--)
	LOOP2(sq, apply_rotation(d, rot_js_b[j], rot_is_b[j], rot_as_b[j],
				 vecs), d_short, rot_len_before);
      clEnqueueReadBuffer(sq, vecs, 0, 0, sizeof(double) * d_short * d,
			  save->bases + i * d_short * d, 0, NULL, NULL);
      clReleaseMemObject(vecs);
    }
    clReleaseMemObject(perm_before);
    clReleaseMemObject(perm_after_i);
    for(size_t j = 0; j < rots_before; j++) {
      clReleaseMemObject(rot_is_b[j]);
      clReleaseMemObject(rot_js_b[j]);
      clReleaseMemObject(rot_as_b[j]);
    }
    free(rot_is_b);
    free(rot_js_b);
    free(rot_as_b);
    for(size_t j = 0; j < rots_after; j++) {
      clReleaseMemObject(rot_is_a[j]);
      clReleaseMemObject(rot_js_a[j]);
      clReleaseMemObject(rot_as_a[j]);
    }
    free(rot_is_a);
    free(rot_js_a);
    free(rot_as_a);
    size_t *counts = malloc(sizeof(size_t) << d_short);
    for(size_t j = 0; j < 1 << d_short; j++)
      counts[j] = 0;
    clFinish(q);
    for(size_t j = 0; j < n; j++)
      counts[sgns[j]]++;
    size_t tmax = counts[0];
    for(size_t j = 1; j < 1 << d_short; j++)
        if(tmax < counts[j])
	    tmax = counts[j];
    size_t *wh = malloc(sizeof(size_t) * tmax << d_short);
    for(size_t j = 0; j < 1 << d_short; j++)
	for(size_t l = counts[j]; l < tmax; l++)
	    wh[j * tmax + l] = n;
    for(size_t j = 0; j < n; j++)
        wh[sgns[j] * tmax + --counts[sgns[j]]] = j;
    free(sgns);
    free(counts);
    cl_mem which = clCreateBuffer(gpu_context,
				  CL_MEM_READ_ONLY |
				  CL_MEM_COPY_HOST_PTR |
				  CL_MEM_HOST_NO_ACCESS,
				  sizeof(double) * tmax << d_short,
				  wh, NULL);
    if(save != NULL) {
        save->which_par[i] = wh;
	save->par_maxes[i] = tmax;
    } else
      free(wh);
    cl_mem which_d = clCreateBuffer(gpu_context,
				    CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
				    sizeof(size_t) *
				    ((d_short + 1) * n * tmax + 1),
				    NULL, NULL);

    LOOP3(q, compute_which(d_short, tmax, signs, which, which_d),
	  n, d_short + 1, tmax);
    clReleaseMemObject(signs);
    clReleaseMemObject(which);
    cl_mem diffs = clCreateBuffer(gpu_context,
				  CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
				  sizeof(double) *
				  (d_short + 1) * n * d * tmax, NULL, NULL);
    LOOP3(q, compute_diffs_squared(d, (d_short + 1) * tmax, n, 0,
	                           which_d, pnts, pnts, diffs),
	     n, (d_short + 1) * tmax, d);
    cl_mem dists = clCreateBuffer(gpu_context,
				  CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
				  sizeof(double) *
				  (((d_short + 1) * n * tmax + 1)),
				  NULL, NULL);

    add_up_cols(d, (d_short + 1) * tmax, 0, n, diffs, dists, q);
    clReleaseMemObject(diffs);
    do_sort((d_short + 1) * tmax, n, which_d, dists, q);
    LOOP2(q, rdups((d_short + 1) * tmax, which_d, dists),
	       n, (d_short + 1) * tmax - 1);
    do_sort((d_short + 1) * tmax, n, which_d, dists, q);
    enqueueCopy2D(q, (d_short + 1) * tmax, k * tries, k * i, which_d,
		  pointers_out, n, k);
    clReleaseMemObject(which_d);
    enqueueCopy2D(q, (d_short + 1) * tmax, k * tries, k * i, dists,
		  dists_out, n, k);
    clReleaseMemObject(dists);
  }
  do_sort(k * tries, n, pointers_out, dists_out, q);
  LOOP2(q, rdups(k * tries, pointers_out, dists_out), n, k * tries - 1);
  do_sort(k * tries, n, pointers_out, dists_out, q);
  cl_mem nedge = clCreateBuffer(gpu_context,
				CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
				sizeof(size_t) * (n * k * (k + 1) + 1),
				NULL, NULL);
  cl_mem ndists = clCreateBuffer(gpu_context,
				 CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
				 sizeof(double) * (n * k * (k + 1) + 1),
				 NULL, NULL);
  LOOP3(q, supercharge(n, k * tries, k * tries, k,
		       pointers_out, pointers_out, nedge), n, k, k);
  enqueueCopy2D(q, k * tries, 0, k * (k + 1), pointers_out, nedge, n, k);
  clReleaseMemObject(pointers_out);
  enqueueCopy2D(q, k * tries, 0, k * (k + 1), dists_out, ndists, n, k);
  clReleaseMemObject(dists_out);  
  cl_mem diffs = clCreateBuffer(gpu_context,
				CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
				sizeof(double) * n * k * k * d, NULL, NULL);
  LOOP3(q, compute_diffs_squared(d, k * (k + 1), n, k,
			      nedge, pnts, pnts, diffs), n, k * k, d);
  clReleaseMemObject(pnts);
  add_up_cols(d, k * (k + 1), k, n, diffs, ndists, q);
  clReleaseMemObject(diffs);
  do_sort(k * (k + 1), n, nedge, ndists, q);
  LOOP2(q, rdups(k * (k + 1), nedge, ndists), n, k * (k + 1) - 1);
  do_sort(k * (k + 1), n, nedge, ndists, q);
  clReleaseMemObject(ndists);
  size_t *fedges = malloc(sizeof(size_t) * n * k);
  if(save != NULL)
    save->graph = fedges;
  size_t a[3] = {0, 0, 0}, c[3] = {n, k, sizeof(size_t)};
  clEnqueueReadBufferRect(q, nedge, 0, a, a, c, k * (k + 1), sizeof(size_t),
			  k, sizeof(size_t), fedges, 0, NULL, NULL);
  clReleaseMemObject(nedge);
  clFinish(q);
  clFinish(sq);
  clReleaseCommandQueue(q);
  clReleaseCommandQueue(sq);
  return(fedges);
}

// We now have points (n by d_long), save->graph (n by k),
// save->row_means (d_long), save->par_maxes (tries),
// save->which_par (tries, then 1 << d_short by save->par_maxes[i]),
// save->bases (tries by d_short by d_long), y (ycnt by d_long).

size_t *query_gpu(const save_t *save, const double *points,
	      size_t ycnt, double *y) {
  cl_command_queue q = clCreateCommandQueueWithProperties(gpu_context,
							  the_gpu, NULL, NULL);
  cl_mem y2 = clCreateBuffer(gpu_context,
			     CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR |
			     CL_MEM_HOST_NO_ACCESS,
			     sizeof(double) * save->d_long * ycnt, y, NULL);
  cl_mem rm = clCreateBuffer(gpu_context,
			     CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR |
			     CL_MEM_HOST_NO_ACCESS,
			     sizeof(double) * save->d_long,
			     save->row_means, NULL);  
  LOOP2(q, subtract_off(save->d_long, y2, rm), ycnt, save->d_long);
  clReleaseMemObject(rm);
  cl_mem bases = clCreateBuffer(gpu_context,
				CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR |
				CL_MEM_HOST_NO_ACCESS,
				sizeof(double) * save->tries * save->d_short *
				save->d_long, save->bases, NULL);
  cl_mem cprds = clCreateBuffer(gpu_context,
				CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
				sizeof(double) * save->tries * ycnt *
				save->d_short * save->d_long, NULL, NULL);

  LOOP3(q, prods(save->d_long, save->tries * save->d_short, y2, bases, cprds),
	ycnt, save->tries * save->d_short, save->d_long);
  clReleaseMemObject(bases);
  clReleaseMemObject(y2);
  cl_mem dprds = clCreateBuffer(gpu_context,
				CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
				sizeof(double) * save->tries * ycnt *
				save->d_short, NULL, NULL);
  add_up_cols(save->d_long, save->d_short, 0, save->tries * ycnt,
	      cprds, dprds, q);
  clReleaseMemObject(cprds);
  size_t *pmaxes = malloc(sizeof(size_t) * save->tries);
  size_t msofar = 0;
  for(int i = 0; i < save->tries; i++) {
    pmaxes[i] = msofar;
    msofar += save->par_maxes[i];
  }
  cl_mem signs = clCreateBuffer(gpu_context,
				CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
				sizeof(size_t) * save->tries * ycnt,
				NULL, NULL);
  LOOP1(q, compute_signs(save->d_short, dprds, signs),
      save->tries * ycnt);
  clReleaseMemObject(dprds);
  cl_mem ipts = clCreateBuffer(gpu_context,
			       CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
			       sizeof(size_t) * (msofar * (save->d_short + 1) *
						 ycnt + 1), NULL, NULL);

  for(int i = 0; i < save->tries; i++) {
    cl_mem ppts = clCreateBuffer(gpu_context,
				 CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
				 sizeof(size_t) * (save->d_short + 1) * ycnt *
				 save->par_maxes[i], NULL, NULL);
    cl_buffer_region bci = {i * ycnt, ycnt};
    cl_mem subsgns = clCreateSubBuffer(signs,
				       CL_MEM_READ_ONLY |
				       CL_MEM_HOST_NO_ACCESS,
				       CL_BUFFER_CREATE_TYPE_REGION,
				       &bci, NULL);
    cl_mem wp = clCreateBuffer(gpu_context,
			       CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR |
			       CL_MEM_HOST_NO_ACCESS,
			       sizeof(size_t) *
			       (save->par_maxes[i] << save->d_short),
			       save->which_par[i], NULL);
    LOOP3(q, compute_which(save->d_short, save->par_maxes[i],
			subsgns, wp,
			ppts),
	  ycnt, save->d_short + 1, save->par_maxes[i]);
    clReleaseMemObject(subsgns);
    clReleaseMemObject(wp);
    enqueueCopy2D(q, save->par_maxes[i] * (save->d_short + 1),
		  msofar * (save->d_short + 1),
		  pmaxes[i] * (save->d_short + 1),
		  ppts, ipts, ycnt, save->par_maxes[i] * (save->d_short + 1));
    clReleaseMemObject(ppts);
  }
  free(pmaxes);
  clReleaseMemObject(signs);
  y2 = clCreateBuffer(gpu_context,
		      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR |
		      CL_MEM_HOST_NO_ACCESS,
		      sizeof(double) * save->d_long * ycnt, y, NULL);
  cl_mem diffs = clCreateBuffer(gpu_context,
				CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
				sizeof(double) *
				msofar * (save->d_short + 1) * save->d_long
				* ycnt, NULL, NULL);
  cl_mem pnts = clCreateBuffer(gpu_context,
			       CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR |
			       CL_MEM_HOST_NO_ACCESS,
			       sizeof(double) * save->n * save->d_long,
			       (void *)points, NULL);
  LOOP3(q, compute_diffs_squared(save->d_long, msofar * (save->d_short + 1),
			      save->n, 0, ipts, y2, pnts, diffs),
	ycnt, msofar * (save->d_short + 1), save->d_long);
  cl_mem dpts = clCreateBuffer(gpu_context,
			       CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
			       sizeof(double) *
			       (msofar * (save->d_short + 1) * ycnt + 1),
			       NULL, NULL);
  add_up_cols(save->d_long, msofar * (save->d_short + 1), 0, ycnt,
	      diffs, dpts, q);
  clReleaseMemObject(diffs);
  do_sort(msofar * (save->d_short + 1), ycnt, ipts, dpts, q);
  LOOP2(q, rdups(msofar * (save->d_short + 1), ipts, dpts),
	ycnt, msofar * (save->d_short + 1) - 1);
  do_sort(msofar * (save->d_short + 1), ycnt, ipts, dpts, q);
  {
    cl_mem ipts2 = clCreateBuffer(gpu_context,
				  CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
				  sizeof(size_t) * (save->k * (save->k + 1) *
						    ycnt + 1), NULL, NULL);
    enqueueCopy2D(q, msofar * (save->d_short + 1), save->k * (save->k + 1), 0,
		  ipts, ipts2, ycnt, save->k);
    cl_mem graph = clCreateBuffer(gpu_context,
				  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR |
				  CL_MEM_HOST_NO_ACCESS,
				  sizeof(size_t) * save->n * save->k,
				  save->graph, NULL);
    LOOP3(q, supercharge(save->n, msofar * (save->d_short + 1),
			 save->k, save->k, ipts, graph, ipts2),
	  ycnt, save->k, save->k);
    clReleaseMemObject(ipts);
    clReleaseMemObject(graph);
    ipts = ipts2;
    cl_mem dpts2 = clCreateBuffer(gpu_context,
				  CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
				  sizeof(double) * (save->k * (save->k + 1) *
						    ycnt + 1), NULL, NULL);
    enqueueCopy2D(q, msofar * (save->d_short + 1), save->k * (save->k + 1), 0,
		  dpts, dpts2, ycnt, save->k);
    clReleaseMemObject(dpts);
    dpts = dpts2;
  }
  diffs = clCreateBuffer(gpu_context,
			 CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
			 sizeof(double) * save->k * save->k * save->d_long *
			 ycnt, NULL, NULL);
  LOOP3(q, compute_diffs_squared(save->d_long, save->k * (save->k + 1),
				 save->n, save->k, ipts, y2, pnts, diffs),
	ycnt, save->k * save->k, save->d_long);
  clReleaseMemObject(pnts);
  clReleaseMemObject(y2);
  add_up_cols(save->d_long, save->k * (save->k + 1), save->k, ycnt,
	      diffs, dpts, q);
  clReleaseMemObject(diffs);
  do_sort(save->k * (save->k + 1), ycnt, ipts, dpts, q);
  LOOP2(q, rdups(save->k * (save->k + 1), ipts, dpts),
	ycnt, save->k * (save->k + 1) - 1);
  do_sort(save->k * (save->k + 1), ycnt, ipts, dpts, q);
  clReleaseMemObject(dpts);
  size_t *results = malloc(sizeof(size_t) * ycnt * save->k);
  size_t a[3] = {0, 0, 0}, c[3] = {ycnt, save->k, sizeof(size_t)};
  clEnqueueReadBufferRect(q, ipts, 0, a, a, c, save->k * (save->k + 1),
			  sizeof(size_t), save->k, sizeof(size_t), results,
			  0, NULL, NULL);
  clReleaseMemObject(ipts);
  clFinish(q);
  clReleaseCommandQueue(q);
  return(results);
}
