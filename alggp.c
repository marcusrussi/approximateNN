#include "ann.h"
#include "rand_pr.h"
#include "gpu_comp.h"

#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define XSTR(s) STR(s)
#define STR(s) #s

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

#ifdef SUPPORT_OPENCL_V1_2
#define clCreateCommandQueueWithProperties(a, b, c, d) \
  clCreateCommandQueue(a, b, 0, d)
#endif

#define MAKE_COMMAND_QUEUE(a, b, c, d, q) cl_command_queue q = \
    clCreateCommandQueueWithProperties(a, b, c, d)

#define max(a, b) ((a) < (b)?b:(a))

cl_program compute;
#ifdef SUPPORT_OPENCL_V1_2
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

static char c = 0;
static void teardown(void) {
  if(!c)
    return;
  c = 0;
#ifndef SUPPORT_OPENCL_V1_2
  clReleaseKernel(add_rows_step_0);
  clReleaseKernel(add_rows_step_n);
  clReleaseKernel(divide_by_length);
  clReleaseKernel(subtract_off);
  clReleaseKernel(apply_rotation);
  clReleaseKernel(apply_permutation);
  clReleaseKernel(apply_perm_inv);
  clReleaseKernel(apply_walsh_step);
  clReleaseKernel(compute_diffs_squared);
  clReleaseKernel(add_cols_step);
  clReleaseKernel(add_cols_fin);
  clReleaseKernel(sort_two_step);
  clReleaseKernel(rdups);
  clReleaseKernel(compute_signs);
  clReleaseKernel(compute_which);
  clReleaseKernel(supercharge);
  clReleaseKernel(prods);
#endif
  clReleaseProgram(compute);
}
static void setup(void) {
  gpu_init();
  if(c)
    return;
  c = 1;
  // LINE_COUNT_OCL is replaced by length of compute.cl.
  char *src_full[LINE_COUNT_OCL + 2] = {
    "#define ftype " XSTR(ftype) "\n",
    "#define as_i_ftype as_u" XSTR(i_ftype) "\n",
    INSERT_COMP_HERE // this line is replaced by the contents of compute.cl
  };
  cl_int error;
  compute = clCreateProgramWithSource(gpu_context, LINE_COUNT_OCL + 2,
				      (const char **)src_full,
				      NULL, &error);
  if(error != CL_SUCCESS) {
    fprintf(stderr, "Error loading OpenCL code.\n");
    exit(1);
  }
  if(clBuildProgram(compute, 0, NULL,
#ifdef SUPPORT_OPENCL_V1_2
		    "",
#else
		    "-cl-std=CL2.0",
#endif		    
		    NULL, NULL) != CL_SUCCESS) {
    fprintf(stderr, "Error building program.\n");
    exit(1);
  }
#ifndef SUPPORT_OPENCL_V1_2
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
  register_cleanup(teardown);
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

static void cp2D(cl_command_queue q,
		 size_t height_pre,
		 size_t height_post, size_t start_post,
		 cl_mem from, cl_mem to, size_t n, size_t k,
		 size_t s) {
  size_t src_ori[3] = {0, 0, 0};
  size_t dst_ori[3] = {0, start_post, 0};
  size_t reg[3] = {s, k, n};
  if(clEnqueueCopyBufferRect(q, from, to, src_ori, dst_ori, reg,
			     s, height_pre * s,
			     s, height_post * s,
			     0, NULL, NULL) != CL_SUCCESS)
    fprintf(stderr, "Failed enqueue of copy.\n"), exit(1);
}

#define enqueueCopy2D(q, t, ...) cp2D(q, __VA_ARGS__, sizeof(t))

static void rd2D(cl_command_queue q,
		 size_t height_pre,
		 size_t height_post, size_t start_post,
		 cl_mem from, void *to, size_t n, size_t k,
		 size_t s) {
  size_t src_ori[3] = {0, 0, 0};
  size_t dst_ori[3] = {0, start_post, 0};
  size_t reg[3] = {s, k, n};
  if(clEnqueueReadBufferRect(q, from, CL_FALSE, src_ori, dst_ori, reg,
			     s, height_pre * s,
			     s, height_post * s,
			     to, 0, NULL, NULL) != CL_SUCCESS)
    fprintf(stderr, "Failed enqueue of copy.\n"), exit(1);
}

#define enqueueRead2D(q, t, ...) rd2D(q, __VA_ARGS__, sizeof(t))

static void enqueueCopyBuf(cl_command_queue q,
			   size_t sz, cl_mem from, cl_mem to) {
  if(clEnqueueCopyBuffer(q, from, to, 0, 0, sz,
			 0, NULL, NULL) != CL_SUCCESS)
    fprintf(stderr, "Failed enqueue of copy.\n"), exit(1);
}

static void enqueueReadBuf(cl_command_queue q,
			   size_t sz, cl_mem from, void *to) {
  if(clEnqueueReadBuffer(q, from, 0, 0, sz, to, 0, NULL, NULL) != CL_SUCCESS)
    fprintf(stderr, "Failed enqueue of read.\n"), exit(1);
}

static void enqueueFinAC(cl_command_queue q, size_t height, size_t k,
			 size_t skip, cl_mem from, cl_mem to, size_t n) {
  size_t src_ori[3] = {0, 0, 0};
  size_t dst_ori[3] = {0, skip, 0};
  size_t reg[3] = {sizeof(ftype), k - skip, n};
  if(clEnqueueCopyBufferRect(q, from, to, src_ori, dst_ori, reg,
			     sizeof(ftype) * height, 0,
			     0, sizeof(ftype) * k,
			     0, NULL, NULL) != CL_SUCCESS)
    fprintf(stderr, "Failed enqueue of copy.\n"), exit(1);
}

#define ska(k, i, o) setKerArg(k, i, sizeof(o), &o)

static void setKerArg(cl_kernel k, cl_uint ai, size_t as, const void *av) {
  if(clSetKernelArg(k, ai, as, av) != CL_SUCCESS)
    fprintf(stderr, "Error setting kernel args.\n"), exit(1);
}


static cl_kernel cr_add_rows_step_0(size_t h, size_t l,
				    cl_mem a, cl_mem r) {
  cl_kernel k = clone_kernel(add_rows_step_0);
  ska(k, 0, h);
  ska(k, 1, l);
  ska(k, 2, a);
  ska(k, 3, r);
  return(k);
}
static cl_kernel cr_add_rows_step_n(size_t h, size_t l, cl_mem r) {
  cl_kernel k = clone_kernel(add_rows_step_n);
  ska(k, 0, h);
  ska(k, 1, l);
  ska(k, 2, r);
  return(k);
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

static cl_kernel cr_apply_perm_inv(size_t hb, size_t ha,
				   cl_mem p, cl_mem a, cl_mem r) {
  cl_kernel k = clone_kernel(apply_perm_inv);
  ska(k, 0, hb);
  ska(k, 1, ha);
  ska(k, 2, p);
  ska(k, 3, a);
  ska(k, 4, r);
  return(k);
}

static cl_kernel cr_apply_walsh_step(size_t l, size_t i, cl_mem a) {
  cl_kernel k = clone_kernel(apply_walsh_step);
  ska(k, 0, l);
  ska(k, 1, i);
  ska(k, 2, a);
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

static cl_kernel cr_add_cols_step(size_t h, size_t s,
				  size_t k, cl_mem m) {
  cl_kernel kk = clone_kernel(add_cols_step);
  ska(kk, 0, h);
  ska(kk, 1, s);
  ska(kk, 2, k);
  ska(kk, 3, m);
  return(kk);
}

static cl_kernel cr_sort_two_step(size_t c, size_t s,
				  size_t ss, cl_mem a,
				  cl_mem o) {
  cl_kernel k = clone_kernel(sort_two_step);
  ska(k, 0, c);
  ska(k, 1, s);
  ska(k, 2, ss);
  ska(k, 3, a);
  ska(k, 4, o);
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

static cl_mem subbuf(cl_mem b, size_t off, size_t sz) {
  cl_buffer_region bci = {off, sz};
  return(clCreateSubBuffer(b,
			   CL_MEM_READ_ONLY |
			   CL_MEM_HOST_NO_ACCESS,
			   CL_BUFFER_CREATE_TYPE_REGION,
			   &bci, NULL));
}

static void waitForQueueThenCall(cl_command_queue q,
				 void (*f)(cl_event e, cl_int s, void *d),
				 void *a) {
  cl_event e;
  clEnqueueMarkerWithWaitList(q, 0, NULL, &e);
  clSetEventCallback(e, CL_COMPLETE, f, a);
}
					
#define OINT cl_int
#define OEVENT cl_event
#define LOOP1(q, a, x) enqueue1D(q, cr_ ## a, x)
#define LOOP2(q, a, x, y) enqueue2D(q, cr_ ## a, x, y)
#define LOOP3(q, a, x, y, z) enqueue3D(q, cr_ ## a, x, y, z)

#define fin_add_cols enqueueFinAC
#define BUFTYPE(t) cl_mem
#define relMem clReleaseMemObject
#define relMemU clReleaseMemObject
#define MK_BUF_COPY_RW_NA(cont, type, sz, src) \
    clCreateBuffer(cont, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | \
		   CL_MEM_HOST_NO_ACCESS, sizeof(type) * (sz), \
		   (void *)src, NULL)
#define MK_BUF_COPY_RO_NA(cont, type, sz, src) \
    clCreateBuffer(cont, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | \
		   CL_MEM_HOST_NO_ACCESS, sizeof(type) * (sz), \
		   (void *)src, NULL)
#define MK_BUF_USE_RO_NA(cont, type, sz, src) \
    clCreateBuffer(cont, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | \
		   CL_MEM_HOST_NO_ACCESS, sizeof(type) * (sz), \
		   (void *)src, NULL)
#define MK_BUF_RW_NA(cont, type, sz)			    \
    clCreateBuffer(cont, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, \
		   sizeof(type) * (sz), NULL, NULL)
#define MK_BUF_RW_RO(cont, type, sz) \
    clCreateBuffer(cont, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, \
		   sizeof(type) * (sz), NULL, NULL)
#define MK_SUBBUF_RO_NA_REG(t, b, o, s) subbuf(b, (o) * sizeof(t), \
					       (s) * sizeof(t))
#define TYPE_OF_COMP gpu
#include "alg.c"
