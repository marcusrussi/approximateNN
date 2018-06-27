#include <math.h>
#include <stdlib.h>
#include "ocl2c.h"
#include "compute.cl"
#include "rand_pr.h"
#include "ann.h"
#include <string.h>

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

#define setup()

#define MAKE_COMMAND_QUEUE(a, b, c, d, q)

static void cp2D(size_t height_pre, size_t height_post, size_t start_post,
		 const void *from, void *to, size_t n, size_t k, size_t s) {
  const char *src = from;
  char *dst = to;
  for(size_t x = 0; x < n; x++)
    memcpy(dst + (x * height_post + start_post) * s,
	   src + x * height_pre * s, k * s);
}

static void add_up_rows(size_t d, size_t n, double *points, double *sums) {
  LOOP2(, add_rows_step_0(d, n, points, sums), n/2, d);
  for(size_t m = n >> 1; m >> 1; m >>= 1)
    LOOP2(, add_rows_step_n(d, m, sums), m/2, d);
}

static void walsh(size_t d, size_t n, double *a) {
  if(d == 1)
    return;
  int l = lg(d);
  size_t nth = max(d / 16, 1);
  for(int i = 0; i < l; i++)
    LOOP2(, apply_walsh_step(l, i, a), n, nth);
}

static void add_up_cols(size_t d, size_t k, size_t skip, size_t n,
		 double *mat, double *out) {
  for(size_t l = d; l >> 1; l >>= 1)
    LOOP3(, add_cols_step(d, l, k - skip, mat), n, k - skip, l / 2);
  for(size_t x = 0; x < n; x++)
    for(size_t y = 0; y < k - skip; y++)
      out[x * k + y + skip] = mat[(x * (k - skip) + y) * d];
}

static void do_sort(size_t k, size_t n, size_t *along, double *order) {
  int lk = lg(k);
  size_t nth = (size_t)1 << max(lk - 4, 0);
  for(int s = 0; s < lk; s++)
    for(int ss = s; ss >= 0; ss--)
      LOOP2(, sort_two_step(k, n, s, ss, along, order), n, nth);
}

static void *mbc(size_t k, const void *src) {
  void *dst = malloc(k);
  memcpy(dst, src, k);
  return(dst);
}

#define AddUpRows(q, ...) add_up_rows(__VA_ARGS__)
#define AddUpCols(q, ...) add_up_cols(__VA_ARGS__)
#define Walsh(q, ...) walsh(__VA_ARGS__)
#define DoSort(q, ...) do_sort(__VA_ARGS__)
#define enqueueCopyBuf(q, sz, src, dst) memcpy(dst, src, sz)
#define enqueueReadBuf(q, sz, src, dst) memcpy(dst, src, sz)
#define enqueueCopy2D(q, t, ...) cp2D(__VA_ARGS__, sizeof(t))
#define enqueueRead2D(q, t, ...) cp2D(__VA_ARGS__, sizeof(t))
#define BUFTYPE(t) t *
#define MB(type, sz) malloc(sizeof(type) * (sz))
#define MBC(type, sz, src) mbc(sizeof(type) * (sz), src)
#define MK_BUF_COPY_RW_NA(cont, type, sz, src) MBC(type, sz, src)
#define MK_BUF_COPY_RO_NA(cont, type, sz, src) MBC(type, sz, src)
#define MK_BUF_USE_RO_NA(cont, type, sz, src) (src)
#define MK_BUF_RW_NA(cont, type, sz) MB(type, sz)
#define MK_BUF_RW_RO(cont, type, sz) MB(type, sz)
#define MK_SUBBUF_RO_NA_REG(t, b, off, sz) ((b) + (off))
#define TYPE_OF_COMP cpu
#define relMem free
#define relMemU(m)
#define clFinish(q)
#define clReleaseCommandQueue(q)
#define waitForQueueThenCall(q, f, a) f(0, 0, a)
#define OINT char
#define OEVENT char

#include "alg.c"
