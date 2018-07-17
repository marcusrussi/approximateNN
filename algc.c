#include <math.h>
#include <stdlib.h>
#include "rand_pr.h"
#include "ann.h"
#include <string.h>

#include "ocl2c.h"
#include "compute.cl"


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

static void *mbc(size_t k, const void *src) {
  void *dst = malloc(k);
  memcpy(dst, src, k);
  return(dst);
}

static void finish_cols(size_t h, size_t k, size_t skip, ftype *mat,
			ftype *out, size_t n) {
  for(size_t x = 0; x < n; x++)
    for(size_t y = 0; y < k - skip; y++)
      out[x * k + y + skip] = mat[(x * (k - skip) + y) * h];
}

#define fin_add_cols(q, ...) finish_cols(__VA_ARGS__)
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
