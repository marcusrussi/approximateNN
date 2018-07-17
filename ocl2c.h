#ifndef ocl2c
#define ocl2c
// This must be included, followed by the OpenCL file.

#define __kernel static
#define __global
#define rsqrt(x) (1/sqrt(x))
#define barrier(x)

#define sincos(a, b) (*(b) = cos(a), sin(a))

static size_t glob_x;
static size_t glob_y;
static size_t glob_z;

typedef unsigned long ulong;

#define LOOP1(q, f, mx) for(glob_x = 0; glob_x < (mx); glob_x++) f
#define LOOP2(q, f, mx, my) for(glob_y = 0; glob_y < (my); glob_y++)\
    LOOP1(q, f, mx)
#define LOOP3(q, f, mx, my, mz) for(glob_z = 0; glob_z < (mz); glob_z++)\
    LOOP2(q, f, mx, my)

#define get_global_id(i) (i == 0? glob_x : i == 1? glob_y : glob_z)

typedef unsigned uint;

#ifndef USE_FLOAT
static ulong as_ulong(double d) {
  return(*(ulong *)&d);
}
#else
static uint as_uint(float f) {
  return(*(uint *)&f);
}
#endif

#define CONCATENATE(a, b) a ## b
#define XCONCAT(a, b) CONCATENATE(a, b)
#define as_i_ftype XCONCAT(as_u, i_ftype)


#endif
