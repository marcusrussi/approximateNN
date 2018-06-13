#ifndef TIMING
#define TIMING
#include <stdint.h>
#ifdef OSX
#include <mach/mach_time.h>
typedef uint64_t tval;
#define gettm(t) t = mach_absolute_time()
#define td(t1, t2) (t2 - t1)
#else
#ifndef LINUX
#warning "Neither -DOSX nor -DLINUX supplied, assuming LINUX."
#endif
#include <time.h>
typedef struct timespec tval;
#define gettm(t) clock_gettime(CLOCK_MONOTONIC, &t)
#define td(t1, t2) (((uint64_t)t2.tv_sec * 1000000000 + t2.tv_nsec) -\
		    ((uint64_t)t1.tv_sec * 1000000000 + t1.tv_nsec))
#endif
#endif
