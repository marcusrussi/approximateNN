#ifndef GPU_COMP
#define GPU_COMP
#ifdef OSX
#include <OpenCL/opencl.h>
#else
#ifndef LINUX
#warning "Neither -DOSX nor -DLINUX supplied, assuming LINUX."
#endif
#include <CL/opencl.h>
#endif
extern void gpu_init(void);
extern void gpu_cleanup(void);
extern void register_cleanup(void (*f)(void));
extern cl_context gpu_context;
extern cl_device_id the_gpu;
#endif
