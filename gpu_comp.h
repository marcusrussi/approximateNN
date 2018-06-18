#ifndef GPU_COMP
#define GPU_COMP
#include <OpenCL/opencl.h>
extern void gpu_init(void);
extern void gpu_cleanup(void);
extern cl_context gpu_context;

#endif
