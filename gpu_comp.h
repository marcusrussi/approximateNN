#ifndef GPU_COMP
#define GPU_COMP
#include <OpenCL/opencl.h>
extern void gpu_init(void);
extern void gpu_cleanup(void);
extern void register_cleanup(void (*f)(void));
extern cl_context gpu_context;
extern cl_device_id the_gpu;
#endif
