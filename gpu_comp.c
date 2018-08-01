#include "gpu_comp.h"
#include <stdio.h>
#include <stdlib.h>
cl_context gpu_context;
cl_device_id the_gpu;

typedef struct CUP {
  void (*f)(void);
  struct CUP *next;
} cleanup;

static cleanup *cup = NULL;

static char init = 0;
void diequick(const char *errinfo, const void *a,
		      size_t b, void *c) {
  fprintf(stderr, "Error on GPU: %s", errinfo);
  a = a; b = b; c = c;
  exit(1);
}

void gpu_init(void) {
  if(init)
    return;
  init = 1;
  cl_uint nplat;
  if(clGetPlatformIDs(0, NULL, &nplat) != CL_SUCCESS) {
    fprintf(stderr, "Error running clGetPlatformIDs.\n");
    exit(1);
  }
  cl_platform_id *plats = malloc(sizeof(cl_platform_id) * nplat);
  if(clGetPlatformIDs(nplat, plats, NULL) != CL_SUCCESS) {
    fprintf(stderr, "Error running clGetPlatformIDs.\n");
    exit(1);
  }
  cl_device_type devtypes[2] =
    { CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, CL_DEVICE_TYPE_CPU };
  for(int qq = 1; qq < 2; qq++)
    for(cl_uint i = 0; i < nplat; i++) {
      cl_uint ndev;
      cl_int err;
      if((err = clGetDeviceIDs(plats[i], devtypes[qq],
			       0, NULL, &ndev)) != CL_SUCCESS) {
	if(err != CL_DEVICE_NOT_FOUND)
	  continue;
	fprintf(stderr, "Error running clGetDeviceIDs.\n");
	exit(1);
      }
      cl_device_id *devs = malloc(sizeof(cl_device_id) * ndev);
      if(clGetDeviceIDs(plats[i], devtypes[qq],
			ndev, devs, NULL) != CL_SUCCESS) {
	fprintf(stderr, "Error running clGetDeviceIDs.\n");
	exit(1);
      }
      for(cl_uint j = 0; j < ndev; j++) {
#ifndef USE_FLOAT
	cl_device_fp_config f;
	if(clGetDeviceInfo(devs[j], CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(f), &f,
			   NULL) != CL_SUCCESS) {
	  fprintf(stderr, "Error running clGetDeviceInfo.\n");
	  exit(1);
	}
	if(f & CL_FP_INF_NAN)
#endif
	  {
	    cl_platform_id mplat = plats[i];
	    the_gpu = devs[j];
	    cl_context_properties props[3];
	    props[0] = CL_CONTEXT_PLATFORM;
	    props[1] = (cl_context_properties)mplat;
	    props[2] = 0;
	    cl_int error;
	    gpu_context = clCreateContext(props, 1, &the_gpu,
					  diequick, NULL, &error);
	    if(error != CL_SUCCESS)
	      fprintf(stderr, "Error creating OpenCL context.\n");
	    else {
	      free(plats);
	      free(devs);
	      return;
	    }
	  }
      }
      free(devs);
    }
  fprintf(stderr, "No "
#ifndef USE_FLOAT
	  "double-supporting "
#endif
	  "GPU found.\n");
  exit(1);
}

void register_cleanup(void (*f)(void)) {
  if(init) {
    cleanup *c = malloc(sizeof(cleanup));
    c->f = f;
    c->next = cup;
    cup = c;
  } else
    f();
}

void gpu_cleanup(void) {
  if(!init)
    return;
  init = 0;
  while(cup != NULL) {
    cup->f();
    cleanup *c = cup->next;
    free(cup);
    cup = c;
  }
  clReleaseContext(gpu_context);
}
