#include "gpu_comp.h"
#include <stdio.h>
#include <stdlib.h>
cl_context gpu_context;
cl_device_id the_gpu;

void diequick(const char *errinfo, const void *a,
		      size_t b, void *c) {
  fprintf(stderr, "Error on GPU: %s", errinfo);
  a = a; b = b; c = c;
  exit(1);
}

void gpu_init(void) {
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
  for(cl_uint i = 0; i < nplat; i++) {
    cl_uint ndev;
    if(clGetDeviceIDs(plats[i],
#ifdef OSX
		      CL_DEVICE_TYPE_CPU |
#endif
		      CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR,
		      0, NULL, &ndev) != CL_SUCCESS) {
      fprintf(stderr, "Error running clGetDeviceIDs.\n");
      exit(1);
    }
    cl_device_id *devs = malloc(sizeof(cl_device_id) * ndev);
    if(clGetDeviceIDs(plats[i],
#ifdef OSX
		      CL_DEVICE_TYPE_CPU |
#endif
		      CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR,
		      ndev, devs, NULL) != CL_SUCCESS) {
      fprintf(stderr, "Error running clGetDeviceIDs.\n");
      exit(1);
    }
    for(cl_uint j = 0; j < ndev; j++) {
      cl_device_fp_config f;
      if(clGetDeviceInfo(devs[j], CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(f), &f,
			 NULL) != CL_SUCCESS) {
	fprintf(stderr, "Error running clGetDeviceInfo.\n");
	exit(1);
      }
      if(f & CL_FP_INF_NAN) {
	cl_platform_id mplat = plats[i];
	the_gpu = devs[j];
	free(plats);
	free(devs);
	cl_context_properties props[3];
	props[0] = CL_CONTEXT_PLATFORM;
	props[1] = (cl_context_properties)mplat;
	props[2] = 0;
	cl_int error;
	gpu_context = clCreateContext(props, 1, &the_gpu,
				      diequick, NULL, &error);
	if(error != CL_SUCCESS) {
	  fprintf(stderr, "Error creating OpenCL context.\n");
	  exit(1);
	}
	return;
      }
    }
    free(devs);
  }
  fprintf(stderr, "No double-supporting GPU found.");
  exit(1);
}

void gpu_cleanup(void) {
  clReleaseContext(gpu_context);
}
