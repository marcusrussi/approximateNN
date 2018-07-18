# approximateNN
Approximate nearest neighbors implementation, in C/OpenCL.

`alg.c` contains the algorithm, in a way that can easily be transformed
into pure C or usage of the OpenCL API.

`algc.c` contains the code to turn the algorithm into pure C;
`algc.h` contains extern declarations for `precomp_cpu` and `query_cpu`.

`alggp.c` contains the code to cause the algorithm to use the OpenCL API.
(It won't compile directly; first the contents of `compute.cl`
must be interpolated -- see the makefile for how.)
`algg.h` contains extern declarations for `precomp_gpu` and `query_gpu`.

`algorithm.txt` is a very weird pseudocode version of the algorithm.

`ann.c` contains code to allow a user to use either CPU or GPU versions with
the same function call;
`ann.h` contains the declarations for `precomp` and `query`,
as well as a definition of `save_t` (the save data structure)
and a declaration of a function to free a `save_t`.

`compare_results.c` is a test to ensure that the OpenCL and pure C versions 
give the same result.

`compute.cl` contains the OpenCL code.

`ftype.h` contains code to enable easy switching between float and double
(separate compilation necessary).

`gpu_comp.c` contains code to setup and teardown OpenCL.
`gpu_comp.h` contains declarations.

`ocl2c.h` is used to convert OpenCL C to regular C.

`randNorm.c` is used by the test code to create random numbers with a
standard normal distribution.
`randNorm.h` has bindings.

`rand_pr.c` contains code to create random subpermutation matrices and
rotation matrices in a fixed format.
`rand_pr.h` has bindings for this.

`test_correctness.c` contains code to check how well the algorithm does
on random data.

`time_results.c` contains code to time the algorithm on random data.

`timing.h` is a hack to allow identical code to be used on OSX and Linux for
very simple timing purposes.
