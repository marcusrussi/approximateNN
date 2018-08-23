#ifndef ANN
#define ANN
#include <stddef.h>
#include "ftype.h"

// A save data structure for randomised approximate nearest neighbor queries.
// Treat as opaque.
typedef struct {
  int tries;
  size_t n, k, d_short, d_long, **which_par, *par_maxes, *graph;
  ftype *row_means, *bases;
} save_t;

/**************************************************************************\
 * Computes nearest neighbors among a set of points;		       	  *
 * optionally precomputes a save structure.			 	  *
 *								 	  *
 * n: Number of points						 	  *
 * k: Number of nearest neighbors				 	  *
 * d: Dimensionality of the points     	       	       	       	 	  *
 * points: Pointer to the coordinates of the points;			  *
 * 	storage is all coordinates of point 0,				  *
 * followed by all coordinates of point 1, etc.				  *
 * tries: Number of times to rotate randomly and select candidates.	  *
 * rots_before: How many times to rotate by random amounts in some number *
 * 	of disjoint coordinate planes before Walsh transforming.	  *
 * rot_len_before: How many disjoint coordinate planes are used		  *
 * 	in each such rotation.						  *
 * rots_after, rot_len_after: Same as rot*_before,			  *
 * 	except after the Walsh transform.				  *
 * save: Pointer to the save data structure, or NULL.			  *
 * 	If not NULL, information for queries will be stored here.	  *
 * dists: Pointer to pointer to distances, or NULL.			  *
 * 	If not null, *dists will be set to the address of		  *
 * 	a new array of floats, which will have the			  *
 * 	distances from points to their neighbors placed inside.		  *
 * use_cpu: If nonzero, forces use of CPU (slower,			  *
 * 	but guaranteed to use only one core).				  *
 *	If compiled off CPUonly branch, this is ignored.	       	  *
 * Returns a pointer to an array of nearest neighbors.			  *
 * Storage here and in *dists is neighbors of 0 in order,		  *
 * then of 1 in order, etc.						  *
\**************************************************************************/
extern size_t *precomp(size_t n, size_t k, size_t d, const ftype *points,
		       int tries, size_t rots_before, size_t rot_len_before,
		       size_t rots_after, size_t rot_len_after, save_t *save,
		       ftype **dists, char use_cpu);

/*************************************************************************\
 * Computes nearest neighbors of query points. 	       	       	       	 *
 * save: Pointer to save data structure.				 *
 * points: Pointer to the array of points.				 *
 * 	Must correspond to save structure.				 *
 * ycnt: How many points are we querying?				 *
 * y: Pointer to array of query points 					 *
 * 	(those we want to find nearest neighbors of).			 *
 * dists, use_cpu, return value: See precomp.				 *
\*************************************************************************/
extern size_t *query(const save_t *save, const ftype *points,
		     size_t ycnt, const ftype *y, ftype **dists, char use_cpu);

// Frees a save data structure.
extern void free_save(save_t *save);
#endif
