#define conc(a, b, c) a ## b ## c
#define concb(a, b, c) conc(a, b, c)
#define MK_NAME(x) concb(x, _, TYPE_OF_COMP)

typedef struct {
  BUFTYPE(size_t) *is, *js;
  BUFTYPE(double) *as;
} rot_info;

typedef struct {
  rot_info rb, ra;
  BUFTYPE(size_t) perm_b, perm_ai;
} ortho_info;

#ifndef ocl2c
#define FST_GONLY(x, ...) x(__VA_ARGS__)
#define TWO_GONLY(x, ...) x(__VA_ARGS__)
#else
#define FST_GONLY(x, y, ...) x(__VA_ARGS__)
#define TWO_GONLY(x, y, z, ...) x(__VA_ARGS__)
#endif

static rot_info FST_GONLY(make_rot_info,
			  cl_context c,
			  size_t rot_len,
			  size_t rots,
			  size_t dim) {
  size_t *ri = malloc(sizeof(size_t) * len);
  size_t *rj = malloc(sizeof(size_t) * len);
  double *ra = malloc(sizeof(double) * len);
  info roti;
  roti.is = malloc(sizeof(BUFTYPE(size_t)) * rots);
  roti.js = malloc(sizeof(BUFTYPE(size_t)) * rots);
  roti.as = malloc(sizeof(BUFTYPE(double)) * rots);
  for(size_t j = 0; j < rots; j++) {
    rand_rot(len, dim, ri, rj, ra);
    roti.is[j] = MK_BUF_COPY_RO_NA(c, size_t, len, ri);
    roti.js[j] = MK_BUF_COPY_RO_NA(c, size_t, len, rj);
    roti.as[j] = MK_BUF_COPY_RO_NA(c, double, len, ra);
  }
  free(ri);
  free(rj);
  free(ra);
  return(roti);
}

static ortho_info FST_GONLY(make_ortho_info,
			    cl_context c,
			    size_t rot_len_b,
			    size_t rots_b,
			    size_t rot_len_a,
			    size_t rots_a,
			    size_t dim_low,
			    size_t dim_high,
			    size_t dim_max) {
  ortho_info orti;
  size_t *perm;
  orti.rb = FST_GONLY(make_info, c, len_b, rots_b, dim_high);
  orti.ra = FST_GONLY(make_info, c, len_a, rots_a, dim_low);
  perm = rand_perm(d, d_max);
  orti.perm_b = MK_BUF_COPY_RO_NA(c, size_t, d_max, perm);
  free(perm);
  perm = rand_perm(d_short, d_max);
  orti.perm_ai = MK_BUF_COPY_RO_NA(c, size_t, d_max, perm);
  free(perm);
  return(orti);
}

static void free_rot_info(size_t rots, info r) {
  for(size_t i = 0; i < rots; i++) {
    relMem(r.is[i]);
    relMem(r.js[i]);
    relMem(r.as[i]);
  }
  free(r.is);
  free(r.js);
  free(r.as);
}

static void free_ortho_info(size_t rots_b, size_t rots_a, ortho_info *o) {
  free_info(rots_b, o->rb);
  free_info(rots_a, o->ra);
  relMem(o->perm_b);
  relMem(o->perm_ai);
}

typedef struct {
  size_t rots_b, rots_a, tries;
  ortho_info *o;
} cleanup_stuff;

static void cleanup(OEVENT e, OINT i, void *stuff) {
  cleanup_stuff *trash = (cleanup_stuff *)stuff;
  for(size_t i = 0; i < tries; i++)
    free_ortho_info(trash->rots_b, trash->rots_a, trash->o + i);
  free(trash->o);
}


// sgns may not have been written to.
// Ensure queue is finished before reading it.
static BUFTYPE(size_t) TWO_GONLY(run_initial, cl_context c, cl_command_queue q,
				 size_t n, size_t d_low,
				 size_t d_high, size_t d_max,
				 size_t rots_b, size_t rot_len_b,
				 size_t rots_a, size_t rot_len_a,
				 const ortho_info *o, BUFTYPE(double) points,
				 size_t **sgns) {
  BUFTYPE(double) pc = MK_BUF_RW_NA(c, double, n * d_high);
  enqueueCopyBuf(q, sizeof(double) * n * d_high, points, pc);
  for(size_t i = 0; j < rots_b; j++)   
    LOOP2(q, apply_rotation(d_high, o->rb.is[j], o->rb.js[j], o->rb.as[j], pc),
	  n, rot_len_b);
  BUFTYPE(double) pc2 = MK_BUF_RW_NA(c, double, n * d_max);
  LOOP2(q, apply_permutation(d_high, d_max, perm_before, pc, pc2), n, d_max);
  relMem(pc);
  Walsh(q, d_max, n, pc2);
  for(size_t j = 0; j < rots_after; j++)
    LOOP2(q, apply_rotation(d_max, o->ra.is[j], o->ra.js[j], o->ra.as[j], pc2),
	  n, rot_len_after);

  pc = MK_BUF_RW_NA(c, double, n * d_low + 1);
  LOOP2(q, apply_perm_inv(d_max, d_low, n * d_low, perm_after_i,
			  pc2, pc), n, d_max);
  relMem(pc2);
  BUFTYPE(size_t) signs = MK_BUF_RW_NA(c, size_t, n);
#ifndef ocl2c
  LOOP1(q, compute_signs(d_short, pc, signs), n);
#else
  LOOP1(q, compute_signs(d_short, (unsigned long *)pc, signs), n);
#endif
  relMem(pc);
  *sgns = malloc(sizeof(size_t) * n);
  enqueueReadBuf(q, sizeof(size_t) * n, signs, sgns);
  return(signs);
}

static void TWO_GONLY(save_vecs, cl_context c, cl_command_queue q,
		      size_t n, size_t d_low, size_t d_high, size_t d_max,
		      size_t rots_b, size_t rot_len_b,
		      size_t rots_a, size_t rot_len_a,
		      const ortho_info *o, double *loc) {
  double *vcs = malloc(sizeof(double) * d_low * d_low);
  for(size_t j = 0; j < d_low; j++)
    for(size_t l = 0; l < d_low; l++)
      vcs[j * d_low + l] = l == j;
  BUFTYPE(double) vecs = MK_BUF_COPY_RO_NA(c, double, d_low * d_low, vcs);
  free(vcs);
  BUFTYPE(double) vecs2 = MK_BUF_RW_NA(c, double, d_low * d_max);
  LOOP2(q, apply_permutation(d_low, d_max, o->perm_ai, vecs, vecs2),
	    d_low, d_max);
  relMem(vecs);
  for(long j = rots_a - 1; j >= 0; j--)
    LOOP2(q, apply_rotation(d_max, o->ra.js[j], o->ra.is[j], o->ra.as[j],
			    vecs2), d_low, rot_len_a);
  Walsh(q, d_max, d_low, vecs2);
  vecs = MK_BUF_RW_NA(c, double, d_low * d_high + 1);
  LOOP2(q, apply_perm_inv(d_max, d, d_low * d_high, perm_before,
			  vecs2, vecs), d_low, d_max);
  relMem(vecs2);
  for(long j = rots_before - 1; j >= 0; j--)
    LOOP2(q, apply_rotation(d_high, o->rb.js[j], o->ra.is[j], o->ra.as[j],
			     vecs), d_low, rot_len_before);
  enqueueReadBuf(q, sizeof(double) * d_low * d_high,
		 vecs, save->bases + i * d_low * d_high);
  relMem(vecs);
}

static size_t FST_GONLY(sort_and_uniq, cl_command_queue q, size_t n,
			size_t k, BUFTYPE(double) order,
			BUFTYPE(size_t) along) {
  DoSort(q, k, n, along, order);
  LOOP2(q, rdups(k, along, order), n, k - 1);
  DoSort(q, k, n, along, order);
}

/* Starting point: */
/* We have an array, points, that is n by d_long. */
/* We also have save, which is a save structure. */
size_t *MK_NAME(precomp) (size_t n, size_t k, size_t d, const double *points,
			  int tries, size_t rots_before, size_t rot_len_before,
			  size_t rots_after, size_t rot_len_after,
			  save_t *save) {
  setup();
  size_t d_short = ceil(log2((double)n / k));
  size_t d_max = d - 1;
  d_max |= d_max >> 1;
  d_max |= d_max >> 2;
  d_max |= d_max >> 4;
  d_max |= d_max >> 8;
  d_max |= d_max >> 16;
  d_max |= d_max >> 32;
  d_max++;
  if(d_short > d_max)
    d_short = d_max;
  MAKE_COMMAND_QUEUE(gpu_context, the_gpu, NULL, NULL, q);
  MAKE_COMMAND_QUEUE(gpu_context, the_gpu, NULL, NULL, sq);
  BUFTYPE(double) pnts =
    MK_BUF_COPY_RW_NA(gpu_context, double, n * d, points);
  BUFTYPE(double) row_sums;
  if(save == NULL)
    row_sums = MK_BUF_RW_RO(gpu_context, double, (n/2) * d);  
  else
    row_sums = MK_BUF_RW_NA(gpu_context, double, (n/2) * d);
  AddUpRows(q, d, n, pnts, row_sums);  
  LOOP1(q, divide_by_length(n, row_sums), d);
  LOOP2(q, subtract_off(d, pnts, row_sums), n, d);
  if(save != NULL) {
    save->tries = tries;
    save->n = n;
    save->k = k;
    save->d_short = d_short;
    save->d_long = d;
    save->row_means = malloc(sizeof(double) * d);
    enqueueReadBuf(sq, sizeof(double) * d, row_sums, save->row_means);
    save->which_par = malloc(sizeof(size_t *) * tries);
    save->par_maxes = malloc(sizeof(size_t) * tries);
    save->bases = malloc(sizeof(double) * tries * d_short * d);
  }
  relMem(row_sums);
  BUFTYPE(size_t) pointers_out =
    MK_BUF_RW_NA(gpu_context, size_t, n * k * tries + 1);
  BUFTYPE(double) dists_out =
    MK_BUF_RW_NA(gpu_context, double, n * k * tries + 1);
  for(int i = 0; i < tries; i++) {
    size_t *ri = malloc(sizeof(size_t) * rot_len_before);
    size_t *rj = malloc(sizeof(size_t) * rot_len_before);
    double *ra = malloc(sizeof(double) * rot_len_before);
    BUFTYPE(size_t) *rot_is_b = malloc(sizeof(BUFTYPE(size_t)) * rots_before);
    BUFTYPE(size_t) *rot_js_b = malloc(sizeof(BUFTYPE(size_t)) * rots_before);
    BUFTYPE(double) *rot_as_b = malloc(sizeof(BUFTYPE(double)) * rots_before);
    for(size_t j = 0; j < rots_before; j++) {
      rand_rot(rot_len_before, d, ri, rj, ra);
      rot_is_b[j] = MK_BUF_COPY_RO_NA(gpu_context, size_t, rot_len_before, ri);
      rot_js_b[j] = MK_BUF_COPY_RO_NA(gpu_context, size_t, rot_len_before, rj);
      rot_as_b[j] = MK_BUF_COPY_RO_NA(gpu_context, double, rot_len_before, ra);
    }
    free(ri);
    free(rj);
    free(ra);
    ri = malloc(sizeof(size_t) * rot_len_after);
    rj = malloc(sizeof(size_t) * rot_len_after);
    ra = malloc(sizeof(double) * rot_len_after);
    BUFTYPE(size_t) *rot_is_a = malloc(sizeof(BUFTYPE(size_t)) * rots_after);
    BUFTYPE(size_t) *rot_js_a = malloc(sizeof(BUFTYPE(size_t)) * rots_after);
    BUFTYPE(double) *rot_as_a = malloc(sizeof(BUFTYPE(double)) * rots_after);
    for(size_t j = 0; j < rots_after; j++) {
      rand_rot(rot_len_after, d_max, ri, rj, ra);
      rot_is_a[j] = MK_BUF_COPY_RO_NA(gpu_context, size_t, rot_len_after, ri);
      rot_js_a[j] = MK_BUF_COPY_RO_NA(gpu_context, size_t, rot_len_after, rj);
      rot_as_a[j] = MK_BUF_COPY_RO_NA(gpu_context, double, rot_len_after, ra);
    }
    free(ri);
    free(rj);
    free(ra);

    size_t *pb = rand_perm(d, d_max);
    BUFTYPE(size_t) perm_before =
      MK_BUF_COPY_RO_NA(gpu_context, size_t, d_max, pb);
    free(pb);
    size_t *pai = rand_perm(d_short, d_max);
    BUFTYPE(size_t) perm_after_i =
      MK_BUF_COPY_RO_NA(gpu_context, size_t, d_max, pai);
    free(pai);
    BUFTYPE(double) pc = MK_BUF_RW_NA(gpu_context, double, n * d);
    enqueueCopyBuf(q, sizeof(double) * n * d, pnts, pc);
    for(size_t j = 0; j < rots_before; j++)   
      LOOP2(q, apply_rotation(d, rot_is_b[j], rot_js_b[j], rot_as_b[j], pc),
	                n, rot_len_before);
    BUFTYPE(double) pc2 = MK_BUF_RW_NA(gpu_context, double, n * d_max);
    LOOP2(q, apply_permutation(d, d_max, perm_before, pc, pc2), n, d_max);
    relMem(pc);
    Walsh(q, d_max, n, pc2);
    for(size_t j = 0; j < rots_after; j++)
      LOOP2(q, apply_rotation(d_max, rot_is_a[j], rot_js_a[j], rot_as_a[j], pc2),
	    n, rot_len_after);

    pc = MK_BUF_RW_NA(gpu_context, double, n * d_short + 1);
    LOOP2(q, apply_perm_inv(d_max, d_short, n * d_short, perm_after_i,
			    pc2, pc), n, d_max);
    relMem(pc2);
    BUFTYPE(size_t) signs = MK_BUF_RW_NA(gpu_context, size_t, n);
#ifndef ocl2c
    LOOP1(q, compute_signs(d_short, pc, signs), n);
#else
    LOOP1(q, compute_signs(d_short, (unsigned long *)pc, signs), n);
#endif
    relMem(pc);
    size_t *sgns = malloc(sizeof(size_t) * n);
    enqueueReadBuf(q, sizeof(size_t) * n, signs, sgns);
    if(save) {
      double *vcs = malloc(sizeof(double) * d_short * d_short);
      for(size_t j = 0; j < d_short; j++)
	for(size_t l = 0; l < d_short; l++)
	  vcs[j * d_short + l] = l == j;
      BUFTYPE(double) vecs =
	MK_BUF_COPY_RO_NA(gpu_context, double, d_short * d_short, vcs);
      free(vcs);
      BUFTYPE(double) vecs2 =
	MK_BUF_RW_NA(gpu_context, double, d_short * d_max);
      LOOP2(sq, apply_permutation(d_short, d_max, perm_after_i, vecs, vecs2),
	    d_short, d_max);
      relMem(vecs);
      for(long j = rots_after - 1; j >= 0; j--)
	LOOP2(sq, apply_rotation(d_max, rot_js_a[j], rot_is_a[j], rot_as_a[j],
				 vecs2), d_short, rot_len_after);
      Walsh(sq, d_max, d_short, vecs2);
      vecs = MK_BUF_RW_NA(gpu_context, double, d_short * d + 1);
      LOOP2(sq, apply_perm_inv(d_max, d, d_short * d, perm_before,
			       vecs2, vecs), d_short, d_max);
      relMem(vecs2);
      for(long j = rots_before - 1; j >= 0; j--)
	LOOP2(sq, apply_rotation(d, rot_js_b[j], rot_is_b[j], rot_as_b[j],
				 vecs), d_short, rot_len_before);
      enqueueReadBuf(sq, sizeof(double) * d_short * d,
		     vecs, save->bases + i * d_short * d);
      relMem(vecs);
    }
    relMem(perm_before);
    relMem(perm_after_i);
    for(size_t j = 0; j < rots_before; j++) {
      relMem(rot_is_b[j]);
      relMem(rot_js_b[j]);
      relMem(rot_as_b[j]);
    }
    free(rot_is_b);
    free(rot_js_b);
    free(rot_as_b);
    for(size_t j = 0; j < rots_after; j++) {
      relMem(rot_is_a[j]);
      relMem(rot_js_a[j]);
      relMem(rot_as_a[j]);
    }
    free(rot_is_a);
    free(rot_js_a);
    free(rot_as_a);
    size_t *counts = malloc(sizeof(size_t) << d_short);
    for(size_t j = 0; j < 1 << d_short; j++)
      counts[j] = 0;
    clFinish(q);
    for(size_t j = 0; j < n; j++)
      counts[sgns[j]]++;
    size_t tmax = counts[0];
    for(size_t j = 1; j < 1 << d_short; j++)
        if(tmax < counts[j])
	    tmax = counts[j];
    size_t *wh = malloc(sizeof(size_t) * tmax << d_short);
    for(size_t j = 0; j < 1 << d_short; j++)
	for(size_t l = counts[j]; l < tmax; l++)
	    wh[j * tmax + l] = n;
    for(size_t j = 0; j < n; j++)
        wh[sgns[j] * tmax + --counts[sgns[j]]] = j;
    free(sgns);
    free(counts);
    BUFTYPE(size_t) which = MK_BUF_COPY_RO_NA(gpu_context, size_t,
					      tmax << d_short, wh);
    if(save != NULL) {
        save->which_par[i] = wh;
	save->par_maxes[i] = tmax;
    } else
      free(wh);
    BUFTYPE(size_t) which_d = MK_BUF_RW_NA(gpu_context, size_t,
					   (d_short + 1) * n * tmax + 1);
    LOOP3(q, compute_which(d_short, tmax, signs, which, which_d),
	  n, d_short + 1, tmax);
    relMem(signs);
    relMem(which);
    BUFTYPE(double) diffs = MK_BUF_RW_NA(gpu_context, double,
					 (d_short + 1) * n * d * tmax);
    LOOP3(q, compute_diffs_squared(d, (d_short + 1) * tmax, n, 0,
	                           which_d, pnts, pnts, diffs),
	     n, (d_short + 1) * tmax, d);
    BUFTYPE(double) dists = MK_BUF_RW_NA(gpu_context, double,
					 (d_short + 1) * n * tmax + 1);
    AddUpCols(q, d, (d_short + 1) * tmax, 0, n, diffs, dists);
    relMem(diffs);
    FST_GONLY(sort_and_uniq, q, n, (d_short + 1) * tmax, which_d, dists);
    enqueueCopy2D(q, size_t, (d_short + 1) * tmax, k * tries, k * i, which_d,
		  pointers_out, n, k);
    relMem(which_d);
    enqueueCopy2D(q, double, (d_short + 1) * tmax, k * tries, k * i, dists,
		  dists_out, n, k);
    relMem(dists);
  }
  FST_GONLY(sort_and_uniq, q, n, k * tries, pointers_out, dists_out)
  BUFTYPE(size_t) nedge = MK_BUF_RW_RO(gpu_context, size_t,
				       n * k * (k + 1) + 1);
  LOOP3(q, supercharge(n, k * tries, k * tries, k,
		       pointers_out, pointers_out, nedge), n, k, k);
  enqueueCopy2D(q, size_t, k * tries, k * (k + 1), 0, pointers_out, nedge,
		n, k);
  relMem(pointers_out);
  BUFTYPE(double) ndists = MK_BUF_RW_NA(gpu_context, double,
					n * k * (k + 1) + 1);
  enqueueCopy2D(q, double, k * tries, k * (k + 1), 0, dists_out, ndists, n, k);
  relMem(dists_out);
  BUFTYPE(double) diffs = MK_BUF_RW_NA(gpu_context, double, n * k * k * d);
  LOOP3(q, compute_diffs_squared(d, k * (k + 1), n, k,
			      nedge, pnts, pnts, diffs), n, k * k, d);
  relMem(pnts);
  AddUpCols(q, d, k * (k + 1), k, n, diffs, ndists);
  relMem(diffs);
#ifdef ocl2c
  nedge[n * k * (k + 1)] = 0;
  ndists[n * k * (k + 1)] = 0;
#endif
  FST_GONLY(sort_and_uniq, q, n, k * (k + 1), nedge, ndists);
  relMem(ndists);
  size_t *fedges = malloc(sizeof(size_t) * n * k);
  if(save != NULL)
    save->graph = fedges;
  enqueueRead2D(q, size_t, k * (k + 1), k, 0, nedge, fedges, n, k);
  relMem(nedge);
  clFinish(q);
  clFinish(sq);
  clReleaseCommandQueue(q);
  clReleaseCommandQueue(sq);
  return(fedges);
}

// We now have points (n by d_long), save->graph (n by k),
// save->row_means (d_long), save->par_maxes (tries),
// save->which_par (tries, then 1 << d_short by save->par_maxes[i]),
// save->bases (tries by d_short by d_long), y (ycnt by d_long).

size_t *MK_NAME(query) (const save_t *save, const double *points,
			size_t ycnt, const double *y) {
  MAKE_COMMAND_QUEUE(gpu_context, the_gpu, NULL, NULL, q);
  BUFTYPE(double) y2 = MK_BUF_COPY_RW_NA(gpu_context, double,
					 save->d_long * ycnt, y);
  BUFTYPE(const double) rm =
    MK_BUF_USE_RO_NA(gpu_context, double, save->d_long, save->row_means);
  LOOP2(q, subtract_off(save->d_long, y2, rm), ycnt, save->d_long);
  relMemU(rm);
  BUFTYPE(const double) bases = MK_BUF_USE_RO_NA(gpu_context, double,
						 save->tries * save->d_short *
						 save->d_long, save->bases);
  BUFTYPE(double) cprds = MK_BUF_RW_NA(gpu_context, double,
				       save->tries * ycnt *
				       save->d_short * save->d_long);
  LOOP3(q, prods(save->d_long, save->tries * save->d_short, y2, bases, cprds),
	ycnt, save->tries * save->d_short, save->d_long);
  relMemU(bases);
  relMem(y2);
  BUFTYPE(double) dprds = MK_BUF_RW_NA(gpu_context, double,
				       save->tries * ycnt * save->d_short);
  AddUpCols(q, save->d_long, save->d_short, 0, save->tries * ycnt,
	      cprds, dprds);
  relMem(cprds);
  size_t *pmaxes = malloc(sizeof(size_t) * save->tries);
  size_t msofar = 0;
  for(int i = 0; i < save->tries; i++) {
    pmaxes[i] = msofar;
    msofar += save->par_maxes[i];
  }
  BUFTYPE(size_t) signs = MK_BUF_RW_NA(gpu_context, double,
				       save->tries * ycnt);
#ifndef ocl2c
  LOOP1(q, compute_signs(save->d_short, dprds, signs), save->tries * ycnt);
#else
  LOOP1(q, compute_signs(save->d_short, (unsigned long *)dprds, signs),
	save->tries * ycnt);
#endif
  relMem(dprds);
  BUFTYPE(size_t) ipts = MK_BUF_RW_NA(gpu_context, size_t,
				      msofar * (save->d_short + 1) * ycnt + 1);
  for(int i = 0; i < save->tries; i++) {
    BUFTYPE(size_t) ppts = MK_BUF_RW_NA(gpu_context, size_t,
					(save->d_short + 1) * ycnt *
					save->par_maxes[i]);
    BUFTYPE(size_t) subsgns =
      MK_SUBBUF_RO_NA_REG(size_t, signs, i * ycnt, ycnt);
    BUFTYPE(const size_t) wp =
      MK_BUF_USE_RO_NA(gpu_context, size_t,
		       save->par_maxes[i] << save->d_short,
		       save->which_par[i]);
    LOOP3(q, compute_which(save->d_short, save->par_maxes[i],
			   subsgns, wp, ppts),
	  ycnt, save->d_short + 1, save->par_maxes[i]);
    relMemU(subsgns);
    relMemU(wp);
    enqueueCopy2D(q, size_t, save->par_maxes[i] * (save->d_short + 1),
		  msofar * (save->d_short + 1),
		  pmaxes[i] * (save->d_short + 1),
		  ppts, ipts, ycnt, save->par_maxes[i] * (save->d_short + 1));
    relMem(ppts);
  }
  free(pmaxes);
  relMem(signs);
  BUFTYPE(const double) y3 =
    MK_BUF_USE_RO_NA(gpu_context, double, save->d_long * ycnt, y);
  BUFTYPE(double) diffs = MK_BUF_RW_NA(gpu_context, double,
				       msofar * (save->d_short + 1) *
				       save->d_long * ycnt);
  BUFTYPE(const double) pnts =
    MK_BUF_USE_RO_NA(gpu_context, double, save->n * save->d_long, points);
  LOOP3(q, compute_diffs_squared(save->d_long, msofar * (save->d_short + 1),
			      save->n, 0, ipts, y3, pnts, diffs),
	ycnt, msofar * (save->d_short + 1), save->d_long);
  BUFTYPE(double) dpts =
    MK_BUF_RW_NA(gpu_context, double, msofar * (save->d_short + 1) * ycnt + 1);
  AddUpCols(q, save->d_long, msofar * (save->d_short + 1), 0, ycnt,
	      diffs, dpts);
  relMem(diffs);
  FST_GONLY(sort_and_uniq, q, ycnt, msofar * (save->d_short + 1), ipts, dpts);
  {
    BUFTYPE(size_t) ipts2 = MK_BUF_RW_NA(gpu_context, size_t,
					 save->k * (save->k + 1) * ycnt + 1);
    enqueueCopy2D(q, size_t, msofar * (save->d_short + 1),
		  save->k * (save->k + 1), 0, ipts, ipts2, ycnt, save->k);
    BUFTYPE(size_t) graph = MK_BUF_USE_RO_NA(gpu_context, size_t,
					     save->n * save->k, save->graph);
    LOOP3(q, supercharge(save->n, msofar * (save->d_short + 1),
			 save->k, save->k, ipts, graph, ipts2),
	  ycnt, save->k, save->k);
    relMem(ipts);
    relMemU(graph);
    ipts = ipts2;
    BUFTYPE(double) dpts2 = MK_BUF_RW_NA(gpu_context, double,
					 save->k * (save->k + 1) * ycnt + 1);
    enqueueCopy2D(q, double, msofar * (save->d_short + 1),
		  save->k * (save->k + 1), 0, dpts, dpts2, ycnt, save->k);
    relMem(dpts);
    dpts = dpts2;
  }
  diffs = MK_BUF_RW_NA(gpu_context, double,
		       save->k * save->k * save->d_long * ycnt);
  LOOP3(q, compute_diffs_squared(save->d_long, save->k * (save->k + 1),
				 save->n, save->k, ipts, y3, pnts, diffs),
	ycnt, save->k * save->k, save->d_long);
  relMemU(pnts);
  relMemU(y3);
  AddUpCols(q, save->d_long, save->k * (save->k + 1), save->k, ycnt,
	      diffs, dpts);
  relMem(diffs);
  FST_GONLY(sort_and_uniq, q, ycnt, save->k * (save->k + 1), ipts, dpts);
  relMem(dpts);
  size_t *results = malloc(sizeof(size_t) * ycnt * save->k);
  enqueueRead2D(q, size_t, save->k * (save->k + 1), save->k, 0,
		ipts, results, ycnt, save->k);
  relMem(ipts);
  clFinish(q);
  clReleaseCommandQueue(q);
  return(results);
}
