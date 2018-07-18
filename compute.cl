// When I write func(params)(x, y = 1, z = 1),
// I mean, call func(params), and have one thread for every
// i, j, k that is possible where 0 <= i < x, 0 <= j < y, 0 <= k < z.
// When I write func(params)(x, y = 1, z = 1)(c),
// I mean the same, but coordinate c must be kept together.

// Add rows:
//   add_rows_step_0(d, n, a, r)(n/2, d);
//   for(m = n >> 1; m; m >>= 1)
//     add_rows_step_n(d, m, r)(m/2, d);

// If a is n by d, r is n/2 by d,
// Θ(1) depth, Θ(nd) work,
// add_rows_step_0(d, n, a, r)(n/2, d);
__kernel void add_rows_step_0(const size_t height, const size_t len,
			      __global const ftype *a,
			      __global ftype *r) {
  size_t x = get_global_id(0), y = get_global_id(1);
  ftype g = len & !x? a[(len - 1) * height + y] : 0;
  r[x * height + y] = a[x * height + y] + a[(x + len / 2) * height + y] + g;  
}

// If r is n by d,
// Θ(1) depth, Θ(nd) work,
// add_rows_step_n(d, n, r)(n/2, d);
__kernel void add_rows_step_n(const size_t height, const size_t len,
			      __global ftype *r) {
  size_t x = get_global_id(0), y = get_global_id(1);
  ftype g = !x && len % 2 ? r[(len - 1) * height + y] : 0;
  r[x * height + y] += r[(x + len / 2) * height + y] + g;
}

// If r is length d,
// Θ(1) depth, Θ(d) work,
// divide_by_length(n, r)(d);
__kernel void divide_by_length(const size_t length, __global ftype *r) {
  size_t x = get_global_id(0);
  r[x] /= length;
}

// If r is length d and m is n by d,
// Θ(1) depth, Θ(nd) work,
// subtract_off(d, m, r)(n, d);
__kernel void subtract_off(const size_t height,
			   __global ftype *a,
			   __global const ftype *r) {
  size_t x = get_global_id(0), y = get_global_id(1);
  a[x * height + y] -= r[y];
}

// Swap i/j to get reverse rotation.
// If i, j, a are length p and m is n by h,
// Θ(1) depth, Θ(np) work,
// apply_rotation(h, i, j, a, m)(n, p);
__kernel void apply_rotation(const size_t height,
			     __global const size_t *i, // first coordinate
			     __global const size_t *j, // second coordinate
			     __global const ftype *ang,
			     __global ftype *a) {
  size_t x = get_global_id(0) * height, y = get_global_id(1);
  size_t k = i[y], l = j[y];
  ftype c;
  ftype s = sincos(ang[y], &c);
  ftype q = a[x + k] * c - a[x + l] * s;
  ftype r = a[x + k] * s + a[x + l] * c;
  a[x + k] = q;
  a[x + l] = r;
}

// Note: if perm[i] = j, carries e_j to e_i.
// If j is too big, annihilates e_j.

// If perm is size e, a is size n by d,
// r is size n by e,
// Θ(1) depth, Θ(ne) work,
// apply_permutation(d, e, perm, a, r)(n, e);
__kernel void apply_permutation(const size_t height_pre,
				const size_t height_post,
				__global const size_t *perm,
				__global const ftype *a,
				__global ftype *r) {
  size_t x = get_global_id(0), y = get_global_id(1);
  ftype g = perm[y] < height_pre? a[x * height_pre + perm[y]] : 0;
  r[x * height_post + y] = g;
}
  
// Undoes apply_permutation.
__kernel void apply_perm_inv(const size_t height_pre,
			     const size_t height_post,
			     __global const size_t *perm,
			     __global const ftype *a,
			     __global ftype *r) {
  size_t x = get_global_id(0), y = get_global_id(1);
  if(perm[y] < height_post)
    r[x * height_post + perm[y]] = a[x * height_pre + y];
}

// If a is n by 1 << l,
// Θ(1) depth, Θ(n * 2^l) work,
// apply_walsh_step(l, s, a)(n, 1 << (l - 4));
__kernel void apply_walsh_step(const size_t lheight,
			       const size_t step,
			       __global ftype *a) {
  ftype rsr = rsqrt(2.0);
  size_t x = get_global_id(0) << lheight, y = get_global_id(1);
  if(!lheight)
    return;
  for(int j = 0; j < 8 && j < 1 << lheight; j++) {
      size_t y1 = y << 3 | j;
      size_t yh = (y1 >> step) << step;
      size_t yl = y1 ^ yh;
      size_t ca = x | yh << 1 | yl;
      size_t cb = ca | 1 << step;
      ftype alpha = a[ca], beta = a[cb];
      a[ca] = (alpha + beta) / (step % 2 + 1);
      a[cb] = (alpha - beta) / (step % 2 + 1);
  }
  if(step == 0 && lheight % 2)
    for(int j = 0; j < 16 && j < 1 << lheight; j++)
      a[x | y << 4 | j] *= rsr;
}
 
// dims: each point is d-dimensional
// count: how many points do we compute distances to (k)
// which: From x, compute distances to which[x][y].
// points: points[x][z] contains proper coordinate.
// pointsa: Same as points, but might be shorter.
// diff_results: Contains results.
// If which is n by k, points is m by d,
// pointsa is n by d,
// diff_results is n by (k - s) by d,
// Θ(1) depth, Θ(n(k-s)d) work,
// compute_diffs_squared(d, k, m, s, which, points, diff_results)(n, k-s, d);
__kernel void compute_diffs_squared(const size_t dims,
				    const size_t count,
				    const size_t npts,
				    const size_t skip,
				    __global const size_t *which,
				    __global const ftype *pointsa,
				    __global const ftype *points,
				    __global ftype *diff_results) {
  size_t x = get_global_id(0), y = get_global_id(1), z = get_global_id(2);
  int c = npts > which[x * count + y + skip];
  c &= (pointsa != points) | (which[x * count + y + skip] != x);
  // so as to put self at end.
  ftype d = pointsa[x * dims + z] -
    points[which[x * count + y + skip] * dims * c + z];
  diff_results[(x * (count - skip) + y) * dims + z] = d * d + (1.0 / c - 1);
  // if out of range, infinite.
}

// Add columns:
//   for(h = d; h >> 1; h >>= 1)
//     add_cols_step(d, h, k, mat)(n, k - s, h / 2);

// If mat is n by k by d,
// Θ(1) depth, Θ(nkd) work,
// add_cols_step(d, s, k, mat)(n, k, s / 2);
__kernel void add_cols_step(const size_t height,
			    const size_t s,
			    const size_t k,
			    __global ftype *mat) {
  size_t x = get_global_id(0), y = get_global_id(1), z = get_global_id(2);
  ftype g = s & !z? mat[(x * k + y) * height + s - 1] : 0;
  mat[(x * k + y) * height + z] += mat[(x * k + y) * height + z + s/2] + g;
}

// Sorting:
// nth = 1 << ceil(lg(k) - 4);
// if(nth <= max_conc_threads)
//   sort_two(k, n, along, order)(n, nth)(1);
// else
//   for(s = 0; (k - 1) >> s; s++)
//     for(ss = s; ss >= 0; ss--)
//       sort_two_step(k, n, s, ss, along, order)(n, nth);

// If order and along are n by k,
// Θ(1) depth, Θ(nk) work,
// sort_two_step(k, n, s, ss, along, order)(n, 1 << ceil(lg(k) - 4));
__kernel void sort_two_step(const size_t count,
			    const int step,
			    const int sstep,
			    __global size_t *along,
			    __global ftype *order) {
  size_t x = get_global_id(0) * count, y = get_global_id(1);
  for(int i = 0; i < 8; i++) {
    size_t y1 = y << 3 | i;
    size_t y_high = (y1 >> sstep) << sstep;
    size_t y_low = y1 ^ y_high;
    size_t y_a = y_high << 1 | y_low;
    if(sstep == step)
      y_low = (1 << sstep) - y_low - 1;
    size_t y_b = y_high << 1 | 1 << sstep | y_low;
    if(count > y_b) {
      size_t alpha = x + y_a;
      size_t beta = x + y_b;
      ftype ao = order[alpha], bo = order[beta];
      size_t aa = along[alpha], ba = along[beta];
      ulong doswap = -(ao > bo); // minimize divergence.
      alpha ^= beta, beta ^= alpha & doswap, alpha ^= beta;
      along[alpha] = aa, along[beta] = ba;
      order[alpha] = ao, order[beta] = bo;
    }
  }
}

// Removes duplicates by setting distances to +infinity.
// If along and order are n by k,
// Θ(1) depth, Θ(nk) work,
// rdups(k, along, order)(n, k - 1);
__kernel void rdups(const size_t count,
		    __global const size_t *along,
		    __global ftype *order) {
  size_t x = get_global_id(0) * count, y = get_global_id(1);
  order[x + y] += 1.0 / (along[x + y] != along[x + y + 1]) - 1;
}

// Note: long is the same length as ftype, we're doing bit tricks here!
// If points is n by d, results is length n,
// Θ(d) depth, Θ(nd) work,
// compute_signs(d, points, results)(n);
__kernel void compute_signs(const size_t d,
			    __global const ftype *points,
			    __global size_t *results) {
  size_t x = get_global_id(0);
  size_t r = 0;
  for(size_t i = 0; i < d; i++)
    r = r << 1 | (as_i_ftype(points[x * d + i]) >> (sizeof(ftype) * 8 - 1));
  results[x] = r;
}

// If wi_rev is length n,
// which_in is 2 ^ d by m,
// which is n by m * (d + 1),
// Θ(1) depth, Θ(nmd) work,
// compute_which(d, m, wi_rev, which_in, which)(n, d + 1, m);
__kernel void compute_which(const size_t d,
			    const size_t max,
			    __global const size_t *wi_rev,
			    __global const size_t *which_in,
			    __global size_t *which) {
  size_t x = get_global_id(0), y = get_global_id(1), z = get_global_id(2);
  size_t p = which_in[(wi_rev[x] ^ (!!y << (y - 1))) * max + z];
  which[(x * (d + 1) + y) * max + z] = p;
}

// If neighborsa is m by la,
// neighbors is n by l, after is m by k * (k + 1),
// Θ(1) depth, Θ(mk^2) work,
// supercharge(n, l, k, neighborsa, neighbors, after)(m, k, k);
__kernel void supercharge(const size_t n,
			  const size_t la,
			  const size_t l,
			  const size_t k,
			  __global const size_t *neighborsa,
			  __global const size_t *neighbors,
			  __global size_t *after) {
  size_t x = get_global_id(0), y = get_global_id(1), z = get_global_id(2);
  size_t w = -(size_t)(n > neighborsa[x * la + y]);
  after[(x * (k + 1) + y + 1) * k + z]
    = neighbors[(w & neighborsa[x * la + y]) * l + z] | (~w & n);
}

// If v is m by d, p is n by d, o is m by n by d,
// Θ(1) depth, Θ(mnd) work,
// prods(d, n, v, p, o)(m, n, d);
__kernel void prods(const size_t d,
		    const size_t n,
		    __global const ftype *v,
		    __global const ftype *p,
		    __global ftype *o) {
  size_t x = get_global_id(0), y = get_global_id(1), z = get_global_id(2);
  o[(x * n + y) * d + z] = v[x * d + z] * p[y * d + z];
}
