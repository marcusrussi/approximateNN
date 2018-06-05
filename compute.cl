// When I write func(params)(x, y = 1, z = 1),
// I mean, call func(params), and have one thread for every
// i, j, k that is possible where 0 <= i < x, 0 <= j < y, 0 <= k < z.

// a is supposed to have a length of l,
// r of l/2.
// End result is that row sums of a are stored in r.
// If a is n by d, r is (n/2 by d),
// Θ(lg n) depth, Θ(d lg n) work,
// add_rows(d, n, a, r)(n/2, d);
__kernel void add_rows(const size_t height, const size_t len,
		    __global const double *a,
		    __global double *r) {
  size_t x = get_global_id(0), y = get_global_id(1);
  r[x * height + y] = a[x * height + y] + a[(x + len / 2) * height + y] +
    as_double(-(ulong)(!x & len & 1) &
	      as_ulong(a[(x + len / 2 + 1) * height + y]));
  for(size_t s = len >> 2, os = (len>>1)&1; s > 0; os = s & 1, s >>= 1) {
    barrier(CLK_GLOBAL_MEM_FENCE);
    r[x * height + y] +=
      as_double(-(ulong)(x < s) & as_ulong(r[(x + s) * height + y])) +
      as_double(-(ulong)(!x & os) & as_ulong(r[(x + s + 1) * height + y]));
  }
}

// If r is length d,
// Θ(1) depth, Θ(d) work,
// divide_by_length(n, r)(d);
__kernel void divide_by_length(const size_t length, __global double *r) {
  size_t x = get_global_id(0);
  r[x] /= length;
}

// If r is length d and m is n by d,
// Θ(1) depth, Θ(nd) work,
// subtract_off(d, m, r)(n, d);
__kernel void subtract_off(const size_t height,
			   __global double *a,
			   __global const double *r) {
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
			     __global const double *ang,
			     __global double *a) {
  size_t x = get_global_id(0) * height, y = get_global_id(1);
  size_t k = i[y], l = j[y];
  double c;
  double s = sincos(ang[y], &c);
  double q = a[x + k] * c - a[x + l] * s;
  double r = a[x + k] * s + a[x + l] * c;
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
				__global const double *a,
				__global double *r) {
  size_t x = get_global_id(0), y = get_global_id(1);
  ulong p = -(ulong)(perm[y] < height_pre);
  r[x * height_post + y] =
    as_double(p & as_ulong(a[x * height_pre + perm[y]]));
}

// Undoes apply_permutation.
__kernel void apply_perm_inv(const size_t height_pre,
			     const size_t height_post,
			     const size_t garbage,
			     __global const size_t *perm,
			     __global const double *a,
			     __global double *r) {
  size_t x = get_global_id(0), y = get_global_id(1);
  ulong p = -(ulong)(perm[y] < height_post);
  size_t z = (x * height_post + perm[y]) & p;
  z |= ~p & garbage;
  r[z] = a[x * height_pre + y];
}

// If perm is size e, inv_p is size d + 1,
// Θ(1) depth, Θ(e) work,
// invert_perm(d, perm, inv_p)(e);
__kernel void invert_perm(const size_t length,
			  __global const size_t *perm,
			  __global size_t *inv_p) {
  size_t x = get_global_id(0);
  size_t t = perm[y];
  size_t p = -(size_t)(t < length);
  t = p & t | ~p & length;
  inv_p[t] = x;
}

// a is source and target, b is workspace..
// Self-inverse.
// If a and b are size n by 1 << l,
// Θ(l) depth, Θ(nl 2^l) work,
// apply_walsh(l, a, b)(n, 1 << (max(l - 3, 0)));
// Note that we skip
__kernel void apply_walsh(const size_t lheight,
			  __global double *a,
			  __global double *b) {
  size_t x = get_global_id(0), y = get_global_id(1);
  double rsr = rsqrt(2.0);
  switch(lheight) {
  case 0:
    return;
  case 1:
    double a1 = a[x << 1] + a[x << 1 | 1];
    double a2 = a[x << 1] - a[x << 1 | 1];
    a[x << 1] = a1 * rsr;
    a[x << 1 | 1] = a2 * rsr;
    return;
  case 2:
    for(int i = 0; i < 4; i++) {
      b[x << 2 | i] = 0;
      for(int j = 0; j < 4; j++) {
	int k = i & j;
	b[x << 2 | i] += a[x << 2 | j] * (1 - ((k ^ (k >> 1)) & 1) * 2);
      }
    }
    for(int i = 0; i < 4; i++)
      a[x << 2 | i] = b[x << 2 | i] / 2;
    return;
  default:
  }
  for(size_t step = 0, s = 1; step < lheight; step++, s <<= 1) {
    for(int j = 0; j < 8; j++) {
      size_t y1 = y << 3 | j;
      int sgn = 1 - (y1 >> (step - 1) & 2)
	if(step & 1)
	  a[(x << lheight) + y1] = (b[(x << lheight) + (y1 & ~s)]
				   + sgn * b[(x << lheight) + (y1 | s)]) / 2;
	else
	  b[(x << lheight) + y1] = a[(x << lheight) + (y1 & ~s)]
	    + sgn * a[(x << lheight) + (y1 | s)];
    }
      barrier(CLK_GLOBAL_MEM_FENCE);
  }
  if(~lheight & 1)
    for(int j = 0; j < 8; j++) {
      size_t y1 = y << 3 | j;
      b[(x << lheight) + y1] = a[(x << lheight) + y1] * rsr;
    }
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
				    __global const double *pointsa
				    __global const double *points,
				    __global double *diff_results) {
  size_t x = get_global_id(0), y = get_global_id(1), z = get_global_id(2);
  int c;
  z *= c = npts > which[x * count + y + skip];
  double d = pointsa[x * dims + z] -
    points[which[x * count + y + skip] * dims + z];
  diff_results[(x * (count - skip) + y) * dims + z] = d * d + (1.0 / c - 1);
  // if out of range, infinite.
}

// If mat is n by (k - s) by d, out is n by k,
// Θ(lg d) depth, Θ(n(k-s) lg d) work,
// add_cols(d, k, s, mat, out)(n, k - s, d / 2)
__kernel void add_cols(const size_t height,
		       const size_t k,
		       const size_t skip,
		       __global double *mat,
		       __global double *out) {
  size_t j = k - skip;
  size_t x = get_global_id(0), y = get_global_id(1), z = get_global_id(2);
  mat[(x * j + y) * height + z] += mat[(x * j + y) * height + z + height / 2] +
    as_double(-(!z & height & 1) &
	      as_ulong(mat[(x * j + y) * height + z + height / 2 + 1]));
  for(size_t s = height >> 2, os = (height>>1)&1; s > 0; os = s & 1, s >>= 1) {
    barrier(CLK_GLOBAL_MEM_FENCE);
    mat[(x * j + y) * height + z] +=
      as_double((z < s) & as_ulong(mat[(x * j + y) * height + z + s])) +
      as_double(-(!y & os) & as_ulong(mat[(x*j + y)*height + z + s * 2 + 1]));
  }
  if(!z)
    out[x * k + y + skip] = mat[(x * j + y) * height];
}

// If order and along are n by k,
// Θ((lg k)^2) depth, Θ(nk (lg k)^2) work,
// sort_two(k, n, along, order)(n, 1 << max(ceil(lg(k) - 4), 0));
__kernel void sort_two(const size_t count,
		       const size_t n,
		       __global size_t *along,
		       __global double *order) {
  size_t x = get_global_id(0) * count, y = get_global_id(1);
  for(int step = 0; (count - 1) >> step; step++)
    for(int sstep = step; sstep >= 0; sstep--)  {
      for(int i = 0; i < 8; i++) {
	size_t y1 = y << 3 | i;
	size_t y_high = (y1 >> sstep) << sstep;
	size_t y_low = y1 ^ y_high;
	size_t y_a = (y_high << 1) | y_low;
	if(sstep == step)
	  y_low = (1 << sstep) - y_low;
	size_t y_b = y_high << 1 | 1 << sstep | y_low;
	ulong doswap = -(order[x + y_a] > order[x + y_b]);
	ulong trash = -(count > y_a);
	size_t alpha = trash & (x + y_a) | ~trash & (n * count);
	size_t beta = trash & (x + y_b) | ~trash & (n * count);
	ulong a = as_ulong(order[x + y_a]), b = as_ulong(order[x + y_b]);
	a ^= b, b ^= a & doswap, a ^= b;
	order[alpha] = as_double(a), order[beta] = as_double(b);
	a = along[x + y_a], b = along[x + y_b];
	a ^= b, b ^= a & doswap, a ^= b;
	along[alpha] = a, along[beta] = b;
      }
      barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

// Removes duplicates by setting distances to +infinity.
// If along and order are n by k,
// Θ(1) depth, Θ(nk) work,
// rdups(k, along, order)(n, k - 1);
__kernel void rdups(const size_t count,
		    __global const size_t *along,
		    __global double *order) {
  size_t x = get_global_id(0) * count, y = get_global_id(1);
  order[x + y] += 1.0 / (along[x + y] != along[x + y + 1]) - 1;
}

// Copies column heads.
// If from is n by k+d, to is n by b+k+a,
// Θ(1) depth, Θ(nk) work,
// copy_some_ints(k+d, b+k+a, b, from, to)(n, k);
__kernel void copy_some_ints(const size_t height_pre,
			     const size_t height_post,
			     const size_t start_post,
			     __global const size_t *from,
			     __global size_t *to) {
  size_t x = get_global_id(0), y = get_global_id(1);
  to[x * height_post + start_post + y] = from[x * height_pre + y];
}

// Copies column heads.
// If from is n by k+d, to is n by b+k+a,
// Θ(1) depth, Θ(nk) work,
// copy_some_floats(k+d, b+k+a, b, from, to)(n, k);
__kernel void copy_some_floats(const size_t height_pre,
			       const size_t height_post,
			       const size_t start_post,
			       __global const double *from,
			       __global double *to) {
  size_t x = get_global_id(0), y = get_global_id(1);
  to[x * height_post + start_post + y] = from[x * height_pre + y];
}

// Note: long is the same length as double, we're doing bit tricks here!
// If points is n by d, results is length n,
// Θ(d) depth, Θ(nd) work,
// compute_signs(d, points, results)(n);
__kernel void compute_signs(const size_t d,
			    __global const long *points,
			    __global size_t *results) {
  size_t x = get_global_id(0);
  size_t r = 0;
  for(size_t i = 0; i < d; i++)
    r = (r | (points[x * d + i] >> 63)) << 1;
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
  which[(x * (d + 1) + y) * max + z] =
    which_in[(wi_rev[x] ^ (!y << (y - 1))) * max + z];
}

// If neighborsa is m by l,
// neighbors is n by l, after is m by k * (k + 1),
// Θ(1) depth, Θ(mk^2) work,
// supercharge(n, l, k, neighborsa, neighbors, after)(m, k, k);
__kernel void supercharge(const size_t n,
			  const size_t l,
			  const size_t k,
			  __global const size_t *neighborsa,
			  __global const size_t *neighbors,
			  __global size_t *after) {
  size_t x = get_global_id(0), y = get_global_id(1), z = get_global_id(2);
  size_t w = -(size_t)(n > neighborsa[x * l + y]) ;
  after[(x * k + y) * k + k + z]
    = neighbors[(w & neighborsa[x * l + y]) * l + z] | (~w & n);
}

// If v is length d, p is n by d, o is n by d,
// Θ(1) depth, Θ(nd) work,
// prods(d, v, p, o)(n, d);
__kernel void prods(const size_t d,
		    __global const double *v,
		    __global const double *p,
		    __global double *o) {
  size_t x = get_global_id(0) * d, y = get_global_id(1);
  o[x + y] = v[y] * p[x + y];
}
