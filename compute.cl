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
			     const size_t *i, // first coordinate
			     const size_t *j, // second coordinate
			     const double *ang,
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

// If perm is size e, inv_p is size d,
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

// a is source, b is target.
// Self-inverse.
// If a and b are size n by 1 << l,
// Θ(l) depth, Θ(nl 2^l) work,
// apply_walsh(l, a, b)(n, 1 << l);
__kernel void apply_walsh(const size_t lheight,
			  __global double *a,
			  __global double *b) {
  size_t x = get_global_id(0), y = get_global_id(1);
  for(size_t step = 0, s = 1; step < lheight; step++, s <<= 1) {
    int sgn = 1 - (y >> (step - 1) & 2)
    if(step & 1)
      a[(x << lheight) + y] = (b[(x << lheight) + (y & ~s)]
			       + sgn * b[(x << lheight) + (y | s)]) / 2;
    else
      b[(x << lheight) + y] = a[(x << lheight) + (y & ~s)]
	+ sgn * a[(x << lheight) + (y | s)];
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
  if(lheight & 1)
    b[(x << lheight) + y] = a[(x << lheight) + y];
  else
    b[(x << lheight) + y] *= rsqrt(2.0);
}

// dims: each point is d-dimensional
// count: how many points do we compute distances to (k)
// which: From x, compute distances to which[x][y].
// points: points[x][z] contains proper coordinate.
// diff_results: Contains results.
// If which is n by k, points is n by d,
// diff_results is n by k by d,
// Θ(1) depth, Θ(nkd) work,
// compute_diffs_squared(d, k, n, which, points, diff_results)(n, k, d);
__kernel void compute_diffs_squared(const size_t dims,
				    const size_t count,
				    const size_t npts,
				    __global const size_t *which,
				    __global const double *points,
				    __global double *diff_results) {
  size_t x = get_global_id(0), y = get_global_id(1), z = get_global_id(2);
  int c;
  z *= c = npts != which[x * count + y];
  double d = points[x * dims + z] - points[which[x * count + y] * dims + z];
  diff_results[(x * count + y) * dims + z] = d * d + (1.0 / c - 1);
  // if out of range, infinite.
}

// If mat is n by k by d, out is n by k,
// Θ(lg d) depth, Θ(nk lg d) work,
// add_cols(d, mat, out)(nk, d / 2)
__kernel void add_cols(const size_t height,
		       __global double *mat,
		       __global double *out) {
  size_t x = get_global_id(0), y = get_global_id(1);
  mat[x * height + y] += mat[x * height + y + height / 2] +
    as_double(-(!y & height & 1) &
	      as_ulong(mat[x * height + y + height / 2 + 1]));
  for(size_t s = height >> 2, os = (height>>1)&1; s > 0; os = s & 1, s >>= 1) {
    barrier(CLK_GLOBAL_MEM_FENCE);
    mat[x * height + y] +=
      as_double((y < s) & as_ulong(mat[x * height + y + s])) +
      as_double(-(!y & os) & as_ulong(mat[x * height + y + s * 2 + 1]));
  }
  if(!y)
    out[x] = mat[x * height];
}

// If order and along are n by k,
// Θ((lg k)^2) depth, Θ(nk (lg k)^2) work,
// sort_two(k, along, order)(n, 1 << ceil(lg(k)));
__kernel void sort_two(const size_t count,
		       __global size_t *along,
		       __global double *order) {
  size_t x = get_global_id(0) * count, y = get_global_id(1);
  for(int step = 0; (count - 1) >> step; step++)
    for(int sstep = step; sstep >= 0; sstep--)  {
      size_t y_high = (y >> sstep) << sstep;
      size_t y_low = y ^ y_high;
      size_t y_a = (y_high << 1) | y_low;
      if(sstep == step)
	y_low = (1 << sstep) - y_low;
      size_t y_b = y_high << 1 | 1 << sstep | y_low;
      ulong doswap = -(y_b < count) * (order[x + y_a] > order[x + y_b]);
      ulong a = as_ulong(order[x + y_a]), b = as_ulong(order[x + y_b]);
      a ^= b, b ^= a & doswap, a ^= b;
      order[x + y_a] = as_double(a), order[x + y_b] = as_double(b);
      a = along[x + y_a], b = along[x + y_b];
      a ^= b, b ^= a & doswap, a ^= b;
      along[x + y_a] = a, along[x + y_b] = b;
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
// This is post-rotation.
// Is there a faster way?
// Let's hope, otherwise the total is Ω(n) depth.

// If points is n by d, placement is n by (k + 1),
// Θ(nd) depth, Θ(dn^2) work,
// select_adjacents(d, n, k, points, placement)(n)

__kernel void select_adjacents(const size_t height,
			       const size_t length,
			       const size_t allowed,
			       __global const long *points,
			       __global size_t *placement) {
  size_t x = get_global_id(0);
  size_t curlist = 0;
  for(size_t curpt = 0; curpt < length; curpt++) {
    int count = 0;
    for(size_t cury = 0; cury < height; cury++)
      count +=
	(points[x * height + cury] ^ points[curpt * height + cury]) >> 63;
    count = (count < 2) & (curpt != x) & (curlist != allowed);
    placement[x * (allowed + 1) + curlist] = curpt;
    curlist += count;
  }
}
