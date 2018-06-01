# approximateNN
Approximate nearest neighbors implementation, in C/OpenCL.
compute.cl contains the OpenCL code to handle the easily parallelizable part.

So far, this is code to apply arbitrary rotations and permutations
(allowing permutations to be submatrices of a permutation matrix),
Walsh transforms,
and inversion of permutations (if they aren't shortening ones),__
to compute the rowwise mean of a matrix and subtract it off every column
(broken into three parts),  
to compute the distances from a point to a selection of other points
(broken into two parts),  
to sort lists of points by their distances and deduplicate,  
and to copy the top `k` rows of a matrix
into the rows starting at `l` of another,
for both integer and double matrices.
