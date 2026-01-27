function val = real_trace(A, B)
% REAL_TRACE  Compute the real part of the inner product of matrices A and B.
%
%   val = real_trace(A, B) returns real(trace(A^H * B)), which is the
%   Euclidean inner product between complex matrices A and B.  This
%   function is used in the Riemannian conjugate gradient algorithm
%   to compute directional derivatives and inner products on the
%   hypersphere manifold.

val = real(trace(A' * B));
end