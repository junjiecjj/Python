function T_new = retraction_sphere(T, dir, step, P0)
% RETRACTION_SPHERE  Retraction on the hypersphere manifold.
%
%   T_new = retraction_sphere(T, dir, step, P0) computes a new point
%   on the hypersphere defined by ||T||_F^2 = P0.  Starting from T,
%   one moves in the direction "dir" scaled by the stepsize "step",
%   then renormalizes to satisfy the power constraint.  This is the
%   retraction used in the RCG algorithm.

Y = T + step * dir;
T_new = sqrt(P0) * Y / norm(Y, 'fro');
end