function Gt = proj_tangent_sphere(T, G)
% PROJ_TANGENT_SPHERE  Project a matrix onto the tangent space of the hypersphere.
%
%   Gt = proj_tangent_sphere(T, G) returns the projection of the
%   Euclidean gradient G onto the tangent space of the complex
%   hypersphere manifold at point T.  The hypersphere is defined by
%   the constraint ||T||_F^2 = constant.  The tangent space at T
%   consists of all matrices F satisfying Re(tr(T^H F)) = 0.  The
%   projection is given by G - Re(tr(T^H G)) * T.

alpha = real_trace(T, G);
Gt    = G - alpha * T;
end