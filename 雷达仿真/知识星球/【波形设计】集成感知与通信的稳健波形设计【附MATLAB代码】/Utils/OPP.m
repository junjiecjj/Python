%{
    Online supplementary materials of the paper titled:
    Robust Waveform Design for Integrated Sensing and Communication

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     1 September 2023, 21 March 2024
    @Home:     https://github.com/Spratm-Asleaf/Robust-Waveform
%}

function [Omega, Val] = OPP(A, B)
% Solution to the Standard Orthogonal Procrustes Problem (OPP)
%       min   || Omega * A - B ||_F 
%       s.t.     Omega*Omega^H = I
%
% See: 
%       https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
%       https://simonensemble.github.io/posts/2018-10-27-orthogonal-procrustes/

    [U, Sigma, V] = svd(B*A');
    [N, L] = size(Sigma);

    % return optimal solution
    if N >= L
        Omega = U*[eye(L, L); zeros(N-L, L)]*V';
    else
        Omega = U*[eye(N, N), zeros(N, L-N)]*V';
    end

    % return optimal objective
    Val = norm(Omega*A - B,'fro');
end