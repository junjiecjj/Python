%{
    Online supplementary materials of the paper titled:
    Distributionally Robust Adaptive Beamforming
    By Shixiong Wang, Wei Dai, and Geoffrey Ye Li
    From Department of Electrical and Electronic Engineering, Imperial College London

    @Author:   Shixiong Wang (s.wang@u.nus.edu; wsx.gugo@gmail.com)
    @Date:     8 Oct 2024, 13 March 2025
    @Home:     https://github.com/Spratm-Asleaf/Beamforming-UDL
%}

function A = a0(N, theta)
% Return N-dimensional column steering vectors associated with DoA's in "theta"
% If "theta" contains only one DoA, it returns only one column steering vector
% "theta" is defined from the array broadside, i.e., normal to array line

    K = length(theta);
    A = zeros(N, K);

    for k = 1:K
        if theta(k) < -pi/2
            theta(k) = -pi/2;
        elseif theta(k) > pi/2
            theta(k) = pi/2;
        end
        
        % Steering Vector
        a = zeros(N, 1);
        for n = 1:N
            a(n) = exp(-1j*pi*(n-1)*sin(theta(k)));
        end
        A(:, k) = a;
    end
end

