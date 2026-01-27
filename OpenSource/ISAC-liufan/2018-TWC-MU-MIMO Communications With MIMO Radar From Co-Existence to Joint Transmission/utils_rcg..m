function out = utils_rcg()
% utils_rcg.m
% Dummy main function, not used. Just to allow subfunctions below.
out = [];
end

%% --------------------------------------------------------------
%% real_trace: real part of <A,B>
function val = real_trace(A,B)
    val = real(trace(A'*B));
end

%% --------------------------------------------------------------
%% proj_tangent_sphere: projection onto tangent of hypersphere
%  gradR = grad - real(tr(T^H grad))*T
function Gt = proj_tangent_sphere(T, G)
    alpha = real_trace(T, G);
    Gt = G - alpha*T;
end

%% --------------------------------------------------------------
%% retraction_sphere: normalization step (式(48))
function T_new = retraction_sphere(T, dir, step, P0)
    Y = T + step*dir;
    T_new = sqrt(P0) * Y / norm(Y,'fro');
end

%% --------------------------------------------------------------
%% compute_user_SINR
% SINR_i = |h_i^T t_i|^2 / ( sum(|h_i^T t_k|^2) - |h_i^T t_i|^2 + N0 )
function SINR_i = compute_user_SINR(H, T, N0)
    [N,K] = size(H);
    SINR_i = zeros(K,1);
    for i = 1:K
        hi  = H(:,i);
        hiT = hi.'*T;  % 1 x K
        sig = abs(hiT(i))^2;
        int = sum(abs(hiT).^2) - sig;
        SINR_i(i) = sig/(int + N0);
    end
end

%% --------------------------------------------------------------
%% compute_PSLR
% PSLR = max(main_lobe)/max(side_lobe), in dB
function [PSLR_dB, P_theta] = compute_PSLR(T, theta_grid, steer, theta0, theta1, theta2)
    C = T*T';
    nTheta = numel(theta_grid);
    P_theta = zeros(nTheta,1);
    for m = 1:nTheta
        a = steer(theta_grid(m)).';
        P_theta(m) = real(a' * C * a);
    end

    main_idx = find(theta_grid >= theta1 & theta_grid <= theta2);
    side_idx = setdiff(1:nTheta, main_idx);

    P_main = max(P_theta(main_idx));
    P_side = max(P_theta(side_idx));

    PSLR_dB = 10*log10(P_main / P_side);
end
