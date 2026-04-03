function [theta_b_deg, aT, info] = design_theta_P0(para)
% DESIGN_THETA_P0  Choose a beam direction that matches the desired illumination power.
%
% Inputs:
%   para        : Parameter struct. The key fields used here are
%                 Nt, PT, P0, theta0_deg, dT, and lambda.
%
% Outputs:
%   theta_b_deg : Selected transmit beam direction in degrees.
%   aT          : Steering vector at theta_b_deg.
%   info        : Struct containing the achieved correlation and power metrics.

theta0_deg = para.theta0_deg;
theta0_rad = deg2rad(theta0_deg);

target_corr2 = para.Nt * para.PT / para.PT;

if target_corr2 < 0 || target_corr2 > para.Nt^2
    error('Infeasible target: Nt * P0 / PT must lie in [0, Nt^2].');
end

if abs(target_corr2 - para.Nt^2) < 1e-12
    theta_b_deg = theta0_deg;
    aT = steering_vec_deg(theta_b_deg, para);

    info = struct();
    info.psi = 0;
    info.target_corr2 = target_corr2;
    info.achieved_corr2 = para.Nt^2;
    info.achieved_PIL = para.PT / para.Nt * info.achieved_corr2;
    info.achieved_ML = info.achieved_PIL^2;
    return;
end

Nt = para.Nt;
f = @(psi) dirichlet_sq(psi, Nt) - target_corr2;

psi_lo = 1e-10;
psi_hi = 2 * pi / Nt - 1e-10;

psi_star = fzero(f, [psi_lo, psi_hi]);
delta_sin = psi_star * para.lambda / (2 * pi * para.dT);

cand = [];

s1 = sin(theta0_rad) + delta_sin;
if abs(s1) <= 1
    cand(end+1) = rad2deg(asin(s1));
end

s2 = sin(theta0_rad) - delta_sin;
if abs(s2) <= 1
    cand(end+1) = rad2deg(asin(s2));
end

if isempty(cand)
    error('No feasible theta_b found from the solved psi.');
end

[~, idx] = min(abs(cand - theta0_deg));
theta_b_deg = cand(idx);

aT = steering_vec_deg(theta_b_deg, para);
a0 = steering_vec_deg(theta0_deg, para);
achieved_corr2 = abs(a0' * aT)^2;
achieved_PIL = para.PT / para.Nt * achieved_corr2;
achieved_ML = achieved_PIL^2;

info = struct();
info.psi = psi_star;
info.target_corr2 = target_corr2;
info.achieved_corr2 = achieved_corr2;
info.achieved_PIL = achieved_PIL;
info.achieved_ML = achieved_ML;
end

function val = dirichlet_sq(psi, Nt)
% DIRICHLET_SQ  Squared Dirichlet-kernel magnitude for a ULA.
if abs(psi) < 1e-12
    val = Nt^2;
else
    val = (sin(Nt * psi / 2) / sin(psi / 2))^2;
end
end

function a = steering_vec_deg(theta_deg, para)
% STEERING_VEC_DEG  ULA steering vector at angle theta_deg.
a = exp(1j * 2 * pi * (0:para.Nt-1).' * para.dT * sind(theta_deg) / para.lambda);
end