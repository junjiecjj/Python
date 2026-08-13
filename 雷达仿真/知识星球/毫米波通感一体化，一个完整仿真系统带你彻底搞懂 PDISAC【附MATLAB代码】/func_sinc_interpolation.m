function [det_range_interp, det_vel_interp] = func_sinc_interpolation( ...
        RD_map_complex, detected_positions, range_axis, vel_axis, N_sinc, N_fine)
%FUNC_SINC_INTERPOLATION  Sub-bin refinement via Whittaker-Shannon sinc.
%
% Inputs
% ──────
%  RD_map_complex      (N_range x N_doppler)  COMPLEX RD map (not magnitude)
%  detected_positions  (K x 2)                [range_bin, doppler_bin]
%  range_axis          (1 x N_range)           physical range   [m]
%  vel_axis            (1 x N_doppler)         physical velocity [m/s]
%  N_sinc              half-width of sinc window in bins (default 16)
%  N_fine              sub-grid points per half-bin search interval (default 128)

    if nargin < 5 || isempty(N_sinc), N_sinc = 16;  end
    if nargin < 6 || isempty(N_fine), N_fine = 128; end

    [N_range, N_doppler] = size(RD_map_complex);
    K = size(detected_positions, 1);

    det_range_interp = zeros(K, 1);
    det_vel_interp   = zeros(K, 1);

    for k = 1 : K
        r = detected_positions(k, 1);
        d = detected_positions(k, 2);

        %% Range dimension ────────────────────────────────────────────────
        r_lo  = max(1,       r - N_sinc);
        r_hi  = min(N_range, r + N_sinc);
        r_win = r_lo : r_hi;

        % Use complex samples, then take magnitude after reconstruction.
        Z_range = RD_map_complex(r_win, d);         % complex (W x 1)

        % Search the local sub-bin interval around the detected peak.
        f_range    = linspace(r - 0.5, r + 0.5, 2*N_fine + 1);

        % sinc matrix: W x (2*N_fine+1), sinc is even so sign does not matter
        sinc_mat_r = sinc(r_win(:) - f_range);

        % Reconstruct complex signal at fine grid, then take magnitude
        X_fine_r   = abs(Z_range.' * sinc_mat_r);

        [~, idx_max_r] = max(X_fine_r);
        r_refined      = f_range(idx_max_r);

        det_range_interp(k) = interp1(1:N_range, range_axis, ...
                                      r_refined, 'linear', 'extrap');

        %% Doppler dimension ──────────────────────────────────────────────
        d_lo  = max(1,         d - N_sinc);
        d_hi  = min(N_doppler, d + N_sinc);
        d_win = d_lo : d_hi;

        Z_dop      = RD_map_complex(r, d_win);      % complex (1 x W)

        f_dop      = linspace(d - 0.5, d + 0.5, 2*N_fine + 1);

        sinc_mat_d = sinc(d_win(:) - f_dop);        % (W x N_fine)

        X_fine_d   = abs(Z_dop * sinc_mat_d);

        [~, idx_max_d] = max(X_fine_d);
        d_refined      = f_dop(idx_max_d);

        det_vel_interp(k) = interp1(1:N_doppler, vel_axis, ...
                                    d_refined, 'linear', 'extrap');
    end
end
