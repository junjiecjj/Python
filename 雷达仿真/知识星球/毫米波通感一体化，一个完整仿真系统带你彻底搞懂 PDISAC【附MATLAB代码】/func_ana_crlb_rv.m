function [crlb_r, crlb_v, F_tars, F_joint] = func_ana_crlb_rv(...
    N_tars, Tars_position, Tars_rcs, Tars_vel, Tx_position, N_tx_ant, Rx_position, N_rx_ant, P_tx, G_tx, G_rx, Lambda, fc, Noise_power_sen, T_chip, N_chip, N_block ...
)
    % F_tars  : 1xN_tars cell array, F_tars{m} is the 2x2 per-target FIM
    %           [F_rr F_rv; F_rv F_vv] for target m, in [r_m, v_m] order.
    % F_joint : 2*N_tars x 2*N_tars block-diagonal joint FIM
    %           F_joint = blkdiag(F_tars{1}, F_tars{2}, ..., F_tars{N_tars}).
    c = physconst('LightSpeed');
    % Convert to spherical coordinates angle from Tx to targets
    Rel_positions = Tars_position - Tx_position;
    [Azi_Tx_tars, Ele_Tx_tars, Range_tars] = cart2sph(Rel_positions(1,:), Rel_positions(2,:), Rel_positions(3,:));
    Azi_Tx_tars = rad2deg(Azi_Tx_tars);
    Ele_Tx_tars = rad2deg(Ele_Tx_tars);
    % Convert to spherical coordinates angle from targets to Rx (echo back)
    Azi_tars_Rx = Azi_Tx_tars;
    Ele_tars_Rx = Ele_Tx_tars;
    Tx_tars_array = cell2mat(arrayfun(@(i) func_get_array_response(N_tx_ant, Lambda/2, Azi_Tx_tars(i), 90-Ele_Tx_tars(i), Lambda, Tx_position), 1:N_tars, 'UniformOutput', false));
    Tars_Rx_array = cell2mat(arrayfun(@(i) func_get_array_response(N_rx_ant, N_tx_ant * Lambda/2, Azi_tars_Rx(i), 90-Ele_tars_Rx(i), Lambda, Rx_position), 1:N_tars, 'UniformOutput', false));
    Radial_tars_vel = zeros(1, N_tars);
    for i = 1:N_tars
        R_hat = Rel_positions(:,i) / norm(Rel_positions(:,i));
        Radial_tars_vel(i) = dot(Tars_vel(:,i), R_hat);
    end
    P_tx_ant   = (P_tx / N_tx_ant) * ones(N_tx_ant,1);
    G_tx_ant   = G_tx * ones(N_tx_ant,1);
    G_rx_ant   = G_rx * ones(N_rx_ant,1);
    A_e_rx_ant = G_rx_ant * Lambda^2 / (4 * pi);
    PL_moving_tars = 1 ./ ((4 * pi * Range_tars.^2) .* (4 * pi * Range_tars.^2));
    N_frame = N_chip * N_block;

    % Closed-form sums over t_i = i*T_chip, i = 0,...,N_frame-1
    sum_ti  = T_chip   * N_frame * (N_frame - 1) / 2;
    sum_ti2 = T_chip^2 * N_frame * (N_frame - 1) * (2*N_frame - 1) / 6;

    % CRLB of each target
    crlb_r = zeros(1, N_tars);
    crlb_v = zeros(1, N_tars);
    F_tars = cell(1, N_tars);
    for i = 1:N_tars
        alpha_m  = sqrt(Tars_rcs(i)) * ((Tars_Rx_array(:,i) .* sqrt(A_e_rx_ant)) * (Tx_tars_array(:,i) .* sqrt(P_tx_ant) .* sqrt(G_tx_ant)).');
        PL_m     = PL_moving_tars(i);
        r_m      = Range_tars(i);

        % FIM elements
        alpha_power = sum(abs(alpha_m(:)).^2);
        F_rr_m = 2 * alpha_power * N_frame * PL_m / Noise_power_sen * (4/r_m^2 + 16*pi^2*fc^2/c^2);
        F_rv_m = 32 * pi^2 * fc^2 * alpha_power * PL_m / (Noise_power_sen * c^2) * sum_ti;
        F_vv_m = 32 * pi^2 * fc^2 * alpha_power * PL_m / (Noise_power_sen * c^2) * sum_ti2;

        % Per-target 2x2 FIM, [r_m, v_m] order
        F_tars{i} = [F_rr_m, F_rv_m; F_rv_m, F_vv_m];

        % 2x2 FIM determinant
        det_F = F_rr_m * F_vv_m - F_rv_m^2;

        % CRLB = diagonal of FIM^{-1}
        crlb_r(i) = F_vv_m / det_F;   % Var(r_hat) >= F_vv / det(F)  [m^2]
        crlb_v(i) = F_rr_m / det_F;   % Var(v_hat) >= F_rr / det(F)  [(m/s)^2]
    end

    % Overall joint FIM: 2*N_tars x 2*N_tars block-diagonal matrix
    F_joint = blkdiag(F_tars{:});
end
