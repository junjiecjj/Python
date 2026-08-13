function [H_sen_tars, Range_tars, Radial_tars_vel] = h_sen_tars(N_tars, Tars_position, Tars_rcs, Tars_vel, Tx_position, N_tx_ant, Rx_position, N_rx_ant, P_tx, G_tx, G_rx, Lambda, fc, T_step)
    
    c = physconst('LightSpeed');
    % Convert to spherical coordinates angle from Tx to targets
    Rel_positions = Tars_position - Tx_position; 
    [Azi_Tx_tars, Ele_Tx_tars, Range_tars] = cart2sph(Rel_positions(1,:), Rel_positions(2,:), Rel_positions(3,:));
    Azi_Tx_tars = rad2deg(Azi_Tx_tars);
    Ele_Tx_tars = rad2deg(Ele_Tx_tars);
    
    % Convert to spherical coordinates angle from targets to Rx (echo back)
    Azi_tars_Rx = Azi_Tx_tars;
    Ele_tars_Rx = Ele_Tx_tars;

    % Compute the antenna array beetween Tx and Rx
    Tx_tars_array = cell2mat(arrayfun(@(i) func_get_array_response(N_tx_ant, Lambda/2, Azi_Tx_tars(i), 90-Ele_Tx_tars(i), Lambda, Tx_position), 1:N_tars, 'UniformOutput', false));
    Tars_Rx_array = cell2mat(arrayfun(@(i) func_get_array_response(N_rx_ant, N_tx_ant * Lambda/2, Azi_tars_Rx(i), 90-Ele_tars_Rx(i), Lambda, Rx_position), 1:N_tars, 'UniformOutput', false));
    
    % Compute radial velocities
    Radial_tars_vel = zeros(1, N_tars);
    for i = 1:N_tars
        R_hat = Rel_positions(:,i) / norm(Rel_positions(:,i));  % (3×1)
        Radial_tars_vel(i) = dot(Tars_vel(:,i), R_hat);  
    end

    P_tx_ant = (P_tx / N_tx_ant) * ones(N_tx_ant,1); % Tx power of each antenna P_tx / N_tx_ant with shape N_tx_ant x 1 
    G_tx_ant = G_tx * ones(N_tx_ant,1); % Tx gain: N_tx_ant x 1 
    G_rx_ant = G_rx * ones(N_rx_ant,1); % Rx gain: N_rx_ant x 1
    A_e_rx_ant = G_rx_ant * Lambda^2 / (4 * pi); % Effective antenna aperture 
    PL_moving_tars = 1 ./ ((4 * pi * Range_tars.^2) .* (4 * pi * Range_tars.^2)); % 1 x N_tars
    Delay_tars = 2 * Range_tars / c; % 1 x N_tars
    Doppler_tars = -2 * Radial_tars_vel / Lambda; % 1 x N_tars
    
    
    % Example sensing channel of the moving targets at time steps T_prbs
    H_sen_tars = zeros(N_rx_ant, N_tx_ant, N_tars);
    for i = 1:N_tars
        H_sen_tars(:,:,i) = sqrt(Tars_rcs(:, i)) ... % target coefficient reflection
                              * ((sqrt(PL_moving_tars(:,i)) .* Tars_Rx_array(:,i) .* sqrt(A_e_rx_ant)) ... % receiver component: antenna gain, path loss, receiver antenna array 
                              * (Tx_tars_array(:,i) .* sqrt(P_tx_ant) .* sqrt(G_tx_ant)).') ... % transmitter component: antenna power, atenna gain, transmit antenna array 
                              * exp(-1j * 2*pi * fc * Delay_tars(:,i)) ... % delay component
                              * exp( 1j * 2*pi * Doppler_tars(:,i) * T_step); % doppler component
    end
end