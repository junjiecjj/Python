function array = func_get_array_response(N_ant, d, azi, ele, lambda, position, is_bbe_pattern)
    % azi: 1 x N_azi vector (degrees)
    % ele: 1 x N_ele vector (degrees)
    % Output: N_ant x (N_azi * N_ele)

    if nargin < 7
        is_bbe_pattern = true; % default behavior
    end

    % Constants
    k = 2*pi/lambda;

    % Meshgrid over all (azi, ele) combinations
    [AZI, ELE] = meshgrid(azi, ele); % N_ele x N_azi
    
    phi   = deg2rad(AZI);  
    theta = deg2rad(ELE);

    % Unit vectors
    ux = sin(theta(:))' .* cos(phi(:))';
    uy = sin(theta(:))' .* sin(phi(:))';
    uz = cos(theta(:))';
    U  = [ux; uy; uz];   % 3 x N_points

    % Array Geometry (Y-axis ULA, centered)
    n_indices = ((0:N_ant-1)' - (N_ant-1)/2);
    pos = [zeros(N_ant,1), n_indices*d, zeros(N_ant,1)];
    pos = pos + position(:)';

    % Steering Matrix
    AF = exp(1j * k * (pos * U));   % N_ant x N_points

    % Back-Baffled Element Pattern
    if is_bbe_pattern
        % Equivalent to -90 < azi < 90
        g = double(ux > 0);   % 1 x N_points
    else
        g = ones(1, size(U,2)); % no attenuation
    end

    % Apply element pattern
    array = AF .* g;

    % Reshape
    N_azi = numel(azi);
    N_ele = numel(ele);
    array = reshape(array, N_ant, N_ele, N_azi);
    array = permute(array, [1, 3, 2]);  % N_ant x N_azi x N_ele
    array = reshape(array, size(array,1), []);
end