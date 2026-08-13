function X_t_delay = func_get_x_delay(Data_Tx, Index_block, Delay_tars, T_chip)
    % X_DELAY  Returns the delayed Tx data at a given block index.
    %
    % Inputs:
    %   Data_Tx     : (N_block, N_sample, N_ant) transmitted data
    %   Index_block : scalar, which block (1-based) to evaluate at
    %   Delay_tars  : (1, N_tars) or (N_tars, 1) integer sample delays per target
    %
    % Output:
    %   X_t_delay   : (N_sample, N_ant, N_tars) delayed signal at block Index_block
    
    [~, N_sample, N_ant] = size(Data_Tx);
    Delay_tars = Delay_tars(:);       % flatten to (N_tars, 1) regardless of input shape
    N_tars     = numel(Delay_tars);
    
    X_t_delay = zeros(N_sample, N_ant, N_tars);
    
    for j = 1:N_tars
        d = floor(Delay_tars(j) / T_chip);
    
        assert(d >= 0, 'Delay must be non-negative (causal system).');
    
        n_blocks_needed = ceil(d / N_sample) + 1;
        block_start     = max(1, Index_block - n_blocks_needed + 1);
    
        X_concat = reshape(permute(Data_Tx(block_start:Index_block,:,:), [2,1,3]), [], N_ant);
    
        tail      = size(X_concat, 1);
        idx_end   = tail - d;
        idx_start = idx_end - N_sample + 1;
    
        if idx_end <= 0
            continue;  % delay exceeds all available history — leave as zeros
        end
    
        if idx_start < 1
            n_zeros = 1 - idx_start;
            valid   = X_concat(1:idx_end, :);
            X_t_delay(:,:,j) = [zeros(n_zeros, N_ant); valid];
        else
            X_t_delay(:,:,j) = X_concat(idx_start:idx_end, :);
        end
    end
end