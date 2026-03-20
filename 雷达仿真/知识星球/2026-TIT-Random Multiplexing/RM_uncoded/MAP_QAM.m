%% 
function BER = MAP_QAM(M, rho)
    k = log2(M);
    pam_order = sqrt(M); 
    pam_levels = -(pam_order-1) : 2 : (pam_order-1);
    [i_comp, q_comp] = meshgrid(pam_levels, pam_levels);
    constellation_unnormalized = i_comp(:) + 1j * q_comp(:);
    avg_energy = mean(abs(constellation_unnormalized).^2);
    constellation = constellation_unnormalized / sqrt(avg_energy);
    %
    bits_per_pam = log2(pam_order);
    gray_map_pam = zeros(pam_order, bits_per_pam);
    for i = 0:(pam_order-1)
        gray_code = bitxor(i, floor(i/2));
        gray_map_pam(i+1, :) = de2bi(gray_code, bits_per_pam, 'left-msb');
    end
    %
    bit_map = zeros(M, k);
    idx = 1;
    %
    for i = 1:pam_order 
        for q = 1:pam_order
            % I-component
            bits_i = gray_map_pam(i,:);
            % Q-component
            bits_q = gray_map_pam(q,:);
            bit_map(idx, :) = [bits_i, bits_q];
            idx = idx + 1;
        end
    end
    sigma_comp = 1 / sqrt(2);
    total_weighted_errors = 0;
    for i = 1:M
        for j = 1:M
            if i == j
                continue;
            end
            d_ij = abs(constellation(i) - constellation(j));
            N_ij = sum(bitxor(bit_map(i,:), bit_map(j,:)));
            q_arg = (sqrt(rho) * d_ij) / (2 * sigma_comp);
            pep = qfunc(q_arg);
            total_weighted_errors = total_weighted_errors + (N_ij * pep);
        end
    end
    normalization_factor = 1 / (M * k);
    BER = normalization_factor * total_weighted_errors;
end