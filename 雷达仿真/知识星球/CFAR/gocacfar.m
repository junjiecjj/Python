function [position, threshold, start_cell, stop_cell] = gocacfar(signal, Pfa, ref_num, guard_num)
    % ======>INPUT:
    % signal: Data of signal(include signal and noise).[DATATYPE: row vector]
    % Pfa: Probability of false alarm.[DATATYPE: scalar]
    % ref_num: Number of reference cell.[DATATYPE: scalar]
    % guard_num: Number of guard cell.[DATATYPE: scalar]
    % ======>OUTPUT:
    % position: positions of target.[DATATYPE: row vector]
    % threshold: CFAR threshold of input signal.[DATATYPE: row vector]
    position = [];
    left_num = guard_num + ref_num;
    start_cell = left_num + 1;
    stop_cell = length(signal) - left_num;
    N = 2 * ref_num;
    alpha = get_alpha(Pfa, N);

    threshold = zeros(1, stop_cell - start_cell + 1);
    for ii = start_cell : stop_cell
        tmp_left_data = signal(ii-left_num : ii-guard_num-1);
        tmp_right_data = signal(ii+guard_num+1 : ii+left_num);

        tmp_data = max(mean(tmp_left_data), mean(tmp_right_data));
        tmp = tmp_data * alpha;
        threshold(ii - left_num) = tmp;
        if tmp < signal(ii)
            position = [position, ii];
        end
    end

    function alpha = get_alpha(Pfa_set, N)
        % 用中点法求alpha
        left_alpha = 0;
        while true
            right_alpha = left_alpha + 1;
            this_pfa = compute_pfa(right_alpha, N);
            if this_pfa < Pfa_set
                break;
            end
            left_alpha = right_alpha;
        end
        
        mid_alpha = 0.5 * (left_alpha + right_alpha);
        this_pfa = compute_pfa(mid_alpha, N);
        while abs(this_pfa - Pfa_set) > 0.000001 * Pfa_set
            if this_pfa > Pfa_set
                left_alpha = mid_alpha;
            else
                right_alpha = mid_alpha;
            end
            mid_alpha = 0.5 * (left_alpha + right_alpha);
            this_pfa = compute_pfa(mid_alpha, N);
        end
        % this_pfa
        alpha = mid_alpha;
    return

    function pfa = compute_pfa(alpha, N)
        % GOCA CFAR
        part1 = (1 + alpha / (N / 2)) ^ (-N / 2);
        part2 = (2 + alpha / (N / 2)) ^ (-N / 2);
        part3 = 0;
        for k = 0:(N/2 - 1)
            part3 = part3 + nchoosek(N/2 - 1 + k, k) * (2 + alpha / (N/2)) ^ (-k);
        end
        pfa = part1 - part2 * part3;
        pfa = 2 * pfa;
    return
return