function [position, threshold, start_cell, stop_cell] = oscfar(signal, Pfa, ref_num, guard_num, k)
    % ======>INPUT:
    % signal: Data of signal(include signal and noise).[DATATYPE: row vector]
    % Pfa: Probability of false alarm.[DATATYPE: scalar]
    % ref_num: Number of reference cell.[DATATYPE: scalar]
    % guard_num: Number of guard cell.[DATATYPE: scalar]
    % k: # k sorted statistical value.[DATATYPE: scalar]
    % ======>OUTPUT:
    % position: positions of target.[DATATYPE: row vector]
    % threshold: CFAR threshold of input signal.[DATATYPE: row vector]
    position = [];
    left_num = guard_num + ref_num;
    start_cell = left_num + 1;
    stop_cell = length(signal) - left_num;
    N = 2*ref_num;
    alpha = get_alpha(Pfa, N, k);

    threshold = zeros(1, stop_cell - start_cell + 1);
    for ii = start_cell : stop_cell
        tmp_data = [signal(ii-left_num : ii-guard_num-1), ...
                    signal(ii+guard_num+1 : ii+left_num)];
        sorted_data = sort(tmp_data);
        T = sorted_data(k) * alpha;
        threshold(ii - left_num) = T;
        if T < signal(ii)
            position = [position, ii];
        end
    end

    function alpha = get_alpha(Pfa_set, N, k)
        % 用中点法求alpha_OS
        left_alpha = 0;
        while true
            right_alpha = left_alpha + 1;
            this_pfa = k * nchoosek(N, k) * beta(right_alpha + N - k + 1, k);
            if this_pfa < Pfa_set
                break;
            end
            left_alpha = right_alpha;
        end
        
        mid_alpha = 0.5 * (left_alpha + right_alpha);
        this_pfa = k * nchoosek(N, k) * beta(mid_alpha + N - k + 1, k);
        while abs(this_pfa - Pfa_set) > 0.000001 * Pfa_set
            if this_pfa > Pfa_set
                left_alpha = mid_alpha;
            else
                right_alpha = mid_alpha;
            end
            mid_alpha = 0.5 * (left_alpha + right_alpha);
            this_pfa = k * nchoosek(N, k) * beta(mid_alpha + N - k + 1, k);
        end
        % this_pfa
        alpha = mid_alpha;
    return
return