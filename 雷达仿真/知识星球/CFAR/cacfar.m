function [position, threshold, start_cell, stop_cell] = cacfar(signal, Pfa, ref_num, guard_num)
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
    N = 2*ref_num;
    alpha = N * (Pfa ^ (-1/N) - 1);

    threshold = zeros(1, stop_cell - start_cell + 1);
    for ii = start_cell : stop_cell
        tmp_data = [signal(ii-left_num : ii-guard_num-1), ...
                    signal(ii+guard_num+1 : ii+left_num)];
        tmp = mean(tmp_data) * alpha;
        % threshold(ii - left_num)
        % tmp
        threshold(ii - left_num) = tmp;
        if tmp < signal(ii)
            position = [position, ii];
        end
    end
return