function [mse,ber1,numBitErrs] = MAMP_CD_M(H, y, v_n, x,xx, it,A,cbsInfo,InfoLen,CodeRate)
    H=sparse(H);
    L = 3;                              % length of damping 
    M_y = length(y);
    N_x = length(x);
    beta = M_y / N_x;
    L1 = 40;
    lambda_s = get_lambda_s(H, M_y, N_x, it, L1);
    L2 = 10;
    [log_w, sign_w] = get_logw(H, lambda_s, M_y, N_x, it, L2, v_n);


    %B = lamda_s * ones(M, 1) - lamda;        % eigenvalue of B


    w_0 = sign_w(1) * exp(log_w(1));
    w_1 = sign_w(2) * exp(log_w(2));
    w_bar_00 = lambda_s * w_0 - w_1 - w_0 * w_0;
    x_phi = zeros(N_x, it);
    v_phi = zeros(it, it);
    x_phi(:, 1) = zeros(N_x, 1);                      % E(x) = 0
    log_theta_ = zeros(1, it);
    theta_w_ = zeros(1, 2*it-1);
    r_hat = zeros(M_y, 1);
    z = zeros(M_y, it);
    z(:, 1) = y;                                    % z1 = y - Ax1, x1 = 0
    v_phi(1, 1) = real(1/N_x * z(:, 1)' * z(:, 1) - beta * v_n) / w_0;
    MSE = zeros(1, it);
    Var = zeros(1, it);
    thres_0 = 1e-9;
    index = [];
    MSE_mle = zeros(1, it);

    % iterations
    for t = 1 : it
        % MLE
%         [log_theta_, theta_w_, r_hat, r, v_gam] = MLE_MAMP(A, x_phi, v_phi, log_theta_, theta_w_, ...
%             z, r_hat, B, sign, log_B, w_0, w_bar_00, lambda_s, t, v_n, N);

            [log_theta_, r_hat, Ar, v_gam] = MLE_MAMP_e(x_phi, v_phi, r_hat, log_theta_, sign_w, log_w, ...
            w_0, w_bar_00, z, H, lambda_s, t, v_n, M_y, N_x);
            v_gam = real(v_gam);
       r=A*Ar;
        % NLE
        [x_hat, v_hat] =  Demodulation(r, v_gam, N_x);
        MSE(t) = (x_hat - x)' * (x_hat - x) / N_x;
        % y1 = QPSK_to_bits(x_hat,length(x_hat));
        % ber = sum(xx~=y1.');
        Var(t) = v_hat;
        if MSE(t) <= thres_0
            tmp = (x_hat - x)' * (x_hat - x) / N_x;
            MSE(t:end) = max(tmp, thres_0);
            Var(t:end) = thres_0;
            break
        elseif t == it
            break
        end
        Ax_hat=A'*x_hat;
        x_phi(:, t+1) = (Ax_hat / v_hat - Ar / v_gam) / (1 / v_hat - 1 / v_gam);
        %v_post = 1 / (1 / v_hat - 1 / v_gam);
        z(:, t+1) = y - H * x_phi(:, t+1);
        for k = 1 : t+1
            v_phi(t+1, k) = (1 / N_x * z(:, t+1)' * z(:, k) - beta * v_n) / w_0;
            v_phi(k, t+1) = v_phi(t+1, k)';
        end
        % damping
        % thres = 0.5;
        % if v_post < thres * v_phi(t+1, t+1)
            back_flag = false;
        [x_phi, z, index] = Damping_NLE(x_phi, v_phi, z, index, L, t, back_flag);
        for k = 1 : t+1
            v_phi(t+1, k) = (1 / N_x * z(:, t+1)' * z(:, k) - beta * v_n) / w_0;
            v_phi(k, t+1) = v_phi(t+1, k)';
        end
    end
    mse=MSE(it);
    rxLLR1 = nrSymbolDemodulate(r,'QPSK',v_gam,'DecisionType','soft');
    maxNumIter = 30;
    rateRecover = nrRateRecoverLDPC(rxLLR1,InfoLen,CodeRate,0,'QPSK',1);
    decBits = nrLDPCDecode(rateRecover,cbsInfo.BGN,maxNumIter);
    % Code block desegmentation
    [blk,~] = nrCodeBlockDesegmentLDPC(decBits,cbsInfo.BGN,InfoLen+cbsInfo.L);
    % CRC decoding
    [out,~] = nrCRCDecode(blk,cbsInfo.CRC);
    numBitErrs = biterr(out,xx);
    ber1=numBitErrs/length(xx);
    disp(['Ber: ' num2str(ber1)])
end

%% Damping at NLE
function [x_phi, z, index] = Damping_NLE(x_phi, v_phi, z, index, L, t, flag)
    % find out damping index
    l = min(L, t+1);
    d = 0;
    dam_ind = [];
    for k = flip(1:t+1)
        if ismember(k, index)
            continue
        else
            d = d + 1;
            dam_ind = [k, dam_ind];
            if d == l
                break
            end
        end
    end
    l = length(dam_ind);
    % delete rows and columns
    del_ind = setdiff(1:t+1, dam_ind);
    v_da = v_phi(1:t+1, 1:t+1);
    v_da(del_ind, :) = [];
    v_da(:, del_ind) = [];
    % obtain zeta
    if flag || rcond(v_da) < 1e-15 || min(eig(v_da)) < 0
        zeta = [zeros(1, l-2), 1, 0];
        index = [index, t+1];
    else
        o = ones(l, 1);
        tmp = v_da \ o;
        v_s = real(o' * tmp);
        zeta = tmp / v_s;
        zeta = zeta.';
    end
    % update
    x_phi(:, t+1) = sum(zeta.*x_phi(:, dam_ind), 2);
    z(:, t+1) = sum(zeta.*z(:, dam_ind), 2);
end