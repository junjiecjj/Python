function [mse,ber1,numBitErrs] = OAMP_det_CD_M(H, y, v_n, x, xx, it,A,cbsInfo,InfoLen,CodeRate)
    [~, Dia, V] = svd(H);
    dia = diag(Dia);
    mse_oamp = zeros(1, it);
    Var = zeros(1, it);
    M_y = length(y);
    N_x = length(x);
    u_nle = zeros(N_x, 1);                    % E(x) = 0
    v_nle = 1;                              % Var(x) = 1
    thres_0 = 1e-9;
    AHy = H' * y;
    %
    lambda = [dia.^2; zeros(M_y-N_x, 1)];
    lambda_s = 0.5 * (max(lambda) + min(lambda));
    B = lambda_s - lambda;
    w_0 = 1 / N_x * (lambda_s * M_y - sum(B));
    
    % iterations
    for t = 1 : it
        % LE
        [u_le_p, v_le_p] = LE_OAMP(u_nle, v_nle, V, AHy, dia, v_n, M_y, N_x);
        [Au_le, v_le] = Orth(u_le_p, v_le_p, u_nle, v_nle);
        % NLE
     
     u_le= A*Au_le;
    
        %u_le=A*Au_le;

        [u_nle_p, v_nle_p] = Demodulation(u_le, v_le, N_x); 
        mse_oamp(t) = (u_nle_p - x)' * (u_nle_p - x) / N_x; 
        % y1 = QPSK_to_bits(u_nle_p,length(x));
        % ber = sum(xx~=y1.');% MSE
        Var(t) = v_nle_p;
        if mse_oamp(t) <= thres_0
            tmp = (u_nle_p - x)' * (u_nle_p - x) / N_x;
            mse_oamp(t:end) = max(tmp, thres_0);
            Var(t:end) = thres_0;
            break
        elseif t == it
            break
        end
        %Au_nle_p=A'*u_nle_p;
     Au_nle_p= A'*u_nle_p;
        [u_nle, ~] = Orth(Au_nle_p, v_nle_p, Au_le, v_le);
        z = y - H * u_nle;
        v_nle = (z' * z / N_x - M_y / N_x * v_n) / w_0;
    end
    mse=mse_oamp(it);
    rxLLR1 = nrSymbolDemodulate(u_le,'QPSK',v_le,'DecisionType','soft');
    maxNumIter = 30;
    rateRecover = nrRateRecoverLDPC(rxLLR1,InfoLen,CodeRate,0,'QPSK',1);
    decBits = nrLDPCDecode(rateRecover,cbsInfo.BGN,maxNumIter);
    % Code block desegmentation and CRC decoding
    [blk,~] = nrCodeBlockDesegmentLDPC(decBits,cbsInfo.BGN,InfoLen+cbsInfo.L);
    % Transport block CRC decoding
    [out,~] = nrCRCDecode(blk,cbsInfo.CRC);
    numBitErrs = biterr(out,xx);
    ber1=numBitErrs/length(xx);
    disp(['Ber: ' num2str(ber1)])
end

%% Orthogonalization
function [u_orth, v_orth] = Orth(u_post, v_post, u_pri, v_pri)
    v_orth = 1 / (1 / v_post - 1 / v_pri);
    u_orth = v_orth * (u_post / v_post - u_pri / v_pri);  
end