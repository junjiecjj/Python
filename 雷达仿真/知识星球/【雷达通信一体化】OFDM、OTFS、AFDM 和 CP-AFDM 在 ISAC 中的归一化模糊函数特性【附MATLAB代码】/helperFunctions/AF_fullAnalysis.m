function [tau_norm, nu_norm, AFdB_zerodop, AFdB_zerodel, metrics_delay, metrics_dopp] = AF_fullAnalysis(s, params, OS_del, OS_dop)
    %
    % =========================================================================
    % Author    : Dr. (Eric) Hyeon Seok Rou
    % Version   : v1.0
    % Date      : Oct 13, 2025
    % =========================================================================

    addpath('helperFunctions/');
    % Compute AF cuts
    [tau_norm, AFdB_zerodop, AF_zerodop] = AF_zerodopplercut(s, OS_del, 4);
    [nu_norm, AFdB_zerodel, AF_zerodel] = AF_zerodelaycut(s, OS_dop, 1);
    lenDopGuard = floor(length(nu_norm)/20);
    nu_norm = nu_norm(lenDopGuard+1:end-lenDopGuard);
    AFdB_zerodel = AFdB_zerodel(lenDopGuard+1:end-lenDopGuard);
    AF_zerodel = AF_zerodel(lenDopGuard+1:end-lenDopGuard);

    % Delay cut metrics
    metrics_delay = compute_delay_metrics(tau_norm, AF_zerodop, params);
    % Doppler cut metrics
    metrics_dopp  = compute_doppler_metrics(nu_norm, AF_zerodel, params);

end
