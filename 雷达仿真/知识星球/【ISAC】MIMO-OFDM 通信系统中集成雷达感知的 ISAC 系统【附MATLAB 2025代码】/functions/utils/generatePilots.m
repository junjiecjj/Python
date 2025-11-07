function [pilotIdxs, pilots] = generatePilots(ofdm, Ntx)
    % This function generates pilot subcarrier indices and pilot signals for the OFDM system.

    Nsub = ofdm.Nsub; % Total number of subcarriers
    numGuardBandCarriers = ofdm.numGuardBandCarriers; % Number of guard band carriers
    Mf = ofdm.Mf; % Range sampling parameter

    pilotIdxs = [(numGuardBandCarriers(1)+1):Mf:(Nsub/2) (Nsub/2+2):Mf:(Nsub-numGuardBandCarriers(2))]';
    pilots = zeros(numel(pilotIdxs), Ntx, Ntx);
    for itx = 1:Ntx
        s = mlseq(Nsub-1, itx);
        pilots(:, itx, itx) = s(1:numel(pilotIdxs));
    end
end