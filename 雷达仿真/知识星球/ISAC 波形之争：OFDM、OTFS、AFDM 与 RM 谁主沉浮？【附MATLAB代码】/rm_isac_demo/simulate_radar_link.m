function res = simulate_radar_link(txSamples, H_radar, snrDb, delayGrid, dopplerGrid)
%SIMULATE_RADAR_LINK 感知链路：双目标回波 + CAF

L = numel(txSamples);
noiseVar = 10^(-snrDb/10);
noise = sqrt(noiseVar/2) * (randn(L,1) + 1j*randn(L,1));

rxRadar = H_radar * txSamples + noise;
CAF = compute_caf(rxRadar, txSamples, delayGrid, dopplerGrid);

res = struct();
res.rxSamples = rxRadar;
res.CAF = CAF;
res.noiseVar = noiseVar;

end
