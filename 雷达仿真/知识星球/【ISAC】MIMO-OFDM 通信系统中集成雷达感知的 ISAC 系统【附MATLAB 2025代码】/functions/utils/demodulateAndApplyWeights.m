function [rxSubframeQamComb, rxPilots] = demodulateAndApplyWeights(onlyData, subframeDemod, rxSubframe, ofdm, precoding, antennas)
    rxPilots = zeros(numel(ofdm.pilotIdxs), antennas.Ntx, antennas.Ntx);

    if onlyData
            % Demodulate subframe A
            rxSubframeQam = subframeDemod(rxSubframe);
            rxSubframeQamComb = zeros(size(rxSubframeQam, 1), size(rxSubframeQam, 2), ofdm.numDataStreams);
            
            % Apply the combining weights
            for nsc = 1:ofdm.numActiveSubcarriers
                rxSubframeQamComb(nsc, :, :) = (squeeze(rxSubframeQam(nsc,:,:)) * squeeze(precoding.Wc(nsc,:,1:ofdm.numDataStreams))) ./ sqrt(precoding.G(nsc,1:ofdm.numDataStreams));
            end
        else
            % Demodulate subframe B
            [rxSubframeQam, rxPilots] = subframeDemod(rxSubframe);
            rxSubframeQamComb = zeros(size(rxSubframeQam, 1), size(rxSubframeQam, 2), ofdm.numDataStreams);

            % Apply the combining weights
            for nsc = 1:numel(ofdm.subframeBdataSubcarrierIdxs)
                idx = ofdm.subframeBdataSubcarrierIdxs(nsc) - ofdm.numGuardBandCarriers(1);
                rxSubframeQamComb(nsc, :, :) = ((squeeze(rxSubframeQam(nsc, :, :))*squeeze(precoding.Wc(idx, :, 1:ofdm.numDataStreams))))./sqrt(precoding.G(idx, 1:ofdm.numDataStreams));
            end
        end
