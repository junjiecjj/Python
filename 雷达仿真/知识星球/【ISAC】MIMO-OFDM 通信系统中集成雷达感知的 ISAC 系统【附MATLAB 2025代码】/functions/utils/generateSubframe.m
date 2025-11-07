function [subframeBin, subframe] = generateSubframe(onlyData, subframeMod, subframeInputSize, systemParams, ofdm, precoding, antennas)
    % Generate binary payload for subframe and modulate data using QAM
    subframeBin = randi([0,1], [subframeInputSize(1) * systemParams.bitsPerSymbol subframeInputSize(2) ofdm.numDataStreams]);
    subframeQam = qammod(subframeBin, ofdm.modOrder, 'InputType', 'bit', 'UnitAveragePower', true);

    if onlyData
        % Precode data subcarriers for subframe A
        subframeQamPre = zeros(size(subframeQam, 1), ofdm.subframeALength, antennas.Ntx);
        for nsc = 1:ofdm.numActiveSubcarriers
            subframeQamPre(nsc, :, :) = squeeze(subframeQam(nsc, :, :))*squeeze(precoding.Wp(nsc, 1:ofdm.numDataStreams,:));
        end
        % Generate OFDM symbols for subframe A
        subframe = subframeMod(subframeQamPre);
    else
        % Precode data subcarriers for subframe B
        subframeQamPre = zeros(size(subframeQam, 1), antennas.Ntx, antennas.Ntx);
        for nsc = 1:numel(ofdm.subframeBdataSubcarrierIdxs)
            idx = ofdm.subframeBdataSubcarrierIdxs(nsc) - ofdm.numGuardBandCarriers(1);
            subframeQamPre(nsc, :, :) = squeeze(subframeQam(nsc, :, :))*squeeze(precoding.Wp(idx, 1:ofdm.numDataStreams,:));
        end
        % Generate OFDM symbols for subframe B
        subframe = subframeMod(subframeQamPre, ofdm.pilots);
    end
end