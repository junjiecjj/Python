function H = helperInterpolateChannelMatrix(numSubcarriers, numGuardBandCarriers, pilotsTx, pilotsRx, pilotIdxs)
    numLeftG = numGuardBandCarriers(1);
    numRightG = numGuardBandCarriers(2);
    subCarrierIdxs = ((numLeftG+1):(numSubcarriers-numRightG)).';

    numTx = size(pilotsRx, 2);
    numRx = size(pilotsRx, 3);
    H = zeros(numel(subCarrierIdxs), numTx, numRx);

    if isvector(pilotIdxs)
        pilotIdxs = repmat(pilotIdxs, 1, numTx);
    end

    if isvector(pilotsTx)
        pilotsTx = repmat(pilotsTx, 1, numTx);
    end 

    if size(pilotsTx, 3)  > 1
        pilotsTx_temp = zeros(size(pilotsTx, 1), size(pilotsTx, 2));
        for ntx = 1:numTx
            pilotsTx_temp(:, ntx) = pilotsTx(:, ntx, ntx);
        end
        pilotsTx = pilotsTx_temp;
    end
    
    for ntx = 1:numTx
        for nrx = 1:numRx
            H(:, ntx, nrx) = interp1(pilotIdxs(:, ntx), pilotsRx(:, ntx, nrx)./pilotsTx(:, ntx), subCarrierIdxs, 'spline', 0);
        end
    end        
end