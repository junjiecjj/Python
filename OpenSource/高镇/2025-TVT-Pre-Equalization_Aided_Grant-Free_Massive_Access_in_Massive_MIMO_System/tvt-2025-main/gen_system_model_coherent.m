function sample = gen_system_model_coherent(parIn)
% Generate system model for coherent detection 

actFlag = zeros(parIn.numUEs,1);
idActUEs = randperm(parIn.numUEs,parIn.numActUEs);
idActUEs = sort(idActUEs);
actFlag(idActUEs) = 1;

dist = (parIn.distMax-parIn.distMin)*rand(parIn.numUEs,1) ...
    + parIn.distMin;
loss = 10.^(- (parIn.lossA + parIn.lossB.*log10(dist))/10);

chSpaAll = zeros(parIn.numBSAntennas,parIn.numUEs,parIn.lenSprCodes);
chAngAll = zeros(parIn.numBSAntennas,parIn.numUEs,parIn.lenSprCodes);
chSpaAct = zeros(parIn.numBSAntennas,parIn.numUEs,parIn.lenSprCodes); % Active
chAngAct = zeros(parIn.numBSAntennas,parIn.numUEs,parIn.lenSprCodes); % Active
%chSpaAmp = zeros(parIn.numBSAntennas,parIn.numUEs,parIn.lenSprCodes); % Amplified
%chAngAmp = zeros(parIn.numBSAntennas,parIn.numUEs,parIn.lenSprCodes); % Amplified

% Generate Channel for each user
for idxUEs = 1:parIn.numUEs
    Lp = randi([parIn.Lp_min,parIn.Lp_max],1,1);
    [chSpaTemp, chAngTemp] = one_ring_channel(parIn.numBSAntennas, parIn.lenSprCodes,...
        parIn.lenSprCodes, parIn.bandWidth, parIn.fs, parIn.lenCP, parIn.pathGain,...
        Lp, parIn.channelTh,parIn.angleSpread);
    chSpaAll(:,idxUEs,:) = sqrt(loss(idxUEs))*chSpaTemp;
    chAngAll(:,idxUEs,:) = sqrt(loss(idxUEs))*chAngTemp;
end

for indexActUEs = 1:parIn.numActUEs
    chSpaAct(:,idActUEs(indexActUEs),:) = chSpaAll(:,idActUEs(indexActUEs),:);
    chAngAct(:,idActUEs(indexActUEs),:) = chAngAll(:,idActUEs(indexActUEs),:);
end

% Power Amplification
chSpaAmp = parIn.PA*chSpaAct;
%chAngAmp = parIn.PA*chAngAmp;
parIn.varNoiseAmp = parIn.PA^2.*parIn.varNoise;


txRef = (randn(parIn.numUEs, parIn.numOFDMSymbols, parIn.lenSprCodes) +...
    1j*randn(parIn.numUEs, parIn.numOFDMSymbols, parIn.lenSprCodes))/sqrt(2);
rxRef = sqrt(parIn.PsMax).*pagemtimes(chSpaAmp,txRef);

sprCodes = (randn(parIn.lenSprCodes,parIn.numUEs) + 1j*randn(parIn.lenSprCodes,parIn.numUEs))/sqrt(2);
txPldNum = randi([0 parIn.modOrder-1],parIn.numUEs,parIn.numOFDMSymbols);
txPldMod = pskmod(txPldNum, parIn.modOrder, 0, 'gray');
txPld = repmat(actFlag,1,parIn.numOFDMSymbols) .* txPldMod;
rxPld = zeros(parIn.lenSprCodes,parIn.numOFDMSymbols,parIn.numBSAntennas);

for idxA = 1:parIn.numBSAntennas
    rxPld(:,:,idxA) = parIn.PA*(squeeze(chSpaAct(idxA,:,:)).'.*sprCodes) * txPld;
end

awgnRef = sqrt(parIn.varNoiseAmp/2).* (randn(parIn.numBSAntennas, parIn.numOFDMSymbols) + ...
    1i.*randn(parIn.numBSAntennas, parIn.numOFDMSymbols));
awgnPld = sqrt(parIn.varNoiseAmp/2).* (randn(parIn.lenSprCodes,parIn.numOFDMSymbols,parIn.numBSAntennas) + ...
    1i.*randn(parIn.lenSprCodes,parIn.numOFDMSymbols,parIn.numBSAntennas));

rxRef = rxRef + awgnRef;
rxPld = rxPld + awgnPld;

sample.txRef = txRef;
sample.txPld = txPld;
sample.txPldNum = txPldNum;
sample.rxRef = rxRef;
sample.rxPld = rxPld;
sample.actFlag = actFlag;
sample.chSpaAct = chSpaAct;
sample.idActUEs = idActUEs;
sample.sprCodes = sprCodes;
end

