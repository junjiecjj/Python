function [pilotIdxs, pilots] = generateRectPilots(ofdm, systemParams)
% Rectangular pilot grid restricted to Subframe B.
%
% Inputs (required in ofdm):
%   ofdm.Nsub                  : #subcarriers (FFT length)
%   ofdm.numGuardBandCarriers  : [left right] guard sizes
%   ofdm.Mt                    : #symbols in Subframe A   (e.g., 14)
%   ofdm.Ntx                   : #symbols in Subframe B   (== Ntx, e.g., 8)
% Optional args:
%   Nf   : pilot spacing in frequency (default: 12)
%   NtB  : pilot spacing in time *within Subframe B* (default: 2)
%
% Outputs:
%   pilotIdxs : [P x 2] integer pairs [k l] (subcarrier, symbol)
%   pilots    : [P x Ntx x Ntx] symbols (diagonal per-TX sequences)

    Nsub = ofdm.Nsub;
    Ng   = ofdm.numGuardBandCarriers;  % [left right]
    Mt   = ofdm.Mt;                    % symbols in Subframe A (no pilots)
    Mf   = ofdm.Mf;                     
    Ntx   = systemParams.Ntx;                   % symbols in Subframe B (== Ntx)
    NB = Ntx;                    % shorthand for pilots in Subframe B

    % usable subcarriers (drop guards + DC)
    kMin = Ng(1)+1; kMax = Nsub - Ng(2);
    kDC  = Nsub/2 + 1;
    kVec = kMin:Mf:kMax;
    kVec(kVec==kDC) = [];

    % split evenly across Tx so **per symbol** sets are disjoint
    KperTot = numel(kVec);
    Kper    = floor(KperTot / Ntx);         % pilots per Tx, per symbol
    kVec    = kVec(1:Kper*Ntx);             % trim to multiple of Ntx
    kMat    = reshape(kVec, Ntx, Kper);     % row tx -> its subcarriers

    % Build indices: each B-symbol l uses the same FDM partition
    pilotIdxs = zeros(Kper, NB, Ntx, 'like', kMat);
    for tx = 1:Ntx
        pilotIdxs(:, :, tx) = repmat(kMat(tx,:).', 1, NB);
    end

    % Per-antenna pilot symbols (CDM sequences, but indices are disjoint)
    pilots = zeros(Kper, NB, Ntx);
    Lseq   = max(Kper*NB, Nsub-1);
    for tx = 1:Ntx
        s = mlseq(Lseq, tx);                 % your PRBS
        pilots(:,:,tx) = reshape(s(1:Kper*NB), Kper, NB);
    end
end
