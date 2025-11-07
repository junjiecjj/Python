function [pilotIdxs, pilots] = generatePilotsHex(ofdm, Ntx, Nf, Nt, tShift)
% Build a hexagonal pilot lattice on the (subcarrier k, symbol l) plane.
%
% Inputs
%   ofdm.Nsub                  : FFT length / #subcarriers
%   ofdm.numGuardBandCarriers  : [left right] guard sizes
%   ofdm.NsymB (or ofdm.NumSymbols/ofdm.Nsym): # OFDM symbols in this subframe
%   Nf     : pilot spacing in frequency (e.g., 12)
%   Nt     : pilot spacing in time     (e.g., 2 or 4)
%   tShift : time shift applied to every other frequency column
%            default = floor(Nt/2)
%
% Outputs
%   pilotIdxs : [P x 2] list of pilot coordinates [k l]
%   pilots    : [P x Ntx x Ntx] per-pilot coding “matrix” rows
%               (we keep your diagonal filling: sequence at ( :, itx, itx ))

    if nargin < 5 || isempty(tShift), tShift = floor(Nt/2); end

    Nsub = ofdm.Nsub;
    Ng   = ofdm.numGuardBandCarriers;            % [left right]
    % Try common field names for #symbols
    if isfield(ofdm,'NsymB'), Nsym = ofdm.NsymB;
    elseif isfield(ofdm,'NumSymbols'), Nsym = ofdm.NumSymbols;
    elseif isfield(ofdm,'Nsym'), Nsym = ofdm.Nsym;
    else, error('OFDM struct must hold #symbols as NsymB/NumSymbols/Nsym'); end

    % -------- usable subcarriers (drop guard bands and DC) ----------
    kMin = Ng(1)+1;
    kMax = Nsub - Ng(2);
    kDC  = Nsub/2 + 1;                           % DC for even Nsub
    kCols = kMin:Nf:kMax;
    kCols(kCols==kDC) = [];                      % remove DC if present

    % -------- base time rows and hex shift ----------
    lBase = 1:Nt:Nsym;

    K = []; L = [];
    for c = 1:numel(kCols)
        k = kCols(c);
        if mod(c,2)==1           % columns 1,3,5,... unshifted
            l = lBase;
        else                     % columns 2,4,6,... shifted
            l = lBase + tShift;
        end
        l = l(l>=1 & l<=Nsym);
        K = [K, repmat(k,1,numel(l))]; %#ok<AGROW>
        L = [L, l];                   %#ok<AGROW>
    end

    pilotIdxs = [K(:) L(:)];     % P×2 pairs [k l]
    P = size(pilotIdxs,1);

    % -------- per-TX orthogonality (CDM): same (k,l), different sequences ----------
    % Keep your diagonal layout: pilots(:,itx,itx) carries TX-itx sequence.
    pilots = zeros(P, Ntx, Ntx);
    % Use your max-length sequence generator; ensure enough length:
    Lseq = max(P, Nsub-1);
    for itx = 1:Ntx
        s = mlseq(Lseq, itx);          % your PRBS per antenna
        s = s(1:P);                    % truncate
        pilots(:, itx, itx) = s;       % diagonal coding, as in your code
    end
end
