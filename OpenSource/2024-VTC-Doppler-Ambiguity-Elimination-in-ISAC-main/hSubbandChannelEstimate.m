%hSubbandChannelEstimate practical subband channel estimation
%   [HEST,NVAR] = hSubbandChannelEstimate(CARRIER,RXGRID,REFIND,REFSYM,BUNDLESIZE)
%   performs channel estimation returning channel estimate H and noise
%   variance estimates NVAR. H is a K-by-N-by-R-by-P array where K is the
%   number of subcarriers, N is the number of OFDM symbols, R is the number
%   of receive antennas and P is the number of reference signal ports. NVAR
%   is an NSB-by-1 vector indicating the measured variance of additive
%   white Gaussian noise on the received reference symbols for every
%   subband.
%
%   CARRIER is a carrier configuration object, <a 
%   href="matlab:help('nrCarrierConfig')"
%   >nrCarrierConfig</a>. Only this
%   object property is relevant for this function:
%
%   CyclicPrefix      - Cyclic prefix ('normal', 'extended')
%
%   RXGRID is an array of size K-by-L-by-R. K is the number of subcarriers,
%   given by CARRIER.NSizeGrid * 12. L is the number of OFDM symbols in one
%   slot, given by CARRIER.SymbolsPerSlot.
%
%   REFIND and REFSYM are the reference signal indices and symbols,
%   respectively. REFIND is an array of 1-based linear indices addressing a
%   K-by-L-by-P resource array. P is the number of reference signal ports
%   and is inferred from the range of values in REFIND. Only nonzero
%   elements in REFSYM are considered. Any zero-valued elements in REFSYM
%   and their associated indices in REFIND are ignored.
%
%   BUNDLESIZE is the PRG bundle size (2, 4, or [] to signify 'wideband').
%
%   [H,NVAR,INFO] = hSubbandChannelEstimate(...,NAME,VALUE,...) specifies
%   additional options as NAME,VALUE pairs:
%
%   'CDMLengths'      - A 2-element row vector [FD TD] specifying the 
%                       length of FD-CDM and TD-CDM despreading to perform.
%                       A value of 1 for an element indicates no CDM and a
%                       value greater than 1 indicates the length of the
%                       CDM. For example, [2 1] indicates FD-CDM2 and no
%                       TD-CDM. The default is [1 1] (no orthogonal
%                       despreading)
%
%   'AveragingWindow' - A 2-element row vector [F T] specifying the number
%                       of adjacent reference symbols in the frequency
%                       domain F and time domain T over which to average
%                       prior to interpolation. F and T must be odd or
%                       zero. If F or T is zero, the averaging value is
%                       determined automatically from the estimated SNR
%                       (calculated using NVAR). The default is [0 0]
%

%  Copyright 2022-2023 The MathWorks, Inc.

function [Hest, noiseEst] = hSubbandChannelEstimate(carrier,rxGrid,refInd,refSym,bundleSize,varargin)

    % Dimensionality information for subband channel estimation
    K = carrier.NSizeGrid * 12;
    L = carrier.SymbolsPerSlot;
    R = size(rxGrid,3);
    P = size(refInd,2);
    
    % Get subcarrier indices 'k' used by the DM-RS, corresponding
    % PRG indices 'prg', and set of unique PRGs 'uprg'
    [k,~,~] = ind2sub([K L],refInd);
    [prg,uprg,prgInfo] = hPRGIndices(carrier,bundleSize,k(:,1));
    
    % Perform channel estimation for each PRG
    Hest = zeros([K L R P]);
    nVarPRGs = zeros(prgInfo.NPRG,1);
    for i = 1:numel(uprg)
    
        [HPRG,nVarPRGs(uprg(i))] = nrChannelEstimate(rxGrid,refInd(prg==uprg(i),:),refSym(prg==uprg(i),:),varargin{:});
        Hest = Hest + HPRG;
    
    end
    
    noiseEst = nVarPRGs(uprg);

end

function [prg,uprg,prgInfo] = hPRGIndices(carrier,bundleSize,k)
% Calculate PRG indices 'prg', and set of unique PRGs 'uprg' for subcarrier
% indices 'k'

    prgInfo = nrPRGInfo(carrier,bundleSize);
    rb = floor((k-1)/12);
    prg = prgInfo.PRGSet(rb+1);
    uprg = unique(prg).';

end