
function [Wprg,Hprg] = hSVDPrecoders(carrier,pdsch,H,prgsize)
%hSVDPrecoders Get PRG precoders from a channel estimate using SVD

%   Copyright 2023 The MathWorks, Inc.

    % Get PRG information
    prgInfo = nrPRGInfo(carrier,prgsize);

    % Average the channel estimate across all subcarriers in each RB and
    % across all OFDM symbols, extract allocated RBs, then permute to shape
    % R-by-P-by-NPRB where R is the number of receive antennas, P is the
    % number of transmit antennas and NPRB is the number of allocated RBs
    gridrbset = getGridRBSet(carrier,pdsch);
    [K,L,R,P] = size(H);
    H = reshape(H,[12 K/12 L R P]);
    H = mean(H,[1 3]);
    H = H(:,gridrbset + 1,:,:,:);
    H = permute(H,[4 5 2 1 3]);

    % For each PRG
    nu = pdsch.NumLayers;
    Wprg = zeros([nu P prgInfo.NPRG]);
    Hprg = zeros([R P prgInfo.NPRG]);
    pdschPRGs = prgInfo.PRGSet(gridrbset + 1);
    uprg = unique(pdschPRGs).';
    for i = uprg

        % Average the channel estimate across all allocated RBs in the PRG
        thisPRG = (pdschPRGs==i);
        Havg = mean(H(:,:,thisPRG),3);
        Hprg(:,:,i) = Havg;

        % Get SVD-based precoder for the PRG
        [~,~,V] = svd(Havg);
        W = permute(V(:,1:nu,:),[2 1 3]);
        W = W / sqrt(nu);
        Wprg(:,:,i) = W;

    end

end

% Get allocated RBs in the carrier resource grid i.e. relative to
% NStartGrid
function gridrbset = getGridRBSet(carrier,pdsch)

    if (pdsch.VRBToPRBInterleaving)
        subs = nrPDSCHIndices(carrier,pdsch, ...
            IndexStyle='subscript',IndexBase='0based');
        gridrbset = unique(floor(double(subs(:,1)) / 12));
    else
        if (isempty(pdsch.NStartBWP))
            bwpOffset = 0;
        else
            bwpOffset = pdsch.NStartBWP - carrier.NStartGrid;
        end
        gridrbset = pdsch.PRBSet + bwpOffset;
    end

end
