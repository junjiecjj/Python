function [Pmusic_dB, peak_coords] = nystromMusicDOA(L, M, mx, my, d, K, SNR_dB, Na, az_true, el_true)
    % Array manifold
    A = zeros(L, M);
    for k = 1:M
        ux = sind(el_true(k))*cosd(az_true(k));
        uy = sind(el_true(k))*sind(az_true(k));
        A(:,k) = exp(1j*2*pi*d*(mx*ux + my*uy));
    end
    
    % Received signal
    SNR = 10^(SNR_dB/10);
    S = (randn(M, K) + 1j*randn(M, K))/sqrt(2);
    W = (randn(L, K) + 1j*randn(L, K))/sqrt(2*SNR);
    X = A*S + W;
    
    % Nyström approximation
    indices = randperm(L, Na);
    Y = X(indices, :);
    A_y = A(indices, :);
    
    R_yy = (Y*Y')/K;
    R_xy = (X*Y')/K;
    
    [U_y, D_y] = eig(R_yy);
    [~, idx] = sort(diag(D_y), 'descend');
    U_y = U_y(:, idx);
    U_ns = R_xy * (U_y(:,1:M) / D_y(1:M,1:M));
    
    % MUSIC spectrum
    az_scan = -90:90; el_scan = 0:90;
    Pmusic = zeros(length(el_scan), length(az_scan));
    for i = 1:length(el_scan)
        for j = 1:length(az_scan)
            ux = sind(el_scan(i))*cosd(az_scan(j));
            uy = sind(el_scan(i))*sind(az_scan(j));
            a = exp(1j*2*pi*d*(mx*ux + my*uy));
            Pmusic(i,j) = 1/(a'*(eye(L) - U_ns*U_ns')*a);
        end
    end
    
    % Peak detection
    Pmusic_dB = 10*log10(abs(Pmusic)/max(abs(Pmusic(:))));
    BW = imregionalmax(Pmusic_dB);
    [rowMax, colMax] = find(BW);
    peak_coords = [az_scan(colMax)', el_scan(rowMax)'];
end