function [ ghat_QIHT ] = func_Broadband_Few_Bit_UPA_QIHT( T_cyclic, r_bit, Params)
% Estimate the channel given the training signals and quantization outputs
% Inputs:
%   T_cyclic:   training signals
%   r_bit:      Quantization resolution
%   Params:     Parameters
% Output:
%   ghat:       estimate of the channel in the angular domain

% Channel model
Nt = Params.Nt;
Nta = Params.Nta;
Nte = Params.Nte;
Nr = Params.Nr;
Nra = Params.Nra;
Nre = Params.Nre;
Nd = Params.Nd;
Np = Params.Np;

SNR_linear = Params.SNR_linear;
bit = Params.bit;
stepsize = Params.stepsize;
% SNR_linear = Params.SNR_linear;
dither_mean = Params.dither_mean;
% wvar = Params.wvar;

% Bt = Params.Bt;
% Br = Params.Br;

V_cyclic = NaN(Nr*Nd, Np);
% V_train = Bt'* T_cyclic(1:Nt,:);
V_train = Bhfast(T_cyclic(1:Nt,:), Nt);
for ii=0:1:Nd-1
    V_cyclic( ii*Nt+1: (ii+1)*Nt, :) = circshift(V_train,ii,2);
end

% stepsize_vector = [1.5956 0.9957 0.586 0.3352 0.1881 0.1041 0.0569 0.0308];
% beta_vector = [0.3634 0.1188 0.03744 0.01154];


%% Compute mu (mu = 1 - beta) and sigma by simulations.  
% only consider the real part, so the noise variance is 1/2
% if bit == +inf
%     mu = 1;
%     sigma = sqrt(1/2);
% else
%     % r = Q(z + w) = mu * (z + w) + noise = mu * z + ( mu * w + noise )
% %     stepsize_scale_factor = Params.stepsize_scale_factor;
%     y_temp = randn(10000,1)*sqrt(Params.received_power_linear);
%     stepsize_temp =  stepsize;
%     r_temp = sign( y_temp ) .* ( min( ceil( abs( y_temp )/stepsize_temp) , 2^(bit-1) ) - 1/2 ) * stepsize_temp;
%     mu = mean(y_temp.*r_temp)/rms(y_temp)^2;
%     sigma = sqrt ( rms((r_temp - mu.*y_temp))^2 + mu^2 * 1/2 );
% %     corr = mean(z_temp.*r_temp)/rms(z_temp)/rms(r_temp)
% end;

%% reconstruct the signal based on the quantized output
if bit == +inf
    r = r_bit;
else
    r = (- 2^( bit -1 ) + real(r_bit) + 1/2 ) * stepsize + ...
        1j * (- 2^( bit -1 ) + imag(r_bit) + 1/2 ) * stepsize;
    r = r + dither_mean + 1j* dither_mean;
end;

opA = @(in, mode) Afast_Digital_UPA(in, T_cyclic, V_cyclic, Params, mode);
quantize = @(y, stepsize) sign(real(y)) .* ( min( ceil( abs(real(y)) /stepsize) , 2^(bit-1) ) - 1/2 ) * stepsize  + ...
    1j* sign(imag(y)) .* ( min( ceil( abs(imag(y)) /stepsize) , 2^(bit-1) ) - 1/2 ) * stepsize;

maxiter = 50; %1; % 50
stopcrit = 10^(-2.5); % 10^(-10); %
xn(:,1) = zeros(Nt*Nr*Nd,1);
mu = 1 - Params.beta;
%alpha = 0.1/(mu^2 * SNR_linear * Np/Nt);


alpha = Params.alpha;
maxiter = Params.maxiter;
stopcrit = Params.stopcrit;
K2 = Params.K2;

K1 = round(( log(Nt*Nr*Nd) )^2 * 6 / pi^2 * 5 );
%% Quantized iterative hard thresholding

% A = kron(V_cyclic.', Br);
tic;
for n = 1:maxiter
    
%     oxn = xn;
    
    % main QIHT iteration
    %     a = xn + alpha * A' * (r - Q(A*xn)); 
    Axn = opA(xn(:,n), 1); 
    if bit == +inf
        QAxn = Axn;
    else
%         stepsize = stepsize_vector(bit) * rms(Axn);
        QAxn = quantize(Axn, stepsize);
    end;
    
    a = xn(:,n) + alpha * opA(r - QAxn, 2);
    
    [~, aidx] = sort(abs(a),'descend');
    %     tau = 0.1 * sqrt(2) * sigma/(mu * sqrt(SNR_linear * Np/Nt));
    %     K2 = find( abs(a_sort) < tau, 1);
    
    %     K = max(K1, K2);
    %K2 = round(1.5*Nt*Nr*Nd/100);
    
    a(aidx(K2+1:end))=0;
    
    xn(:,n+1) = a;
   
    if (norm(xn(:,n+1)-xn(:,n))/norm(xn(:,n)) < stopcrit)
        break;
    end
end
timeQIHT = toc
if 0
    Gtrue = Params.Gtrue;
    figure,
    semilogy(sum(abs(bsxfun(@minus, xn, Gtrue(:))).^2)./norm(Gtrue(:))^2)
    10 * log10 ( sum(abs(bsxfun(@minus, xn, Gtrue(:))).^2)/norm(Gtrue(:))^2 )
    xlabel('Iteration')
    ylabel('Normalized MSE')
    title('QIHT')
    grid on;
end;

ghat_QIHT = xn(:,end);

end

