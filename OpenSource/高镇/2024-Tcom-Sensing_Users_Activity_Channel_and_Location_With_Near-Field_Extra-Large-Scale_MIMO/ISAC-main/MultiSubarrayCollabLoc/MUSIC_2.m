function center = MUSIC_2(Y,fs,Subc_Index_set,N_subc)
% load channel2.mat
% load channel3.mat
[N, ~]  = size(Y);
range   = 3e8/fs*N/2;
thre    = 1e-10;
center  = 3e8/fs*N/2;
Nit     = 100;
[Ev, ~] = eig(Y*Y');
En  = Ev(:,1:(end-1));
f = -fs/2+Subc_Index_set*fs/N_subc;
k = 1j*2*pi*f/3e8;
while range > thre
    r     = linspace(center - range, center + range, Nit);
    P = zeros(1, Nit);
    for i = 1:Nit
%         a   = exp(1j*pi*theta_1(i)*(0:(N-1)));
        a = exp(r(i)*k);
%         a   = exp(-1i*2*pi*theta_1(i)*(-fs/2+((Subc_Index_set-1)*delta_subc+1)*fs/N_subc));
%         exp(1j*pi*theta_1(i)*(0:(N-1)));
        P(i) = 1/norm(a*En);
    end
    [~, index]  = max(abs(P));
    center  = center - range + 2*range/(Nit-1)*(index-1);
    range   = range/10;
end
end
% h_UE2Suba_klm*exp(-1i*2*pi*tau_klm*(-fs/2+((index_pp_1-1)*delta_subc+1)*fs/N_subc))*e_vec_klm;