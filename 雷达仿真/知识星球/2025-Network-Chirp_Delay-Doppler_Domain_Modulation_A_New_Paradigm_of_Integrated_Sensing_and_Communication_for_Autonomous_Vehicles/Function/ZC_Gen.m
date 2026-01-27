function q = ZC_Gen(Nt)
%% Generate the ZC sequence
% input
%       Nt          # Transmit antenna
% output
%       q           ZC-sequence Nt×Nc
%%
N_ZC = 131;% prime is better
u_All = [];
for i = 1:N_ZC - 1
    if gcd(i, N_ZC) == 1
        u_All = [u_All i];
    end
end
c_f = mod(N_ZC, 2);
x = zeros(length(u_All), N_ZC);
p = 0;


%% Generation
for u = 1:length(u_All)
    for n = 1:N_ZC
        x(u,n) = exp(-1j*pi*u_All(u)*(n-1)*...
                (n-1+c_f+2*p)/...
                N_ZC);
    end
end
% Cut Down Nc dimension
x = x(:,1:128);
%% Cut Down Nt dimension
q = zeros(Nt,128);
for i = 1:Nt
    q(i,:) = x((i-1)*128/Nt+1,:);
end
end