function [H,alpha,G] = generate_channel(L_B,L_U,L_T,Tx_power,Tx,Rx,Nr)
%   L_B: N by 2  location of BSs
%   L_U: N by 2 location of users
%   L_T: location of target
%   Tx_power: N by 1
%% generate channel
N = size(Tx_power,1); d = zeros(N,2); % the distance between 2 users and BS
H = cell(N,2);
for n = 1:N
    for i = 1:2
        d(n,i) = norm(L_B(n,:)-L_U(i,:));
    end
end
PL = zeros(N,2);
for n = 1:N
    for i = 1:2
        PL(n,i) = -32.6 - 36.7*log10(d(n,i));
    end
end
for n = 1:N
    for i = 1:2
        H{n,i} = sqrt(power(10,PL(n,i)/10))*(randn(Rx,Tx)+1i*randn(Rx,Tx))/sqrt(2);
    end
end
d11 = norm(L_B(1,:)-L_B(2,:)); PLBS = -32.6 - 36.7*log10(d11);
G = sqrt(power(10,PLBS/10))*(randn(Nr,Tx)+1i*randn(Nr,Tx))/sqrt(2);

alpha = sqrt(power(10,2*(-32.6-36.7*log10(norm(L_B(1,:)-L_T)))/10));
end