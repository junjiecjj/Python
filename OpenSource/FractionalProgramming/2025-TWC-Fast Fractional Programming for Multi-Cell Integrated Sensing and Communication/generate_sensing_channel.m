function [d_G] = generate_sensing_channel(Nt,Nr,L,theta)
% M is L, num of BSs
%%
d_G = zeros(Nr,Nt,L);
a = zeros(Nt,L); a_prime = zeros(Nt,L);
b = zeros(Nr,L); b_prime = zeros(Nr,L);
for m = 1:L
    for i =1:Nt
        a(i,m) = exp(-1i*pi*sin(theta(m))*(i-1));
        a_prime(i,m) = -1i*pi*(i-1)*a(i,m)*cos(theta(m));
    end
    for i = 1:Nr
        b(i,m) = exp(-1i*pi*sin(theta(m))*(i-1));
        b_prime(i,m) = -1i*pi*(i-1)*b(i,m)*cos(theta(m));
    end
    d_G(:,:,m) = b_prime(:,m)*a(:,m).'+b(:,m)*a_prime(:,m).'; % \dot G_mm in our SPAWC paper
end
end