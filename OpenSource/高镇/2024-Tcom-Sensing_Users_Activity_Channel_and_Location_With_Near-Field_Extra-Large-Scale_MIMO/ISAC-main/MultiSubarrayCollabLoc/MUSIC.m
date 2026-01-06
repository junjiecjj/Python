function center = MUSIC(Y)
[N, ~]  = size(Y);
range   = 1;
thre    = 1e-10;
center  = 0;
Nit     = 20;
[Ev, ~] = eig(Y*Y');
En  = Ev(:,1:(end-1));
% f = -fs/2:fs/N:(fs/2-fs/N);
% k = 2*pi*f/3e8;
while range > thre
    theta_1     = linspace(center - range, center + range, Nit);
    P = zeros(1, Nit);
    for i = 1:Nit
        a = exp(-1j*pi*theta_1(i)*(0:(N-1)));
        P(i) = 1/norm(a*En);
    end
    [~, index]  = max(abs(P));
    center  = center - range + 2*range/(Nit-1)*(index-1);
    range   = range/10;
end
end