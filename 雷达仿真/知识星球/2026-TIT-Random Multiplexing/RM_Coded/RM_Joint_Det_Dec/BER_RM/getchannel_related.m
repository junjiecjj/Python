function [HH,g,vv,pdb,tau,NN,t]= getchannel_related(M,N,delta_f,fs,fs_N,s,P,index_D,dop,beta,h)

tau_fix=[0 0.3 0.67 0.95 1.5].*1e-6;  %delay of the TU channel
% pdb_fix=[1 0.8 0.7 0.5 0.2];   %power of the TU  channel

% tex = pi*(2*rand(1,length(pdb_fix))-1);
% pdb_fix=pdb_fix.*exp(tex*1i);


tau = tau_fix(1:P)+3*fs_N/fs;
% pdb = pdb_fix(1:P)./norm(pdb_fix(1:P));
pdb = h;
if index_D == 1
    theat = pi*(2*rand(1,P)-1);
    vv = dop*cos(theat);
%  v_m = round(dop*N/delta_f);
%  vv = delta_f/N*randi([-v_m,v_m],1,P);
else
    vv = zeros(1,P);
end

NN = 0:1/fs:max(tau)+3*fs_N/fs;

t = [0:1/fs:length(s)/fs-1/fs];
g = [];
for mc = 1:length(t)
for i=1:length(NN)
    for pp = 1:P
        hh(pp) = pdb(pp)*exp(1i*2*pi*vv(pp)*(t(mc)-NN(i)))*myrrc(beta,NN(i)-tau(pp),fs/fs_N,1);
    end
    hhh(i) = sum(hh);
end
g(mc,:) = hhh;
end

HH = zeros(N*M);
for i = 1:P
    mu_k(i) = round(vv(i)*N/delta_f);
    if vv(i)>0
        beta_k(i) = -(abs(mu_k(i)*delta_f/N)-abs(vv(i)))/delta_f*N;
    else beta_k(i) = (abs(mu_k(i)*delta_f/N)-abs(vv(i)))/delta_f*N;
    end
end
mu_k;
beta_k;


for l = 0:M-1
    for k = 0:N-1
        k1 = k+1;
        for pp1 = 1:length(NN)
            pp = pp1 - 1;
            Tep = 0;
            for i = 1:P
            for q = 0:N-1
            Tep = pdb(i)*exp(1i*2*pi*(l-pp)*(mu_k(i)+beta_k(i))/M/N)*myrrc(beta,NN(pp1)-tau(i),fs/fs_N,1)*((exp(-1i*2*pi*(-q-beta_k(i)))-1)/(exp(-1i*2*pi*(-q-beta_k(i))/N)-1))/N;
            if l < pp
                Tep = Tep*exp(-1i*2*pi*mod(k-mu_k(i)+q,N)/N);
            end 
            HH(k*M+l+1,mod(k-mu_k(i)+q,N)*M+mod(l-pp,M)+1) = HH(k*M+l+1,mod(k-mu_k(i)+q,N)*M+mod(l-pp,M)+1) + Tep;
            end
            end
        end
    end
end

end



