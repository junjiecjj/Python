function [HH,g,vv,pdb,tau,NN,t]= getchannel_related(M,N,delta_f,fs,fs_N,s,P,index_D,dop,beta,h,delayspread)
% 如果用tdl_A信道的时延分布的话，设置多径数量P=23
% 加了一个delayspread输入，可以设置为1e-7(100ns)或3e-8(30ns)或者其他
% tau_fix=[0 0.3 0.67 0.95 1.5].*1e-6;  %delay of the TU channel
% pdb_fix=[1 0.8 0.7 0.5 0.2];   %power of the TU  channel
tdl_A = [0,0.3819,0.4025,0.5868,0.4610,0.5375,0.6708,0.5750,0.7618,1.5375, ...
    1.8978,2.2242,2.1718,2.4942,2.5119,3.0582,4.0810,4.4579,4.5695,4.7966,5.0066,5.3043,9.6586];
tau_fix = tdl_A * delayspread;
pdb_A = [-13.4,0,-2.2,-4,-6,-8.2,-9.9,-10.5,-7.5,-15.9,-6.6,-16.7,-12.4,...
    -15.2,-10.8,-11.3,-12.7,-16.2,-18.3,-18.9,-16.6,-19.9,-29.7];
pdb = 10.^(pdb_A/10);
% tex = pi*(2*rand(1,length(pdb_fix))-1);
% pdb_fix=pdb_fix.*exp(tex*1i);


tau = tau_fix(1:P)+3*fs_N/fs;
% pdb = pdb_fix(1:P)./norm(pdb_fix(1:P));
pdb = h.*pdb;
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



