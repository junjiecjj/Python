close all;
clear;
clc;

%% System Parameters
par.Rc = 500;                                                               % cell radius
par.N = 48;                                                                 % number of antennas
par.C = 3;                                                                  % number of cells C
par.K = 15;                                                                 % number of uses per cell
par.M = 1;                                                                  % number of SPs M
par.P_max = dBm2W(33);                                                      % maximum power
par.N_max = 48;                                                            % maximum antennas
par.BW = 15*1e3;                                                            % system bandwidth 15k
par.N0 = -174;                                                              % -174 dbm/Hz
par.NF = 10;                                                                % noise figure 10 dB
par.noise = dBm2W(par.N0+10*log10(par.BW)+par.NF);                          % noise
par.alpha_array = [0.998];                                                  % alpha_h array
par.alpha = 0.998;                                                          % correlation alpha_h
par.T = 100;                                                                % time horizon
par.tau_l = 0;                                                              % local delay
par.tau_r = 1;                                                              % remote delay
par.eta = 2e6;                                                              % step size                                                        
par.J = 1;                                                                  % steps at CC
par.J_l = 1;                                                                % steps at APs

%% SOPIP
load('Cell.mat');
load('CSI.mat');
[ft,Rt] = SOPIP(par,H_GM);      
figure;
subplot(2,1,1);
plot(1:par.T,ft);
ylabel({'$\bar{f}(T)$ (\%)'},'Interpreter','latex');
xlabel({'$T$'},'Interpreter','latex');
subplot(2,1,2);
plot(1:par.T,Rt);
ylabel({'$\bar{R}(T)$ (bpcu)'},'Interpreter','latex');
xlabel({'$T$'},'Interpreter','latex');

%% SOPIP Algorithm
function [ft,Rt] = SOPIP(par,H_GM)
    C = par.C;
    N = par.N;
    Nc = N/C;
    K = par.K;
    P_max = par.P_max;
    Pc_max = P_max/C;
    J = par.J;
    J_l = par.J_l;
    T = par.T;
    tau_l = par.tau_l;
    tau_r = par.tau_r;
    eta = par.eta;
    [H,D] = Demand(par,H_GM);
    for t = 1:T
        V{t} = [];
        X{t} = [];
        if t <= tau_l + tau_r
            for c = 1:C
                tmp = ones(Nc,K);
                Vc{t}{c} = zeros(Nc,K);
                Xc{t}{c} = zeros(Nc,K);
                V{t} = [V{t};Vc{t}{c};];
                X{t} = [X{t};Xc{t}{c};];
            end
        else
            for j = 1:J
                if j == 1
                    XnI = V{t-max(tau_l,tau_r)};
                else
                    for c = 1:C
                        H_tau_c = H{t-tau_l-tau_r}(:,Nc*(c-1)+1:Nc*c);
                        XcnI = XnI(Nc*(c-1)+1:Nc*c,:);
                        tmp = 	XcnI - 1/eta * H_tau_c'*(H{t-tau_l-tau_r}*XnI-D{t-tau_l-tau_r});
                        Pc = (norm(tmp,'fro'))^2;
                        if Pc <= Pc_max
                            XcnI = tmp;
                        else
                            XcnI = sqrt(Pc_max/Pc)*tmp;
                        end 
                        XnI(Nc*(c-1)+1:Nc*c,:) = XcnI;
                    end
                end
            end
            X{t} = XnI;
            for c = 1:C
                Xc{t}{c} = X{t}(Nc*(c-1)+1:Nc*c,:);
            end
            for c = 1:C
                xtld_c = Xc{t}{c};
                H_t_c = H{t-tau_l}(:,Nc*(c-1)+1:Nc*c);
                H_tauc = H{t-tau_l-tau_r};
                H_tauc(:,Nc*(c-1)+1:Nc*c) = H_t_c; 
                for jl = 1:J_l
                    X{t}(Nc*(c-1)+1:Nc*c,:) = xtld_c;
                    tmp = xtld_c - 1/eta*H_t_c'*(H_tauc*X{t}-D{t-tau_l-tau_r});
                    Pc = (norm(tmp,'fro'))^2;
                    if Pc <= Pc_max
                        xtld_c = tmp;
                    else
                        xtld_c = sqrt(Pc_max/Pc)*tmp;
                    end
                end
                V{t} = [V{t};xtld_c;];
            end
        end
    end
    [ft,Rt] = Performance(par,H,V,D);
end

%% Demand
function [H,D] = Demand(par,H_GM)
    C = par.C;
    M = par.M;
    N = par.N;
    Nc = N/C;
    K = par.K;
    Km = K/M;
    N_max = par.N_max;
    Nc_max = N_max/C;
    P_max = par.P_max;
    P_max_c = P_max/C;
    T = par.T;
    N_pick = [1:Nc];
    for c = 1:C-1
        N_pick = [N_pick,Nc_max*c+1:Nc_max*c+Nc];
    end
    idx = find(par.alpha == par.alpha_array);
    H_GMa = H_GM{idx};  
    for t = 1:T
        H{t} = H_GMa{t}(1:K,N_pick);
        D{t} = [];
        for m = 1:M
            Hm = H{t}(Km*(m-1)+1:Km*m,:);
            tmp = Hm'*pinv(Hm*Hm');        
            Wm = sqrt(1/M)/(norm(tmp,'fro'))*tmp;
            D{t} = blkdiag(D{t},Hm*Wm);                   
        end
        for c = 1:C
            Hc = H{t}(:,Nc*(c-1)+1:Nc*c);
            PcD(c) = P_max_c/norm(Hc'*pinv(H{t}*H{t}')*D{t},'fro')^2;
        end
        PD = min(PcD);
        D{t} = sqrt(PD)*D{t};
    end    
end

%% Performance Measure
function [ft,Rt] = Performance(par,H,V,D)
    T = par.T;
    K = par.K;
    M = par.M;
    Km = K/M;
    for t = 1:T
        Dt = H{t}*V{t};
        for k = 1:K
            S(k)=abs(Dt(k,k))^2;
            I(k)=sum(abs(Dt(k,:)).^2)-S(k);
            SINR(k)=S(k)/(I(k)+1);
            DR(k)=log2(1+SINR(k));
        end
        for m = 1:M
            Rm_min(m) = min(DR(Km*(m-1)+1:Km*m));
        end
        R(t) = mean(DR);
        f(t) = norm(H{t}*V{t}-D{t},'fro')^2/norm(D{t},'fro')^2*100;  
    end
    for t = 1:T     
        ft(t) = mean(f(1:t));
        Rt(t) = mean(R(1:t));
    end
end

%% dBm to W
function [P] = dBm2W(dBm)
    P = 10^(dBm/10)*1e-3;  
end
