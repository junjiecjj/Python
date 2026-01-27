close all;
clear;
clc;

%% Computation Parameters
sys.N = 10;                                                                 % number of nodes
sys.T = 1000;                                                               % time horizon
sys.D_min = 20;                                                             % minimum data samples
sys.D_max = 20;                                                             % maximum data samples
sys.D_vary = 1;                                                             % batch data
sys.d = 784;                                                                % data dimension
sys.C = 10;                                                                 % number of classes

%% Computation Systems Generation
Case.nonIID_Gen = 0;                                                        % 1: generate non IID data
Case.Test_Gen = 0;                                                          % generate test data
Case.Test_SampleSize = 1000;                                                % test data size
Case.MAC_Gen = 1;                                                           % generate MAC

%% Communication Parameters
com.s = sys.d * sys.C;                                                      % subchannels
com.BW = 15*10^3;                                                           % 15 k for each subchannel
com.N0 = -174;                                                              % noise spectral density
com.NF = 10;                                                                % noise figure
com.noise_dBm = com.N0+10*log10(com.BW)+com.NF;                             % noise (dBm) per subchannel 15 kHz: -122 dBm
com.noise = Func_dBm2W(com.noise_dBm);                                      % noise (W) per subchannel
com.RX_SNR_noise_dBm = Func_W2dBm(com.noise*sys.d*sys.C);                   % noise
com.psi = 8;                                                                % shadowing
com.dis = [100,100];                                                        % distance of user to BS
com.Halpha = 0.997;                                                         % Gauss Markov constant

%% Simulation parameters
par.noniid = 1;                                                             % non iid data
par.dis = com.dis(1);                                                       % user distance
par.Pnbar_dBm = 16;                                                         % average power limit
par.Halpha = com.Halpha;                                                    % channel corralation
par.add_noise = 1;                                                          % add noise to global model
par.T = sys.T;                                                              % run time
par.channel = 500;                                                          % number of sub channels 
par.slot = sys.d*sys.C/par.channel;                                         % number of time slots
par.alpha = 1e5;                                                            % step size                                        
par.lambda = 5e-4;                                                          % power scaling
par.gamma = 2e-2;                                                           % step size

%% OMUAA
System_Initialization(Case,sys,com);
load('nonIID.mat');
TestData = load('test.mat');
load('MAC.mat');
[xtc,xtnc,lambda] = Func_OMUAA(Data,Label,H,Z,sys,par);
[Accuracy,Cost,Power] = Func_Performance(TestData,Data,Label,H,lambda,xtc,xtnc,sys,com,par); 
figure;
subplot(3,1,1);
plot(1:par.T,Accuracy);
ylabel({'$\textrm{Test accuracy}~\bar{A}(T)$ (\%)'},'Interpreter','latex');
xlabel({'Iterations $T$'},'Interpreter','latex'); 
subplot(3,1,2);
plot(1:par.T,Cost);
ylabel({'Training loss $\bar{f}(T)$'},'Interpreter','latex');
xlabel({'Iterations $T$'},'Interpreter','latex');
subplot(3,1,3);
plot(1:par.T,Power);
ylabel({'Transmit power $\bar{P}(T)$'},'Interpreter','latex');
xlabel({'Iterations $T$'},'Interpreter','latex');

%% System Initialization
function System_Initialization(Case,sys,com)
    if Case.nonIID_Gen == 1
        MNIST = load('MNIST_train.mat');
        [Data,Label,sys] = Func_nonIID_System(MNIST.Data,MNIST.Sample,sys);
        fprintf('nonIID Data Generated \n');
        tic;save('nonIID.mat','Data','Label','sys', '-v7.3');
        fprintf('nonIID Data Saved \n');
    end
    if  Case.Test_Gen == 1
        MNIST = load('MNIST_test.mat');
        SampleSize = Case.Test_SampleSize;
        idx = randperm(10000,SampleSize);
        Data = MNIST.Data(idx,2:785).';
        Label = MNIST.Data(idx,1).';
        tic;save('test.mat','Data','Label','SampleSize','-v7.3');
        fprintf('Test Data Saved, Sample:%d \n',SampleSize);
    end
    if Case.MAC_Gen == 1
        [H,Z,com] = Func_MAC_Gen(sys,com);
        fprintf('MAC Channel Generate \n');
        tic;save('MAC.mat','H','Z','com','-v7.3');
        fprintf('MAC Channel Saved \n');
    end
end

%% nonIID Case Generate
function [Data,Label,sys] = Func_nonIID_System(MNIST_Data,MNIST_Sample,sys)
   for t = 1:sys.T
        if t == 1
            sys.D(t,:) = randi([sys.D_min sys.D_max],1,sys.N);
            for n = 1:sys.N
                Sample = randi([1,MNIST_Sample(n)],1,sys.D(t,n));
                Data{t}{n} = MNIST_Data{n}(Sample,:);
            end
        else
            if sys.D_vary == 1
                sys.D(t,:) = randi([sys.D_min sys.D_max],1,sys.N);
                for n = 1:sys.N
                    Sample = randi([1,MNIST_Sample(n)],1,sys.D(t,n));
                    Data{t}{n} = MNIST_Data{n}(Sample,:);
                end
            else
                sys.D(t,:) = sys.D(t-1,:);
                for n = 1:sys.N
                    Data{t}{n} = Data{t-1}{n};
                end
            end
            
        end
   end
   for t = 1:sys.T
        for n = 1:sys.N
            Label{t}{n} = Data{t}{n}(:,1);
            Data{t}{n} = Data{t}{n}.';
            Data{t}{n}(1,:) = [];
        end
   end
end

%% MAC Chanenl Generation
function [H,Z,com] = Func_MAC_Gen(sys,com)
    C = sys.C;
    d = sys.d;
    com.U_pos = randi(com.dis,sys.N,1);
    for n = 1:sys.N
        com.PL_dB = -31.54 - 33*log10(abs(com.U_pos)) - com.psi*randn(sys.N,1);
        com.PL = (10.^(com.PL_dB/10));
    end
    for c = 1:C
        H{1}{c} = sqrt(diag(com.PL)/2)*(randn(sys.N,com.s/sys.C)+1i*randn(sys.N,com.s/sys.C));
        Z{1}{c} = sqrt(com.noise)*randn(d,1);
    end
    for t = 2:sys.T
        for c = 1:C
            Hz = sqrt((1-com.Halpha)*diag(com.PL)/2)*(randn(sys.N,com.s/sys.C)+1i*randn(sys.N,com.s/sys.C)); 
            H{t}{c} = com.Halpha * H{t-1}{c} + Hz;
            Z{t}{c} = sqrt(com.noise)*randn(d,1);
        end
    end
end

%% OMUAA
function [xtc,xtnc,lambda] = Func_OMUAA(Data,Label,H,Z,sys,par)
    N = sys.N;
    T = par.T;
    D = sys.D;
    d = sys.d;
    C = sys.C;
    alpha = par.alpha;
    gamma = par.gamma;
    lambda = par.lambda*ones(1,T);
    Pnc_bar = Func_dBm2W(par.Pnbar_dBm)/C*par.slot;   
    for c = 1:C
        xtc{1}{c} = zeros(d,1);
        for n = 1:N
            xtnc{1}{n}{c} = zeros(d,1);
            Qtnc{1}(n,c) = 1e1;
        end
    end
    for t = 1:T-1
        tic;
        w(t,:) = D(t,:)/sum(D(t,:));
        GD_tnc_f = zeros(d*C,N);
        for n = 1:N
            for i = 1:D(t,n)
                d_tni = Data{t}{n}(:,i);
                b_tni = Label{t}{n}(i);
                p_tni = zeros(1,C);
                for k = 1:C
                    p_tni(k) = exp( d_tni.' * xtc{t}{k});
                end
                hsum_tn = sum(p_tni);
                for c = 1:C
                    idx = d*(c-1)+1:d*c;
                    GD_tnc_f(idx,n) = GD_tnc_f(idx,n) - 1/D(t,n) * ( (b_tni == c-1) - p_tni(c)/hsum_tn ) * d_tni;
                end
            end
        end
        GD_tnc_g = zeros(d*C,N);
        for n = 1:N
            for c = 1:C
                idx = d*(c-1)+1:d*c;
                H_tnc = H{t+1}{c}(n,:);
                e_tnc = (1./abs(H_tnc).^2).';
                GD_tnc_g(idx,n) = (2 * lambda(t+1)^2 * w(t,n)^2 * e_tnc);
            end
        end
        for c = 1:C
            xtc{t+1}{c} = zeros(d,1);
        end
        for c = 1:C
            for n = 1:N
                idx = d*(c-1)+1:d*c;
                H_tnc = H{t+1}{c}(n,:);
                e_tnc = (1./abs(H_tnc).^2).';
                A_tnc = 2*alpha*ones(d,1)+ gamma*Qtnc{t}(n,c)*GD_tnc_g(idx,n);
                B_tnc = 2*alpha*xtc{t}{c} - GD_tnc_f(idx,n);
                xtnc{t+1}{n}{c} = (1./A_tnc) .* B_tnc;
                xtc{t+1}{c} =  xtc{t+1}{c} + w(t,n) * xtnc{t+1}{n}{c};
                gtnc{t}(n,c) = lambda(t+1)^2 * w(t,n)^2 * xtnc{t+1}{n}{c}.' * diag(e_tnc) * xtnc{t+1}{n}{c} - Pnc_bar;    
                Qtnc{t+1}(n,c) = max(0 , Qtnc{t}(n,c) + gtnc{t}(n,c));
            end
            if par.add_noise == 1
                xtc{t+1}{c} = xtc{t+1}{c} + 1/lambda(t+1) * Z{t+1}{c};
            end
        end
        time = toc;
        fprintf('OMUAA, t:%d \n',t);
    end
end

%% Performance
function [Accuracy,Cost,Power] = Func_Performance(TestData,Data,Label,H,lambda,xtc,xtnc,sys,com,par)
    T = par.T;
    C = sys.C;
    d = sys.d;
    D = sys.D;
    N = sys.N;
    Accuracy(1) = 0;
    for t = 1:T
        At = 0;
        wrong = 0;
        for i = 1:TestData.SampleSize
            d_i = TestData.Data(:,i);
            b_i = TestData.Label(i);
            h_ti = zeros(1,C);
            for k = 1:C
                h_ti(k) = exp( d_i.' * xtc{t}{k});
            end
            hsum_ti = sum(h_ti);
            [P_pred(i),idx] = max(h_ti/hsum_ti);
            Label_pred(i) = idx - 1;
            if TestData.Label(i) ~= Label_pred(i)
                wrong = wrong +1;
            end
        end
        At = (1 - wrong/TestData.SampleSize)*100;
        if t == 1
            Accuracy(t) = At;
        else
            Accuracy(t) = ( Accuracy(t-1) * (t-1) + At ) / t;
        end
        fprintf('t:%d, Accuracy: %.2f \n',t,Accuracy(t));
    end
    % Cost
    for t = 1:T
        Ct = 0;
        for n = 1:N
            for i = 1:D(t,n)
                d_tni = Data{t}{n}(:,i);
                b_tni = Label{t}{n}(i);
                h_tni = zeros(1,C);
                for k = 1:C
                    h_tni(k) = exp( d_tni.' * xtc{t}{k});
                end
                hsum_tn = sum(h_tni);
                Ct = Ct - log(exp( d_tni.' * xtc{t}{b_tni+1}) / hsum_tn );
            end
        end
        if t == 1
            Cost(t) = Ct/sum(D(t,:));
        else
            Cost(t) = ( Cost(t-1) * (t-1) + Ct/sum(D(t,:)) ) / t;
        end
        fprintf('t:%d, cost: %.2f \n',t,Cost(t));
    end
    Power(1) = 0;
    Noise_Power = com.noise * d * C;
    for t = 1:T-1
        TX_Pt = 0;
        w(t,:) = D(t,:)/sum(D(t,:));
        for n = 1:N
            for c = 1:C
                H_tnc = H{t+1}{c}(n,:);
                e_tnc = (1./abs(H_tnc).^2).';
                TX_Pt = TX_Pt + xtnc{t+1}{n}{c}.' * lambda(t+1)^2 * w(t,n)^2 * diag(e_tnc) * xtnc{t+1}{n}{c};
            end
        end
        Power(t+1) = ( Power(t) * t +  TX_Pt/(N*par.slot)  ) / (t+1);
        fprintf('t:%d, Pt: %.2f \n',t+1,Func_W2dBm(Power(t)));
    end
    Power = Func_W2dBm(Power);
end

%% W to dBm
function [dBm] = Func_W2dBm(P)
    N=max(size(P));
    for n=1:N
        dBm(n)=10*log10(P(n)*1e3);
    end
end

%% dBm to W
function [P] = Func_dBm2W(dBm)
    P = 10^(dBm/10)*1e-3;  
end
