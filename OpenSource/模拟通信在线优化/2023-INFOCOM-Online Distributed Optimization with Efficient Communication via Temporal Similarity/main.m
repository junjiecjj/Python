close all;
clear;
clc;

%% Computation Parameters
sys.N = 10;                                                                 % number of nodes
sys.D = 20;                                                                 % batch data samples
sys.d = 784;                                                                % data dimension
sys.C = 10;                                                                 % number of class

%% Simulation Settings
par.T = 500;                                                                % time horizon                                         
par.xmax = 1e-3;                                                            % x_max
par.b = 5;                                                                  % quantization level
par.alpha = 1e5;                                                            % step size
par.gamma = 0.5;                                                            % step size
par.gn_bar = 1e-6;                                                          % limit on g
par.eta = 5e5;                                                              % step size

%% MNIST Data

TrainData = load('MNIST_Train.mat');
TestData = load('MNIST_Test.mat');

%% ODOTS

T = par.T;
C = sys.C;
d = sys.d;
D = sys.D;
N = sys.N;
alpha = par.alpha;
gamma = par.gamma;
eta = par.eta;
xmax = par.xmax;
b = par.b;                                                                  
granularity = 1/(2^b-1);                                                    
gn_bar = par.gn_bar;

for c = 1:C
    xtc_hat{1}{c} = zeros(d,1);
    for n = 1:N
        xtnc{1}{n}{c} = zeros(d,1);
        xtnc_hat{1}{n}{c} = zeros(d,1);
        Qnt_xtnc{1}{n}{c} = zeros(d,1);
        Qtn{1}(n) = 0;
        gtn{1}(n) = 0;
    end
end

w = ones(T,N)/N;   
Accuracy = zeros(1,T);
Lost = zeros(1,T);
Lost(1) = 2.3;
Power = zeros(1,T);
Marg_Entropy = zeros(1,T);
Cond_Entropy = zeros(1,T);
Queue = sum(Qtn{1})/N;

for t = 1:T-1
    % gradient computation
    GDf_tnc = zeros(d*C,N);
    for n = 1:N
        for i = 1:D
            d_tni = TrainData.Data{t}{n}(:,i);
            b_tni = TrainData.Label{t}{n}(i);
            h_tni = zeros(1,C);
            for c = 1:C
                h_tni(c) = exp( d_tni.' * xtc_hat{t}{c});
            end
            hsum_tn = sum(h_tni);
            for c = 1:C
                idx = d*(c-1)+1:d*c;
                GDf_tnc(idx,n) = GDf_tnc(idx,n) - 1/D * ( (b_tni == c-1) - h_tni(c)/hsum_tn ) * d_tni;
            end
        end
    end
    
    % local model update
    for c = 1:C
        for n = 1:N
            idx = d*(c-1)+1:d*c;
            if t >1
                xtmp = ( 2*alpha*xtc_hat{t}{c} - GDf_tnc(idx,n) + 2*eta*Qtn{t}(n)*xtnc_hat{t}{n}{c} )  / ( 2*alpha + 2*eta*Qtn{t}(n) );
            else
                xtmp = ( 2*alpha*xtc_hat{t}{c} - GDf_tnc(idx,n) ) / (2*alpha);
            end
            xtmp(xtmp>xmax) = xmax;
            xtmp(xtmp<-xmax) = -xmax;
            xtnc{t+1}{n}{c} = xtmp;
        end
    end

    % local virtual queue update
    for n = 1:N
        if t > 1
            gtn{t}(n) = - gn_bar;
            for c = 1:C
                idx = d*(c-1)+1:d*c;
                gtn{t}(n) = gtn{t}(n) + norm( xtnc{t+1}{n}{c} - xtnc_hat{t}{n}{c})^2;
            end
        else
            gtn{t}(n) = 0;
        end
        Qtn{t+1}(n) = max(0, (1-gamma^2)*Qtn{t}(n)+ gamma*eta*gtn{t}(n) );
    end

    % quantization
    for c = 1:C
        for n = 1:N
            dtnc = xtnc{t+1}{n}{c};
            Stnc = sign(dtnc);
            ltnc = floor(abs(dtnc)/(granularity*xmax)+1/2*ones(d,1));
            Qnt_xtnc{t+1}{n}{c} = Stnc .* ltnc;
            xtnc_hat{t+1}{n}{c} = xmax * Qnt_xtnc{t+1}{n}{c} * granularity ;
        end
    end

    % global model update
    for c = 1:C
        xtc_hat{t+1}{c} = zeros(d,1);
        for n = 1:N
            xtc_hat{t+1}{c} = xtc_hat{t+1}{c} + w(t,n) * xtnc_hat{t+1}{n}{c};
        end
    end
    
    % Accuracy
    wrong = 0;
    Label_pred = zeros(1,TestData.SampleSize);
    for i = 1:TestData.SampleSize
        d_i = TestData.Data(:,i);
        h_ti = zeros(1,C);
        for k = 1:C
            h_ti(k) = exp( d_i.' * xtc_hat{t+1}{k});
        end
        hsum_ti = sum(h_ti);
        [~,idx] = max(h_ti/hsum_ti);
        Label_pred(i) = idx - 1;
        if TestData.Label(i) ~= Label_pred(i)
            wrong = wrong +1;
        end
    end
    At = (1 - wrong/TestData.SampleSize)*100;
    Accuracy(t+1) = ( Accuracy(t) * t + At ) / (t+1);        
    
    % Lost
    Lt = 0;
    for n = 1:N
        for i = 1:D
            d_tni = TrainData.Data{t+1}{n}(:,i);
            b_tni = TrainData.Label{t+1}{n}(i);
            h_tni = zeros(1,C);
            for k = 1:C
                h_tni(k) = exp( d_tni.' * xtc_hat{t+1}{k});
            end
            hsum_tn = sum(h_tni);
            Lt = Lt - log(exp( d_tni.' * xtc_hat{t+1}{b_tni+1}) / hsum_tn );
        end
    end
    Lost(t+1) = ( Lost(t) * t + Lt/(D*N) ) / (t+1);
    
    % Decision Dissimilarity
    Pt = 0;
    for n = 1:N
        Ptn = 0;
        for c = 1:C
            Ptn = Ptn + norm( xtnc{t+1}{n}{c} - xtnc_hat{t}{n}{c})^2;
        end
        Pt = Pt + Ptn / N;
    end
    Power(t+1) =  ( Power(t) * t + Pt ) / (t+1) ;

    % Conditional Entropy
    MEt = 0;
    CEt = 0;
    for n = 1:N
        x = zeros(d*C,1);
        y = zeros(d*C,1);
        for c = 1:C
            idx = d*(c-1)+1:d*c;
            x(idx) = Qnt_xtnc{t}{n}{c};
            y(idx) = Qnt_xtnc{t+1}{n}{c};
        end
        [MEtn,CEtn,~] = Func_Entropy(x,y,b);
        MEt = MEt + MEtn * w(t+1,n) * C ;
        CEt = CEt + CEtn * w(t+1,n) * C;
    end
    Marg_Entropy(t+1) = Marg_Entropy(t) + MEt;
    Cond_Entropy(t+1) = Cond_Entropy(t) + CEt;

    % Virtual Queue
    Qt = sum(Qtn{t+1});
    Queue(t+1) = ( Queue(t) * t + Qt/N ) / (t+1); 

    fprintf('t:%d, A: %.2f, L: %.3f \n',t+1,Accuracy(t+1), Lost(t+1));

end

Accuracy(1) = nan;
Lost(1) = nan;
Power(1) = nan;
Marg_Entropy = Marg_Entropy * d / 1e6;
Marg_Entropy(1) = nan;
Cond_Entropy = Cond_Entropy * d / 1e6;
Cond_Entropy(1) = nan;

% Accuracy
subplot(2,2,1);
plot(1:par.T,Accuracy);
ylabel({'Test accuracy $\bar{A}(T)$ (\%)'},'Interpreter','latex');
xlabel({'Time $T$'},'Interpreter','latex'); 

% loss
subplot(2,2,2);
plot(1:par.T,Lost);
ylabel({'Training loss $\bar{f}(T)$'},'Interpreter','latex');
xlabel({'Time $T$'},'Interpreter','latex'); 

% Bits
subplot(2,2,3);
plot(1:par.T,Cond_Entropy);
ylabel({'Transmitted bits $B(T)$ (Mb)'},'Interpreter','latex');
xlabel({'Time $T$'},'Interpreter','latex','FontName','Times New Roman'); 
set(gca, 'YScale', 'log');

% Power
subplot(2,2,4);
plot(1:par.T,Power);
ylabel({'Decision dis-similarity $\bar{g}(T)$'},'Interpreter','latex');
xlabel({'Time $T$'},'Interpreter','latex'); 

%% Marginal & Conditional Entropy
function [Marg_Erp,Cond_Erp,MI] = Func_Entropy(x,y,b)

        lmax = 2^b - 1;
        edge = -lmax:1:lmax;
        levels = 2^(b+1)-1;
        Py = zeros(1,levels);
        count = histcounts(y,edge,'Normalization','Probability');
        Py(1:size(count,2)) = count;
        Px = zeros(1,levels);
        count = histcounts(x,edge,'Normalization','Probability');
        Px(1:size(count,2)) = count;
        Pxy = zeros(levels,levels);
        count2 = histcounts2(x,y,edge,edge,'Normalization', 'probability');
        Pxy(1:size(count2,1),1:size(count2,2)) = count2;

        Marg_Erp = 0;
        Cond_Erp = 0;
        for i = 1:levels
            if Py(i) ~= 0
                Marg_Erp = Marg_Erp - Py(i)*log2(Py(i));
            end
            for j = 1:levels
                if Pxy(i,j) ~= 0 && Px(i)~=0
                   Cond_Erp = Cond_Erp - Pxy(i,j)*log2(Pxy(i,j)/Px(i));
                end
            end
        end
        MI = Marg_Erp - Cond_Erp;

end

