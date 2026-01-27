


function H = channel(N,K)
    L = 15;
    theta = 2*pi*rand(K,L);%均匀分布在[0,2*pi)
    H = zeros(K,N);
    a = zeros(1,N);
    for x = 1:1:K % K个用户的信道
        temp = zeros(1,N);
        for l = 1:1:L
            %求 a_t (N*1)
            for n = 1:1:N
                a(n) = exp(1j *(n-1)* pi * sin(theta(x,l)));
            end
            a = a/sqrt(N);
            temp = temp + randn()*1*conj(a); %a_r = 1;
        end
        H(x,:)=sqrt(N*1/L)*temp; % M=1
    end
end


function V_RF = change_V_RF()
    global N N_RF
    global H_t V_RF
    A = cell(1,9);
    B = cell(1,9);
    D = cell(1,9);
    zeta_B = zeros(N,N_RF);
    zeta_D = zeros(N,N_RF);
    eta_B = zeros(N,N_RF);
    eta_D = zeros(N,N_RF);
    c = zeros(N,N_RF);
    z = zeros(N,N_RF);
    phi = zeros(N,N_RF);
    for j = 1:N_RF
        V_RF_bar = V_RF;
        V_RF_bar(:,j) = [];
        A{j} = H_t * V_RF_bar*(V_RF_bar')* H_t';
        % 计算B_j,D_j,用于计算zeta和eta
        B{j} = H_t'* (A{j}^(-2)) * H_t;
        D{j} = H_t'* (A{j}^(-1)) * H_t;
        for i = 1:N
            % 计算zeta和eta
            temp = 0;
            for m = 1:1:N
                if( m~=i )
                    for n = 1:1:N
                        if(n~=i)
                            temp = temp + conj(V_RF(m,j))*B{j}(m,n)*V_RF(n,j);
                        end
                    end
                end
            end
            zeta_B(i,j)= B{j}(i,i)+2*real(temp);
            temp = 0;
            for m = 1:1:N
                if( m~=i )
                    for n = 1:1:N
                        if(n~=i)
                            temp = temp + conj(V_RF(m,j))*D{j}(m,n)*V_RF(n,j);
                        end
                    end
                end
            end
            zeta_D(i,j)= D{j}(i,i)+2*real(temp);
            V_B = B{j} * V_RF;
            V_D = D{j} * V_RF;
            eta_B(i,j) = V_B(i,j) - B{j}(i,i)*V_RF(i,j);
            eta_D(i,j) = V_D(i,j) - D{j}(i,i)*V_RF(i,j);
            % 计算theta_1，theta_2
            c(i,j) = (1 + zeta_D(i,j)) * eta_B(i,j) - zeta_B(i,j) * eta_D(i,j);
            z(i,j) = imag( 2 * eta_B(i,j) * eta_D(i,j));
            phi(i,j) = 0;
            tt = asin(imag(c(i,j))/abs(c(i,j)));
            if(real(c(i,j))>=0) 
                phi(i,j) = tt; 
            else 
                phi(i,j) = pi - tt; 
            end
            theta_1 = -phi(i,j) + asin(z(i,j)/abs(c(i,j)));
            theta_2 = pi - phi(i,j) - asin(z(i,j)/abs(c(i,j)));
            % 判断最优theta
            V_RF1 = exp(-1j * theta_1);
            V_RF2 = exp(-1j * theta_2);
            f1 = N * trace(A{j}^(-1)) -  N * ( zeta_B(i,j) + 2 * real(conj(V_RF1)*eta_B(i,j)))/ (1+zeta_D(i,j)+2*real(conj(V_RF1)*eta_D(i,j)));
            f2 = N * trace(A{j}^(-1)) - N * ( zeta_B(i,j) + 2 * real(conj(V_RF2)*eta_B(i,j)))/ (1+zeta_D(i,j)+2*real(conj(V_RF2)*eta_D(i,j)));
            if(f1 < f2) 
                theta_opt = theta_1; 
            else 
                theta_opt = theta_2;
            end
            V_RF(i,j) = exp(-1j*theta_opt);
        end
    end
end

% https://github.com/cao-mengyuan/HBF-simulation-code
clear;
clc;
close all;

global N N_RF K P_t
global H H_t P
global V_RF

K = 8;
N = 64;
N_RF = 9;

P_t = K;
beta = ones(1,K);% beta_k = 1; %均匀分配权重
sigma2 = K;

SNR = P_t/sigma2;% SNR = P/sigma^2;
SNR_dB = 10 * log10(SNR);

H = channel(N,K); % 生成信道

% 生成随机可行解
% 从可行的一个解开始
P = eye(K);
% V_RF = ones(N,N_RF);
tt = 2*pi*rand(1,N*N_RF);
ttt = exp(1j*tt); 
V_RF = reshape(ttt,N,N_RF);
% 生成随机数，才能保证V_RF满秩，否则后面的A_j不满秩

H_t = P^(-0.5) * H; 

% 预分配空间
for loop = 1:20
    V_RF_last = V_RF;
    V_RF = change_V_RF();
end

% 生成功率分配矩阵
V_D_t = (V_RF') * (H')/ (H * V_RF * (V_RF') * (H'));
Q_t = (V_D_t') * (V_RF') * V_RF * V_D_t;

% 迭代求出lamda
lamda = 1;
while 1
    initPower = 0;
    posi = 0;
    for k = 1:1:K
        tttt = (beta(k)/lamda) - Q_t(k,k)*sigma2;
        if(tttt > 0 )
            initPower = initPower + tttt;
            posi = posi + 1;
        end
    end
    if( abs(initPower / P_t -1) <= 0.05 )
        disp("find P");
        break;
    end
    if(posi > 0) 
        lamda = lamda + 0.5*(initPower - P_t)/posi;
    else 
        lamda = lamda/4;
    end
end
% 求出P
P = zeros(K,K);
for kk = 1:1:K
    P(kk,kk) = max([(beta(kk)/lamda) - Q_t(kk,kk)*sigma2, 0]) / Q_t(kk,kk);
end
% V_D
V_D = (V_RF') * (H') / ( H *V_RF * (V_RF') *(H'));
% 计算 R_k 求和
R = zeros(1,K);
for k = 1:1:K
    R(k) = beta(k) * log2(1+(P(k,k)/sigma2));
end
Sum_R = sum(R)




