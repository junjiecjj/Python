%% This demo shows how to minimizes the sum of CRBs (Co-design method)
clear; clc
%% new setting M=5
M = 5;                                       % the number of radar systems.
Nt = [4;2;2;2;2];                            % the number of transmit antennas.
Nr = [6;4;4;4;4];                            % the number of receive antennas.
sigma = 10*ones(5,1);                        % power of the noise. 
Pt = 10^3*ones(5,1);                         % transmit power budget 
L = 4;                                       % number of samples.
alpha = ones(5,1);                           % reflection coefficient.
theta = [1/6*pi;1/3*pi;1/4*pi;2/5*pi;3/7*pi];

Max_inter = 100;
test_num = 100;
test_results = zeros(Max_inter+1,test_num);
load('initial.mat');
%% Define A  

A = cell(M,1);
a = cell(M,1); a_prime = cell(M,1);
b = cell(M,1); b_prime = cell(M,1);

for m = 1:M
    a{m} = zeros(Nt(m),1); a_prime{m} = zeros(Nt(m),1);
    b{m} = zeros(Nr(m),1); b_prime{m} = zeros(Nr(m),1);
    for i =1:Nt(m)
        a{m}(i) = exp(-1i*pi*sin(theta(m))*(i-1));
        a_prime{m}(i) = -1i*pi*(i-1)*a{m}(i)*cos(theta(m));
    end
    for i = 1:Nr(m)
        b{m}(i) = exp(-1i*pi*sin(theta(m))*(i-1));
        b_prime{m}(i) = -1i*pi*(i-1)*b{m}(i)*cos(theta(m));
    end
    A{m} = b_prime{m}*a{m}'+b{m}*a_prime{m}';
end

G = cell(M,M);
for i = 1:M
    for j = 1:M
        G{i,j} = b{i}*a{j}'; % radar i transmits, radar j receives.
    end
end

for test = 1:test_num
    inter_result = zeros(Max_inter+1,1);
    %% S initialization
    s = used_data{test};
    for index = 1:Max_inter
        inter_result(index) = sum(Compute_CRB(M,A,s,G,L,alpha,sigma,Nr));
        %% s iteration 
        g = cell(M,1);
        each_results = zeros(M,M);
        each_s_update = cell(M,1);
        for m = 1:M
            each_s_update{m} = s;  
            cvx_begin
            variable sm(Nt(m)*L,1) complex  
            Q = sigma(m)*eye(Nr(m)*L);
            for i = 1:M
                if i~=m
                    Q  = (kron(eye(L),G{m,i})*s{i})*(kron(eye(L),G{m,i})*s{i})'+Q;
                end
            end
            E = kron(eye(L),A{m})'/Q*kron(eye(L),A{m});
            minimize -real(s{m}'*E*sm)
            subject to
            norm(sm)<=sqrt(Pt(m));
            cvx_end
            g{m}=sm;
            each_s_update{m}{m} = sm;
            each_results(:,m) = Compute_CRB(M,A,each_s_update{m},G,L,alpha,sigma,Nr); 
        end
        s=g;
    end

   
    inter_result(Max_inter+1) = sum(Compute_CRB(M,A,s,G,L,alpha,sigma,Nr));
    test_results(:,test) = inter_result;
end
