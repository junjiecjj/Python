%% This demo shows how to minimize the sum of CRBs.
% This work was done by Yannan CHEN from CUHK (SZ).
% If you have any questions regarding this code, please feel free to contact me at yannanchen@link.cuhk.edu.cn.

clear; clc
%% new setting M=5
M = 5;                                       % the number of radar systems.
Nt = [4;2;2;2;2];
Nr = [6;4;4;4;4];
sigma = 10*ones(5,1);
Pt = 10^3*ones(5,1);
L = 4;                                      % number of samples.
alpha = ones(5,1);                           % reflection coefficient.
theta = [1/6*pi;1/3*pi;1/4*pi;2/5*pi;3/7*pi]; 
Max_inter = 100;
test_num = 1;
test_results = zeros(Max_inter+1,test_num);
load('initial.mat'); % some randomly generated initial point
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
    A{m} = b_prime{m}*a{m}.'+b{m}*a_prime{m}.';
end

G = cell(M,M);
for i = 1:M
    for j = 1:M
        G{i,j} = b{i}*a{j}'; % radar i transmits, radar j receives.
    end
end
for test = 1+1:test_num+1
    inter_result = zeros(Max_inter+1,1);
    %% S initialization
    s = initial;
    for index = 1:Max_inter
        inter_result(index) = sum(Compute_CRB(M,A,s,G,L,alpha,sigma,Nr));
        %% find optimal y
        Y = cell(M,1);
        for m = 1:M
            Y{m} = zeros(Nr(m)*L,1);
            %% compute Q
            Q = sigma(m)*eye(Nr(m)*L);
            for i = 1:M
                if i~=m
                    Q = (kron(eye(L),G{m,i})*s{i})*(kron(eye(L),G{m,i})*s{i})'+Q;
                end
            end
            Y{m}=Q\(kron(eye(L),A{m})*s{m});
        end
        cvx_begin
        
        variable s1(Nt(1)*L,1) % If you find a way to define the variables and constraints more concisely
        variable s2(Nt(2)*L,1) % please feel free to let me know.
        variable s3(Nt(3)*L,1)
        variable s4(Nt(4)*L,1)
        variable s5(Nt(5)*L,1)

        variable t1
        variable t2
        variable t3
        variable t4
        variable t5

        variable s11(Nt(1)*L,Nt(1)*L)
        variable s22(Nt(2)*L,Nt(2)*L)
        variable s33(Nt(3)*L,Nt(3)*L)
        variable s44(Nt(4)*L,Nt(4)*L)
        variable s55(Nt(5)*L,Nt(5)*L)
   
        expression Q1
        expression Q2
        expression Q3
        expression Q4
        expression Q5

        Q1 = kron(eye(L),G{1,2})*s22*kron(eye(L),G{1,2})'+...
            kron(eye(L),G{1,3})*s33*kron(eye(L),G{1,3})'+...
            kron(eye(L),G{1,4})*s44*kron(eye(L),G{1,4})'+...
            kron(eye(L),G{1,5})*s55*kron(eye(L),G{1,5})'+sigma(1)*eye(Nr(1)*L); % W22 - W2*W2' == hermitian_semidefinite(Nt(2))

        Q2 = kron(eye(L),G{2,1})*s11*kron(eye(L),G{2,1})'+...
            kron(eye(L),G{2,3})*s33*kron(eye(L),G{2,3})'+...
            kron(eye(L),G{2,4})*s44*kron(eye(L),G{2,4})'+...
            kron(eye(L),G{2,5})*s55*kron(eye(L),G{2,5})'+sigma(2)*eye(Nr(2)*L);

        Q3 = kron(eye(L),G{3,1})*s11*kron(eye(L),G{3,1})'+...
            kron(eye(L),G{3,2})*s22*kron(eye(L),G{3,2})'+...
            kron(eye(L),G{3,4})*s44*kron(eye(L),G{3,4})'+...
            kron(eye(L),G{3,5})*s55*kron(eye(L),G{3,5})'+sigma(3)*eye(Nr(3)*L);

        Q4 = kron(eye(L),G{4,1})*s11*kron(eye(L),G{4,1})'+...
            kron(eye(L),G{4,2})*s22*kron(eye(L),G{4,2})'+...
            kron(eye(L),G{4,3})*s33*kron(eye(L),G{4,3})'+...
            kron(eye(L),G{4,5})*s55*kron(eye(L),G{4,5})'+sigma(4)*eye(Nr(4)*L);

        Q5 = kron(eye(L),G{5,1})*s11*kron(eye(L),G{5,1})'+...
            kron(eye(L),G{5,2})*s22*kron(eye(L),G{5,2})'+...
            kron(eye(L),G{5,3})*s33*kron(eye(L),G{5,3})'+...
            kron(eye(L),G{5,4})*s44*kron(eye(L),G{5,4})'+sigma(5)*eye(Nr(5)*L);


        v1 = kron(eye(L),A{1})*s1;
        v2 = kron(eye(L),A{2})*s2;
        v3 = kron(eye(L),A{3})*s3;
        v4 = kron(eye(L),A{4})*s4;
        v5 = kron(eye(L),A{5})*s5;

        minimize 1/(2*abs(alpha(1))^2)*inv_pos(t1)+...
            1/(2*abs(alpha(2))^2)*inv_pos(t2)+...
            1/(2*abs(alpha(3))^2)*inv_pos(t3)+...
            1/(2*abs(alpha(4))^2)*inv_pos(t4)+...
            1/(2*abs(alpha(5))^2)*inv_pos(t5)
        subject to
        (Y{1}'*v1+v1'*Y{1})-quad_form(Y{1}, Q1)>=t1   
        (Y{2}'*v2+v2'*Y{2})-quad_form(Y{2}, Q2)>=t2   
        (Y{3}'*v3+v3'*Y{3})-quad_form(Y{3}, Q3)>=t3   
        (Y{4}'*v4+v4'*Y{4})-quad_form(Y{4}, Q4)>=t4   
        (Y{5}'*v5+v5'*Y{5})-quad_form(Y{5}, Q5)>=t5   

        norm(s1)<=sqrt(Pt(1));
        norm(s2)<=sqrt(Pt(2));
        norm(s3)<=sqrt(Pt(3));
        norm(s4)<=sqrt(Pt(4));
        norm(s5)<=sqrt(Pt(5));
        
        [s11, s1;s1', eye(1)] == hermitian_semidefinite(Nt(1)*L+1);
        [s22, s2;s2', eye(1)] == hermitian_semidefinite(Nt(2)*L+1);
        [s33, s3;s3', eye(1)] == hermitian_semidefinite(Nt(3)*L+1);
        [s44, s4;s4', eye(1)] == hermitian_semidefinite(Nt(4)*L+1);
        [s55, s5;s5', eye(1)] == hermitian_semidefinite(Nt(5)*L+1);
       
        cvx_end        
        s{1} = s1; s{2} = s2; s{3} = s3; s{4} = s4; s{5} = s5; 
    end
    inter_result(Max_inter+1) = sum(Compute_CRB(M,A,s,G,L,alpha,sigma,Nr));
    test_results(:,test) = inter_result;
end




