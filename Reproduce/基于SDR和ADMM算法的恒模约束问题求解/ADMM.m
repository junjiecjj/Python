
clc;
clear all;
close all;



% s = rng(0);
N = 10;
A = rand(N) +1i*rand(N);
A=A*A';
A = (A +A')/2;

R = rand(N) +1i*rand(N);
R=R*R';
R = (R +R')/2;

iter_num = 100;
w0 =exp(1i*2*pi*rand(N,1));
phi = rand(N,1)*2*pi;
u = rand(N,1);
rho = 5;
val = zeros(iter_num,1);
val_pri = zeros(iter_num,1);
epsilon = 1e-4;
for iter = 1:iter_num
    
    %update w
    w = rho.*inv((2*A+rho.*eye(N)))*(phi-u);
    % cvx_begin quiet
    %     variable w(N,1) complex
    %     minimize (real(w'*A*w)+rho/2*square_pos(norm(w-exp(1i*phi)+u,2)))
    %     subject to 
    %         real(w0'*R*w0+w0'*R*(w-w0))>=1;
    %         norm(w,2)<=1;
    % cvx_end
    % 
    %update v
    phi =  angle(w+u);
    
    u = u + w - exp(1i*phi);
    
    
    val(iter) = real(w'*A*w)+rho/2*norm(w- exp(1i*phi)+u,2)^2-rho/2*norm(u,2)^2;
    val_pri(iter) = real(w'*A*w);
    w0 = w;
    
    if norm(w - exp(1i*phi),2)<=epsilon
        break
    end
 iter   
end
figure
plot(val(1:iter),'b-','linewidth',1.5);
hold on
plot(val_pri(1:iter),'-','linewidth',1.5);
grid on
legend('lagrangian function','objective function');
xlabel('iteration number');
ylabel('value')
set(gca,'fontsize',12);
ax=gca;
ax.FontName = 'Times New Roman';
abs(w)
