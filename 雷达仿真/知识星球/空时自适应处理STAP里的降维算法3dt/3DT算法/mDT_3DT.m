%��ʱ��ά3dt����


clc;clear all;close all;
load clutter_matrix.mat;

[NK,L]=size(clutter_matrix);
N=16;
K=10;
CNR=60;
Rc=clutter_matrix*clutter_matrix'/L;
noise=max(max(Rc))/(10^(CNR/10))*eye(N*K);
Rx=Rc+noise;
anoise=max(max(noise));                          %��������
Qs=eye(N);                                       %����ά����
psi0=pi/2;

% ȫάSTAP
fd=-1:1/50:1;
inv_Rx=inv(Rx);
for i=1:length(fd);
    Ss=exp(j*pi*(0:N-1)'*cos(psi0));            %Ŀ�귽��ȷ��ʱ��Ss�̶�����Ŀ��dopplerƵ��δ֪����ÿһ��fd��һ������Ȩʸ��wopt
    St=exp(j*pi*(0:K-1)'*fd(i));
    S=kron(St,Ss);
    wopt=inv_Rx*S/(S'*inv_Rx*S);
    IF(i)=abs(wopt'*S)^2*(10^(CNR/10)+1)*anoise/(wopt'*Rx*wopt);
end
figure
plot(fd,10*log10(abs(IF)))
xlabel('2f_d/f_r');ylabel('IF/dB');
grid on


index=0;
for fd=-1:1/50:1
    index=index+1;
    Qt=[exp(j*pi*(0:K-1).'*(fd-1/K)),exp(j*pi*(0:K-1).'*fd),exp(j*pi*(0:K-1).'*(fd+1/K))];%ʱ��ά����
    Q=kron(Qt,Qs);
    Ry=Q'*Rx*Q;
    inv_Ry=inv(Ry);
    Ss=exp(j*pi*(0:N-1).'*cos(psi0));
    St=exp(j*pi*(0:K-1).'*fd);
    g1=(exp(j*pi*(0:K-1).'*(fd-1/K)))'*St/((exp(j*pi*(0:K-1).'*fd))'*St);
    g2=(exp(j*pi*(0:K-1).'*(fd+1/K)))'*St/((exp(j*pi*(0:K-1).'*fd))'*St);
    Sy=[g1*Ss;Ss;g2*Ss];
    %          Sy1=kron(Qt'*St,Qs'*Ss)/K;
    W_3dt=inv_Ry*Sy/(Sy'*inv_Ry*Sy);
    IF_3dt(:,index)=abs(W_3dt'*Sy).^2*(10^(CNR/10)+1)*anoise/(W_3dt'*Ry*W_3dt);
end
hold on
plot(-1:1/50:1,10*log10(IF_3dt),'r');
legend('����','3DT')