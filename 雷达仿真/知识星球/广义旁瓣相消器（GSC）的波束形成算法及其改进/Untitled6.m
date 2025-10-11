clear all;close all;

m=8 ;% 阵元
p=4; %  信号数
N=2000;% 迭代次数 或快拍数
A=zeros(m,p); % 阵列流型
theta=[-30 -14 -40 60]*pi/180;% DOA 30为期望信号方向
j=sqrt(-1);
% w=[0.1 0.02 0.3 0.4]; % 各个信号的数字频率
w=[0.0 0.00 0.0 0.0]; % 各个信号的数字频率

s=to_get_s(w,N,p);
s_rec=get_s_rec(s,m,p,theta);
S=s_rec; % m*N 的阵元输出数据矩阵

%%%%——————————————%% 自适应调节权
y=S;% input data of GSC;
ad=exp(-j*pi*[0:m-1]'*sin(theta(1))) % 期望信号方向导向矢量
c=5;% 波束形成条件
C=ad';
Wc=C'*inv(C*C')*c; % 主通道固定权
u=0.01;
wa(1:m-1,1)=0;  % 辅助通道自适应权
Bm=get_B(m,theta); % get Block Matrix 得到阻塞矩阵
B=Bm;
Rxx=get_Rxx(s_rec,N,p,m);
R=Rxx;
wa=inv(B*R*B')*B*R*Wc;
%%%%-----------------------------
wop=Wc-B'*wa;

%%-------------------------画图
drawpp(m,wop);
