%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%**程序名字：四元数MUSIC仿真的主程序用于谐波估计
%**作者：    汪飞
%**         {(1+i(rou)exp[j(fai)])*exp[-j(thita)]}*beita*exp[j(alfa)]    
%**         此处假设beita=1，alfa=0
% EMAIL:wangxiaoxian@nuaa.edu.cn, zhangxiaofei@nuaa.edu.cn
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all
clc;

K = 0;
for Kk = 1:1
    
N = 3;
% 构造信号
thita_1 = 0.85;
rou_1 = 3;
fai_1 = 0.27;
beita_1 = 1;
alfa_1 = 0;
SourSig = SteerVector(N, thita_1, rou_1, fai_1, beita_1, alfa_1);
% 加入噪声
Noise = normrnd(0,0.01,N,1);
SourSig(:,1) = SourSig(:,1) + Noise;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 列信号矢量的共轭
Cj_SourSig = ConjQVector(SourSig);
% 列向量变为横向量
R_SourSig = ColToRow(Cj_SourSig);

% 四元数的两个矢量积,即信号的相关矩阵
CorR = VectorMulti(SourSig, R_SourSig);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 对每一个四元数作复表示
S_C_Q = IsoMatrix(CorR);
% 复表示后的矩阵重排序，使之成为四元数矩阵的导出阵
A = AssignAgain(S_C_Q);

[V,D] = eig(A);
% 找到D最小的值，对应的V
c = diag(D);
d = min(c);
e = find(c==d );

% 将复数的所有特征向量转变成四元数特征向量
QVector = FVeToQVe(V);
% 取对应最小D值的四元数向量
MinQV = QVector(:,(e-1)*4+1:e*4);

% 估计信号参量
% 首先得到 Uk*Uk'
R_MinQV = ColToRow(MinQV);
NoiseCorR = VectorMulti(MinQV, R_MinQV);

% 构造搜索步长因子
x = 0.01;
y = 0.5;
p = 0;

% 在下面的搜索中一直是先确定两个，然后再搜第三个的原则
% 搜索thita
for i = -1.5:x:1.5
    p = p+1; r = 0;
 
    % 搜索rou 暂时不搜索了
    j = rou_1;
    %for j = 2.6:y:3.4
    %    r = r+1; s = 0;
    
    % 搜索fai
    k = fai_1;
    %    for k = -0.1:y:0.8
    %        s = s+1;
    
            beita = beita_1;
            alfa = alfa_1;
            % 列向量
            SourSig = SteerVector(N, i, j, k, beita, alfa);
            Cj_SourSig = ConjQVector(SourSig);
            % 横向量
            R_SourSig = ColToRow(Cj_SourSig);
            % 列向量填0构造成一个方阵
            [a,b] = size(SourSig);
            c = zeros(a,(a-1)*b);
            SourSig = [SourSig,c];
             % 横向量填0构造成一个方阵
            [a,b] = size(R_SourSig);
            c = zeros(b/4-1,b);
            R_SourSig = [R_SourSig;c];
            
            % 开始估计
            E = QuatMuti(R_SourSig, NoiseCorR);
            F = QuatMuti(E,SourSig);
            % 提出矩阵(0，0)处的四元数值
            G = F(1,1:4);
            H(p) = 1/sum(G.^2);
            
            % 搜索三个变量时用的
            %H(p,r,s) = 1/sum(G.^2);
   %   end
   %end
end

% 将Kk次实验结果相加
K = K + H;
end
H = K/Kk;
figure(1)
plot(-1.5:x:1.5,H);







    
    

            
            
            
            
            
            
            


















