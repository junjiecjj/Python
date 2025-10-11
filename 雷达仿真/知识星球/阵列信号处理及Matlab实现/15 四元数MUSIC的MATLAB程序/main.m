%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%**�������֣���Ԫ��MUSIC���������������г������
%**���ߣ�    ����
%**         {(1+i(rou)exp[j(fai)])*exp[-j(thita)]}*beita*exp[j(alfa)]    
%**         �˴�����beita=1��alfa=0
% EMAIL:wangxiaoxian@nuaa.edu.cn, zhangxiaofei@nuaa.edu.cn
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all
clc;

K = 0;
for Kk = 1:1
    
N = 3;
% �����ź�
thita_1 = 0.85;
rou_1 = 3;
fai_1 = 0.27;
beita_1 = 1;
alfa_1 = 0;
SourSig = SteerVector(N, thita_1, rou_1, fai_1, beita_1, alfa_1);
% ��������
Noise = normrnd(0,0.01,N,1);
SourSig(:,1) = SourSig(:,1) + Noise;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ���ź�ʸ���Ĺ���
Cj_SourSig = ConjQVector(SourSig);
% ��������Ϊ������
R_SourSig = ColToRow(Cj_SourSig);

% ��Ԫ��������ʸ����,���źŵ���ؾ���
CorR = VectorMulti(SourSig, R_SourSig);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ��ÿһ����Ԫ��������ʾ
S_C_Q = IsoMatrix(CorR);
% ����ʾ��ľ���������ʹ֮��Ϊ��Ԫ������ĵ�����
A = AssignAgain(S_C_Q);

[V,D] = eig(A);
% �ҵ�D��С��ֵ����Ӧ��V
c = diag(D);
d = min(c);
e = find(c==d );

% ��������������������ת�����Ԫ����������
QVector = FVeToQVe(V);
% ȡ��Ӧ��СDֵ����Ԫ������
MinQV = QVector(:,(e-1)*4+1:e*4);

% �����źŲ���
% ���ȵõ� Uk*Uk'
R_MinQV = ColToRow(MinQV);
NoiseCorR = VectorMulti(MinQV, R_MinQV);

% ����������������
x = 0.01;
y = 0.5;
p = 0;

% �������������һֱ����ȷ��������Ȼ�����ѵ�������ԭ��
% ����thita
for i = -1.5:x:1.5
    p = p+1; r = 0;
 
    % ����rou ��ʱ��������
    j = rou_1;
    %for j = 2.6:y:3.4
    %    r = r+1; s = 0;
    
    % ����fai
    k = fai_1;
    %    for k = -0.1:y:0.8
    %        s = s+1;
    
            beita = beita_1;
            alfa = alfa_1;
            % ������
            SourSig = SteerVector(N, i, j, k, beita, alfa);
            Cj_SourSig = ConjQVector(SourSig);
            % ������
            R_SourSig = ColToRow(Cj_SourSig);
            % ��������0�����һ������
            [a,b] = size(SourSig);
            c = zeros(a,(a-1)*b);
            SourSig = [SourSig,c];
             % ��������0�����һ������
            [a,b] = size(R_SourSig);
            c = zeros(b/4-1,b);
            R_SourSig = [R_SourSig;c];
            
            % ��ʼ����
            E = QuatMuti(R_SourSig, NoiseCorR);
            F = QuatMuti(E,SourSig);
            % �������(0��0)������Ԫ��ֵ
            G = F(1,1:4);
            H(p) = 1/sum(G.^2);
            
            % ������������ʱ�õ�
            %H(p,r,s) = 1/sum(G.^2);
   %   end
   %end
end

% ��Kk��ʵ�������
K = K + H;
end
H = K/Kk;
figure(1)
plot(-1.5:x:1.5,H);







    
    

            
            
            
            
            
            
            


















