function pos = satPos(t,Toc,PRN,para)
bdsPi    = 3.1415926535898;
Omega_e  = 7.2921151467e-5;  % 地球自转角速度, [rad/s]
GM       = 3.986004418e14;   % 地球常数
F        = -4.44280731e-10;  % 相对论效应常数, [sec/(meter)^(1/2)]

if PRN<=5 || PRN >= 59
    isGEO = true;
else
    isGEO = false;
end
a0 = para(1);
a1 = para(2);
a2 = para(3);
AODE = para(4);
Crs = para(5);
DeltaN = para(6);
M0 = para(7);
Cuc = para(8);
e = para(9);
Cus = para(10);
sqrtA = para(11);
Toe = para(12);
Cic = para(13);
Omega0 = para(14);
Cis = para(15);
i0 = para(16);
Crc = para(17);
omega = para(18);
OmegaDot = para(19);
IDOT = para(20);
DataSource = para(21);
Week = para(22);
Adot = para(23);
ACCU = para(24);
SatH = para(25);
TGD1 = para(26);
TGD2 = para(27);
SOW  = para(28);
AODC = para(29);
%% 半长轴
A = sqrtA.*sqrtA; 
% 归一化时间
dTsv = a0 + a1*(t-Toc) + a2*(t-Toc).^2;
t = t - dTsv; 
tk = t - Toe;
% 平均角速度
n0 = sqrt(GM/A.^3);
n  = n0 + DeltaN;
% 平近点角
M = M0 + n*tk;
M = rem(M+2*bdsPi,2*bdsPi);
% 迭代求解偏近点角
E = M;
for ii = 1:10
    E_old   = E;
    E       = M + e * sin(E);
    dE      = rem(E - E_old, 2*bdsPi);
    
    if abs(dE) < 1.e-12
        break;
    end
end
% 相对论效应修正量
dtr = F*e*sqrtA*sin(E);
tk = tk - dtr;
% 计算真近点角
f   = atan2(sqrt(1 - e.^2) * sin(E), cos(E)-e);
%计算升交点角距
phi = f + omega;
%Correct argument of latitude
u = phi           +Cuc*cos(2*phi)+Cus*sin(2*phi);
%Correct radius
r = A*(1-e*cos(E))+Crc*cos(2*phi)+Crs*sin(2*phi);
%Correct inclination
i = i0 + IDOT*tk  +Cic*cos(2*phi)+Cis*sin(2*phi);
if isGEO
    Omega = Omega0 + OmegaDot * tk - Omega_e * Toe;
    Xk = cos(u)*r * cos(Omega) - sin(u)*r * cos(i)*sin(Omega);
    Yk = cos(u)*r * sin(Omega) + sin(u)*r * cos(i)*cos(Omega);
    Zk = sin(u)*r * sin(i);
    Rx = [1,    0    ,   0    ;
          0, cosd(-5),sind(-5);
          0,-sind(-5),cosd(-5)];
    Rz = [cos(Omega_e*tk),sin(Omega_e*tk),0;
         -sin(Omega_e*tk),cos(Omega_e*tk),0;
                  0      ,      0        ,1];
    pos = Rz * Rx *[Xk;Yk;Zk];
    pos = pos.';
else  
    Omega = Omega0 + (OmegaDot - Omega_e)*tk - Omega_e * Toe;
    X = r*cos(u)*cos(Omega) - r*sin(u)*cos(i)*sin(Omega);
    Y = r*cos(u)*sin(Omega) + r*sin(u)*cos(i)*cos(Omega);
    Z = r*sin(u)*sin(i);
    pos = [X,Y,Z];
end

end
