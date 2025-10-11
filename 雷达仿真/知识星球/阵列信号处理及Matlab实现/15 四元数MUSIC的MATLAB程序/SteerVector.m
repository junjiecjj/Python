%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%**程序名字：将舵矢量加到极化信息中
%**作者：    汪飞
%**日期：    2006-6-10
%**修改人：
%**日期：      
%**描述：    仿真Q_MUSIC方法
%**         {(1+i(rou)exp[j(fai)])*exp[-j(thita)]}*beita*exp[j(alfa)]    
%**         此处假设beita，afa皆为1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function output = SteerVector(N, thita, rou, fai, beita, alfa)
    
    a = zeros(N,4);
    c = PolarSource(rou, fai);
   for i = 1:N
        b = [cos(-(i-1)*thita),0,sin(-(i-1)*thita),0]*beita;
        a(i,:) = hpc(c,b);
        d = [cos(alfa),0,sin(alfa),0];
% d = [cos((i-1)*alfa),0,sin((i-1)*alfa),0];
        a(i,:) = hpc(a(i,:),d);
  end
    
  output = a;
    
    
    
        