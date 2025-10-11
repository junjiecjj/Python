%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%**程序名字：列信号的共轭
%**作者：    汪飞
%**日期：    2006-6-10
%**修改人：
%**日期：      
%**描述：    仿真Q_MUSIC方法
%**          
%**         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function output = ConjQVector(SourSig)

    Cj_SourSig = SourSig;
    Cj_SourSig(:,2) = SourSig(:,2)*(-1);
    Cj_SourSig(:,3) = SourSig(:,3)*(-1);
    Cj_SourSig(:,4) = SourSig(:,4)*(-1);
    
    output = Cj_SourSig;