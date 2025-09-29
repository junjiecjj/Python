function [s_A s_B] = target(G,Fc,Fs,Fr,Num_Tr_CPI,Theta0,Wa,BeamWidth,s,R_set,V_set,RCS,Theta_target_set)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%      目标仿真参数     %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Tr=1./Fr;
c=3e8;  % 光速

s_A=cell(1,length(Fr));
s_B=cell(1,length(Fr));


Wavelength=c/Fc;

RCS_Ground_0=10^(-30/10);         % 地面单位面积后向散射截面积【m^2】
   
for n_prf=1:length(Fr)
 s_n=s{1,n_prf};   
 s_all_A=zeros(size(s{1,n_prf}));%所有目标回波
 s_all_B=zeros(size(s{1,n_prf}));%所有目标回波


s_singal_A=zeros(size(s{1,n_prf}));%单个目标回波
s_singal_B=zeros(size(s{1,n_prf}));%单个目标回波

for No_PRI=1:Num_Tr_CPI
    
    Theta_bp=Theta0+Wa*No_PRI*Tr(n_prf);             % 波束主瓣指向【度】
    
        
        
        delay_target=2*(R_set-V_set*No_PRI*Tr(n_prf))/c;
        delay_ambiguity=mod(delay_target,Tr(n_prf));
        % 目标时延【sec】
        RVP=-2*pi*Fc*delay_target;%1*3 double
        % 目标回波视频检波剩余相位
        Gt_A=G*(sinc((Theta_target_set-(Theta_bp-BeamWidth/2))/BeamWidth)).^2;
        Gr_A=G*(sinc((Theta_target_set-(Theta_bp-BeamWidth/2))/BeamWidth)).^2;
        % 波束A在目标方向上的增益
        Gt_B=G*(sinc((Theta_target_set-(Theta_bp+BeamWidth/2))/BeamWidth)).^2;
        Gr_B=G*(sinc((Theta_target_set-(Theta_bp+BeamWidth/2))/BeamWidth)).^2;
        % 波束B在目标方向上的增益
        Lp=1./((4*pi)^2*R_set.^4);
        % 目标处的电磁波传播损耗
        Magnitude_echo_A=sqrt(RCS*(Gt_A+Gt_B)*Gr_A*Wavelength^2/(4*pi).*Lp);%1*3 double
        % 波束A中目标回波幅度
        Magnitude_echo_B=sqrt(RCS*(Gt_A+Gt_B)*Gr_B*Wavelength^2/(4*pi).*Lp);%1*3 double
        % 波束B中目标回波幅度
       
       Echo_delay=round(delay_ambiguity*Fs);  %延迟点数 

       s_singal_A(:,No_PRI)=circshift(s_n(:,No_PRI),[Echo_delay 0])...%循环移位
           *Magnitude_echo_A(n_prf)*exp(j*RVP(n_prf));%加回波幅度和 解波相位项
       %将s_n的各列向下移位Echo_delay，而行方向不变。
       s_singal_B(:,No_PRI)=circshift(s_n(:,No_PRI),[Echo_delay 0])...
           *Magnitude_echo_B(n_prf)*exp(j*RVP(n_prf));%加回波幅度和 解波相位项
        % 将第n个目标的复回波的添加到其在信号数组中的对应位置；
end
s_all_A=s_all_A+s_singal_A;%多个目标叠加
s_all_B=s_all_B+s_singal_B;%多个目标叠加



s_A{1,n_prf}=s_all_A;
s_B{1,n_prf}=s_all_B;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

