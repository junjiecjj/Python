function [s_R s_D Target_R Target_D target_num Target_Range_all Target_Doppler_all ] = measure(S_out,S_Sigma_a,Num_Tr_CPI ,Fs,Tp,Fr,Wavelength)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  目标距离、多普勒粗测   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c=3e8;
Delta_t=1/Fs;
target_num=zeros(1,length(S_out));

s_R=cell(1,length(S_out));%目标显示现实
s_D=cell(1,length(S_out));%目标显示现实

Target_Range_all=cell(1,length(S_out));%目标的距离号
Target_Doppler_all=cell(1,length(S_out));%目标的多普勒号

Target_D=cell(1,length(S_out));%精确测量多普勒信息
Target_R=cell(1,length(S_out));%精确测量距离信息



No_dopper_channel_set=1:Num_Tr_CPI;
% 进行CFAR检测的多普勒通道序号
Num_ReservedCell=3;
% 待测单元附近的保护单元长度（前或后）
Num_TestWin=50;
% 待测单元附近的统计杂波噪声功率的窗口长度（前或后）
N1=Num_TestWin+Num_ReservedCell;

for n_prf=1:length(S_out)

  S_output=S_out{1,n_prf};
  Num_sample=size(S_output,1);
No_target=0;%检测到的存在目标的单元数。
Target_Doppler_No=[];%存在目标多普勒通道号
Target_Range_No=[];%存在目标的距离号
for No_dopper_channel=No_dopper_channel_set
    if sum( S_output(:,No_dopper_channel))>0
        for No_tr=4:Num_sample-2*N1-3
            condition_1=S_output(No_tr,No_dopper_channel)~=0;
            condition_2=sum(S_output(No_tr-3:No_tr-1,No_dopper_channel))==0;
            if No_dopper_channel==1
                condition_3=1;
            else
            condition_3=sum(S_output(No_tr-3:No_tr+3,No_dopper_channel-1))==0;
            end
            % 判决是否为一个新目标的条件
            if condition_1 && condition_2 && condition_3 
            No_target=No_target+1;
            Target_Doppler_No(No_target)=No_dopper_channel;
            Target_Range_No(No_target)=No_tr+N1;%因为在CFAR时实际的与存储的相差Ｎ１个位置
            %记录每个目标的距离序号和多普勒序号
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%     目标距离测量      %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if No_target~=0
target_num(1,n_prf)=length(Target_Doppler_No);%检测到的目标数目
Target_Range_all{1,n_prf}=Target_Range_No;
Target_Doppler_all{1,n_prf}=Target_Doppler_No;

s_Sigma_a=S_Sigma_a{1,n_prf};
s_R{1,n_prf}=zeros(size(s_Sigma_a));
s_D{1,n_prf}=zeros(size(s_Sigma_a));

for No_target=1:length(Target_Range_No)
    s=ifft(fft(s_Sigma_a(:,Target_Doppler_No(No_target))),10*Num_sample);
    % 距离向插值细化
    
%     if Target_Doppler_No(No_target)<=5
%         [vmax,pmax]=max(abs(s(1:(10*Target_Range_No(No_target)+49))));
%     Target_Range(No_target)=(10*Target_Range_No(No_target)+pmax-1)/(Fs*10)*c/2-(Tp-Delta_t)/2*c/2;%%去掉暂态点 
%     % 利用细化后的距离向峰值点测量目标距离
%     elseif (10*Target_Range_No(No_target)+49)>length(s)
%         [vmax,pmax]=max(abs(s(10*Target_Range_No(No_target)-49):length(s)));
%     Target_Range(No_target)=((10*Target_Range_No(No_target)-49)+pmax-1)/(Fs*10)*c/2-(Tp-Delta_t)/2*c/2;%%去掉暂态点  
%     else
%         
     [vmax,pmax]=max(abs(s((10*Target_Range_No(No_target)-39):(10*Target_Range_No(No_target)+39))));
    Target_Range(No_target)=((10*Target_Range_No(No_target)-39)+pmax-1)/(Fs*10)*c/2-(Tp-Delta_t)/2*c/2;%%去掉暂态点 
    % 利用细化后的距离向峰值点测量目标距离
  %  end
    s_R{1,n_prf}(Target_Range_No(No_target),Target_Doppler_No(No_target))= Target_Range(No_target);
    %对应的目标的距离号和多普勒号的单元存放目标的距离
end
% disp(Target_Range/1e3)
Target_R{1,n_prf}=Target_Range;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%     目标速度测量      %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for No_target=1:length(Target_Range_No)
    s=fftshift(fft(ifft(ifftshift(s_Sigma_a(Target_Range_No(No_target),:))),10*Num_Tr_CPI));
    % 多普勒域插值细化
    if Target_Doppler_No(No_target)<=10
        [vmax,pmax]=max(abs(s(1:(10*Target_Doppler_No(No_target)+100))));
        Target_Doppler(No_target)=-Fr(n_prf)/2+(10*Target_Doppler_No(No_target)+pmax-1)*(Fr(n_prf)/(10*Num_Tr_CPI)); 
    elseif Target_Doppler_No(No_target)>=54
        [vmax,pmax]=max(abs(s((10*Target_Doppler_No(No_target)-100):length(s))));
        Target_Doppler(No_target)=-Fr(n_prf)/2+(10*Target_Doppler_No(No_target)+pmax-1)*(Fr(n_prf)/(10*Num_Tr_CPI)); 
    else    
    [vmax,pmax]=max(abs(s((10*Target_Doppler_No(No_target)-100):(10*Target_Doppler_No(No_target)+100))));
    Target_Doppler(No_target)=-Fr(n_prf)/2+(10*Target_Doppler_No(No_target)-100+pmax-1)*(Fr(n_prf)/(10*Num_Tr_CPI));
    % 利用细化后的多普勒频谱峰值点测量目标距离
    end
    s_D{1,n_prf}(Target_Range_No(No_target),Target_Doppler_No(No_target))=Target_Doppler(No_target)*Wavelength/2;
end
% disp(Target_Doppler*Wavelength/2)
Target_D{1,n_prf}=Target_Doppler*Wavelength/2;
end

end

end

