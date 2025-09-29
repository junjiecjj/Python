function [ S_out] = CFAR(S_Sigma_a,Num_Tr_CPI )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%   CFAR（恒虚警检测）  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S_out=cell(1,3);
for n_prf=1:length(S_Sigma_a)
 

s_Sigma_a=S_Sigma_a{1,n_prf};
S_output=zeros(size(s_Sigma_a));
Num_sample=size(s_Sigma_a,1);%存储维数。
Pfa=1e-6;
% 虚警概率
No_dopper_channel_set=1:Num_Tr_CPI;
% 进行CFAR检测的多普勒通道序号
Num_ReservedCell=3;
% 待测单元附近的保护单元长度（前或后）
Num_TestWin=50;
% 待测单元附近的统计杂波噪声功率的窗口长度（前或后）
N1=Num_TestWin+Num_ReservedCell;
m=1;
for No_dopper_channel=No_dopper_channel_set
    n=1;
    for No_tr=1+N1:Num_sample-N1%从保护单元后的单元开始检测，最后53个单元也不进行检测。
        Tranning_set=[(No_tr-N1:No_tr-Num_ReservedCell-1),...
            (No_tr+N1:No_tr+Num_ReservedCell+1)];
        % 训练数据序号
        Power_noise_clutter=mean(abs(s_Sigma_a(Tranning_set,No_dopper_channel)).^2);
        % 统计杂波噪声功率
        Threshold=sqrt(2*Power_noise_clutter*log(1/Pfa));
        % 检测门限
        if abs(s_Sigma_a(No_tr,No_dopper_channel))>=Threshold
            % 门限判决
            S_output(n,m)=abs(s_Sigma_a(n,m));
        else
            S_output(n,m)=0;
        end
        n=n+1;
    end
    m=m+1;
end
S_out{1,n_prf}=S_output;%S_output与S_out{1,n_prf}维数相同都是Num_sample*64
end

end

