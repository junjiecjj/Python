function [s_R s_D Target_R Target_D target_num Target_Range_all Target_Doppler_all ] = measure(S_out,S_Sigma_a,Num_Tr_CPI ,Fs,Tp,Fr,Wavelength)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%  Ŀ����롢�����մֲ�   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
c=3e8;
Delta_t=1/Fs;
target_num=zeros(1,length(S_out));

s_R=cell(1,length(S_out));%Ŀ����ʾ��ʵ
s_D=cell(1,length(S_out));%Ŀ����ʾ��ʵ

Target_Range_all=cell(1,length(S_out));%Ŀ��ľ����
Target_Doppler_all=cell(1,length(S_out));%Ŀ��Ķ����պ�

Target_D=cell(1,length(S_out));%��ȷ������������Ϣ
Target_R=cell(1,length(S_out));%��ȷ����������Ϣ



No_dopper_channel_set=1:Num_Tr_CPI;
% ����CFAR���Ķ�����ͨ�����
Num_ReservedCell=3;
% ���ⵥԪ�����ı�����Ԫ���ȣ�ǰ���
Num_TestWin=50;
% ���ⵥԪ������ͳ���Ӳ��������ʵĴ��ڳ��ȣ�ǰ���
N1=Num_TestWin+Num_ReservedCell;

for n_prf=1:length(S_out)

  S_output=S_out{1,n_prf};
  Num_sample=size(S_output,1);
No_target=0;%��⵽�Ĵ���Ŀ��ĵ�Ԫ����
Target_Doppler_No=[];%����Ŀ�������ͨ����
Target_Range_No=[];%����Ŀ��ľ����
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
            % �о��Ƿ�Ϊһ����Ŀ�������
            if condition_1 && condition_2 && condition_3 
            No_target=No_target+1;
            Target_Doppler_No(No_target)=No_dopper_channel;
            Target_Range_No(No_target)=No_tr+N1;%��Ϊ��CFARʱʵ�ʵ���洢�����Σ���λ��
            %��¼ÿ��Ŀ��ľ�����źͶ��������
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%     Ŀ��������      %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if No_target~=0
target_num(1,n_prf)=length(Target_Doppler_No);%��⵽��Ŀ����Ŀ
Target_Range_all{1,n_prf}=Target_Range_No;
Target_Doppler_all{1,n_prf}=Target_Doppler_No;

s_Sigma_a=S_Sigma_a{1,n_prf};
s_R{1,n_prf}=zeros(size(s_Sigma_a));
s_D{1,n_prf}=zeros(size(s_Sigma_a));

for No_target=1:length(Target_Range_No)
    s=ifft(fft(s_Sigma_a(:,Target_Doppler_No(No_target))),10*Num_sample);
    % �������ֵϸ��
    
%     if Target_Doppler_No(No_target)<=5
%         [vmax,pmax]=max(abs(s(1:(10*Target_Range_No(No_target)+49))));
%     Target_Range(No_target)=(10*Target_Range_No(No_target)+pmax-1)/(Fs*10)*c/2-(Tp-Delta_t)/2*c/2;%%ȥ����̬�� 
%     % ����ϸ����ľ������ֵ�����Ŀ�����
%     elseif (10*Target_Range_No(No_target)+49)>length(s)
%         [vmax,pmax]=max(abs(s(10*Target_Range_No(No_target)-49):length(s)));
%     Target_Range(No_target)=((10*Target_Range_No(No_target)-49)+pmax-1)/(Fs*10)*c/2-(Tp-Delta_t)/2*c/2;%%ȥ����̬��  
%     else
%         
     [vmax,pmax]=max(abs(s((10*Target_Range_No(No_target)-39):(10*Target_Range_No(No_target)+39))));
    Target_Range(No_target)=((10*Target_Range_No(No_target)-39)+pmax-1)/(Fs*10)*c/2-(Tp-Delta_t)/2*c/2;%%ȥ����̬�� 
    % ����ϸ����ľ������ֵ�����Ŀ�����
  %  end
    s_R{1,n_prf}(Target_Range_No(No_target),Target_Doppler_No(No_target))= Target_Range(No_target);
    %��Ӧ��Ŀ��ľ���źͶ����պŵĵ�Ԫ���Ŀ��ľ���
end
% disp(Target_Range/1e3)
Target_R{1,n_prf}=Target_Range;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%     Ŀ���ٶȲ���      %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for No_target=1:length(Target_Range_No)
    s=fftshift(fft(ifft(ifftshift(s_Sigma_a(Target_Range_No(No_target),:))),10*Num_Tr_CPI));
    % ���������ֵϸ��
    if Target_Doppler_No(No_target)<=10
        [vmax,pmax]=max(abs(s(1:(10*Target_Doppler_No(No_target)+100))));
        Target_Doppler(No_target)=-Fr(n_prf)/2+(10*Target_Doppler_No(No_target)+pmax-1)*(Fr(n_prf)/(10*Num_Tr_CPI)); 
    elseif Target_Doppler_No(No_target)>=54
        [vmax,pmax]=max(abs(s((10*Target_Doppler_No(No_target)-100):length(s))));
        Target_Doppler(No_target)=-Fr(n_prf)/2+(10*Target_Doppler_No(No_target)+pmax-1)*(Fr(n_prf)/(10*Num_Tr_CPI)); 
    else    
    [vmax,pmax]=max(abs(s((10*Target_Doppler_No(No_target)-100):(10*Target_Doppler_No(No_target)+100))));
    Target_Doppler(No_target)=-Fr(n_prf)/2+(10*Target_Doppler_No(No_target)-100+pmax-1)*(Fr(n_prf)/(10*Num_Tr_CPI));
    % ����ϸ����Ķ�����Ƶ�׷�ֵ�����Ŀ�����
    end
    s_D{1,n_prf}(Target_Range_No(No_target),Target_Doppler_No(No_target))=Target_Doppler(No_target)*Wavelength/2;
end
% disp(Target_Doppler*Wavelength/2)
Target_D{1,n_prf}=Target_Doppler*Wavelength/2;
end

end

end

