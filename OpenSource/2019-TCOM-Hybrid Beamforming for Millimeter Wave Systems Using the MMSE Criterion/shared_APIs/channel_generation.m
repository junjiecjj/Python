function [H,AT,AR] = channel_generation(N_t,N_r)

%input: the numbers of transmit antennas and receive antennas
%output: the realized channel, codebook for vrf and wrf of OMP method 
%the comments is originally written in Chinese, hopefully I can have time
% to tranfer it to English, but not now.

N_c=5;
N_ray=10;
E_aoa = 2*pi* rand(N_c,1);                               %cluster�ľ�ֵ�����ӣ�0,2*pi���ľ��ȷֲ�
sigma_aoa = 10*pi/180;                                    %�Ƕ���չΪ10�㣬��Ϊ���ȣ�����׼��
b = sigma_aoa/sqrt(2);                                    %���ݱ�׼������Ӧ��b���߶Ȳ���
a = rand(N_c,N_ray)-0.5;                                  %����(-0.5,0.5)�����ھ��ȷֲ����������;
aoa = repmat(E_aoa,1,N_ray)-b*sign(a).*log(1-2*abs(a));   %���ɷ���������˹�ֲ����������(ÿ�д���ÿ��cluster�ĽǶ�)
aoa = sin(aoa);
%-----------AOD
E_aod = 2*pi* rand(N_c, 1);                               %cluster�ľ�ֵ�����ӣ�0,2*pi���ľ��ȷֲ�
sigma_aod = 10*pi/180;                                    %�Ƕ���չΪ10�㣬��Ϊ���ȣ�����׼��
b = sigma_aod/sqrt(2);                                    %���ݱ�׼������Ӧ��b���߶Ȳ���
a = rand(N_c,N_ray)-0.5;                                  %����(-0.5,0.5)�����ھ��ȷֲ����������;
aod = repmat(E_aod,1, N_ray)-b*sign(a).*log(1-2*abs(a));   %���ɷ���������˹�ֲ����������(ÿ�д���ÿ��cluster�ĽǶ�)
aod = sin(aod);


signature_t = [0:(N_t-1)]';
signature_t = 1i*pi* signature_t;                           %Ϊ��������signature������׼��
signature_r = [0:(N_r-1)]';
signature_r = 1i*pi* signature_r;                           %Ϊ��������signature������׼��

H_ray = zeros(N_r, N_t, N_c, N_ray);
H_cl = zeros(N_r, N_t, N_c);


    for i= 1: N_c
        for m = 1: N_ray
            H_ray(:,:,i,m)=complex(randn(1),randn(1))/sqrt(2)*exp((aoa(i,m)*signature_r))*exp((aod(i,m)*signature_t))'/sqrt(N_t*N_r); 
        end
    end  
        H_cl = sum(H_ray, 4);    
    
    H(:,:) = sqrt(N_t*N_r/N_c/N_ray)*sum(H_cl,3);   
    
    aod = aod(:).';
    aoa = aoa(:).';
    A = kron(aod,signature_t);
    AT = 1/sqrt(N_t)*exp(A);
    A = kron(aoa,signature_r);
    AR = 1/sqrt(N_r)*exp(A);
