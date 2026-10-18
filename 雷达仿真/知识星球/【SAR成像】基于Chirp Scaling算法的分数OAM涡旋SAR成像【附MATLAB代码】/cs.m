%% 2024 CS algorithm  2024-liuyu-06-29
%%================================================================
clear;clc;close all;
%%================================================================   
C=3e8;Fc=4e10;lambda=C/Fc; %Impact Factor1                                
S=2*pi/lambda;a=120*lambda; 
disp(a)
H=10000;
Xmin=-50;Xmax=50;
beta=30*pi/180; %Impact Factor2
Yc=H*tan(beta);Y0=100;
V=300;R0=sqrt(Yc^2+H^2);
Lsar=200;Tsar=Lsar/V;                                     
%%========================azimuth========================================================================
Ka=-2*V^2/lambda/R0;
Ba=abs(Ka*Tsar);PRF=2*Ba;
PRT=1/PRF;ds=PRT;   
Nslow=ceil((Xmax-Xmin+Lsar)/V/ds);Nslow=2^nextpow2(Nslow); %for fft
sn=linspace((Xmin-Lsar/2)/V,(Xmax+Lsar/2)/V,Nslow);
PRT=(Xmax-Xmin+Lsar)/V/Nslow;PRF=1/PRT;ds=PRT;                                  
%%========================range========================================================================
Tr=2e-6;Br=3e8;Kr=Br/Tr;                                
Fsr=2*Br;dt=1/Fsr;
Rmin=sqrt((Yc-Y0)^2+H^2);Rmax=sqrt((Yc+Y0)^2+H^2+(Lsar/2)^2);                                  
Nfast=ceil(2*(Rmax-Rmin)/C/dt+Tr/dt); Nfast=2^nextpow2(Nfast);                
tm=linspace(2*Rmin/C-Tr/2,2*Rmax/C+Tr/2,Nfast);dt=(2*Rmax/C+Tr-2*Rmin/C)/Nfast; 
Fsr=1/dt; 
N=Nslow;M=Nfast;
%%========================Point target========================                      
Ntarget=20;
Ptarget=...
[
12.75 ,Yc-12.5
12.5,Yc-10
12.25,Yc-7.5
11.75,Yc-7.5
11.5,Yc-10
11.25,Yc-12.5
13.2,Yc-3
12.9 ,Yc-1
12.6 ,Yc+1
12.3,Yc+3
11.7 ,Yc+3
11.4,Yc+1
11.1,Yc-1
10.8,Yc-3
11.75 ,Yc-12.5
12.25,Yc-12.5
12,Yc-5
12 ,Yc
12 ,Yc-10
12 ,Yc+5
];
K=Ntarget;T=Ptarget; Srnm=zeros(N,M);
%%========================Parameter setting========================
f=linspace(-PRF/2,+PRF/2,Nslow);fr=linspace(-Fsr/2,+Fsr/2,Nfast); 
r_ref=sqrt(Yc^2+H^2);r=tm/2*C;r_sub=ones(N,1)*(r-r_ref).^2;
CS_f=1./sqrt(1-(lambda*f/2/V).^2)-1;
R_ref=r_ref*(1+CS_f);
Ks=Kr./(1+Kr*r_ref.*(2*lambda/C^2).*((lambda*f/2/V).^2)./(1-(lambda*f/2/V).^2).^1.5); 
phase_cor=(4*pi/C^2*Ks.*(1+CS_f).*CS_f)'*ones(1,M).*r_sub;
%%========================echo========================================================================
for k=1:K
sigma=1;                          
Dslow=T(k,1)-sn*V;
R=sqrt(Dslow.^2+T(k,2)^2+H^2);  
tau=2*R/C;                              
phi=asin(Yc./sqrt(R.^2-H^2));
theta=acos(H./R);   
Dfast=ones(N,1)*tm-tau'*ones(1,M);
    ww=6;
    if floor(ww)==ww
    l=ww;
    phase=pi*Kr*Dfast.^2-(4*pi/lambda)*(R'*ones(1,M))+l*(phi'*ones(1,M));
    Srnm=Srnm+sigma.*exp(1i*phase).*(-Tr/2<Dfast&Dfast<Tr/2).*((abs(Dslow)<Lsar/2)'*ones(1,M)).*besselj(l,a*S.*sin(theta'*ones(1,M)))*exp(1i*l*pi/2);% 回波信号
    Srnm_xfft=fftshift(fft(fftshift(Srnm))); % 1    for azimuth fft
    CS_phase=exp(1i*pi.*(Ks.*CS_f)'*ones(1,M).*(ones(N,1)*tm-(2*R_ref/C)'*ones(1,M) ).^2 ); % for CS
    Srnm_cs=Srnm_xfft.*CS_phase; % for H1
    Srnm_yfft=fftshift(fft(fftshift(Srnm_cs.'))).'; % 2     for range fft
    rmc_phase=exp(1i*pi*(ones(N,1)*fr.^2)./((Ks.*(1+CS_f))'*ones(1,M))).*exp( 1i*4*pi/C*(ones(N,1)*fr)*r_ref.*(CS_f'*ones(1,M)) );% for seek rcmc
    Srnm_rmc=Srnm_yfft.*rmc_phase; % H2
    Srnm_yifft=fftshift(ifft(fftshift(Srnm_rmc.'))).'; % 1      for range ifft
    Srnm_xphase=exp(-1i*2*pi*C/lambda.*ones(N,1)*tm.*(1-sqrt(1-((lambda*f/2/V).^2)'*ones(1,M)))+1i*phase_cor);
    Srnm_cor=Srnm_yifft.*Srnm_xphase; % Phase compensation
    f_xy=fftshift(ifft(fftshift(Srnm_cor))); % 2        for azimuth ifft
    fxy=f_xy./besselj(l,a*S.*sin(theta'*ones(1,M))).*exp(-1i*l*pi/2);
    Ga=abs(fxy);
    else
    phase=pi*Kr*Dfast.^2-(4*pi/lambda)*(R'*ones(1,M))+ww*(phi'*ones(1,M));
    Srnm=Srnm+sigma.*exp(1i*phase).*(-Tr/2<Dfast&Dfast<Tr/2).*((abs(Dslow)<Lsar/2)'*ones(1,M));% 回波信号
    Srnm_xfft=fftshift(fft(fftshift(Srnm))); % 1    for azimuth fft
    CS_phase=exp(1i*pi.*(Ks.*CS_f)'*ones(1,M).*(ones(N,1)*tm-(2*R_ref/C)'*ones(1,M) ).^2 ); % for CS
    Srnm_cs=Srnm_xfft.*CS_phase; % for H1
    Srnm_yfft=fftshift(fft(fftshift(Srnm_cs.'))).'; % 2     for range fft
    rmc_phase=exp(1i*pi*(ones(N,1)*fr.^2)./((Ks.*(1+CS_f))'*ones(1,M))).*exp( 1i*4*pi/C*(ones(N,1)*fr)*r_ref.*(CS_f'*ones(1,M)) );% for seek rcmc
    Srnm_rmc=Srnm_yfft.*rmc_phase; % H2
    Srnm_yifft=fftshift(ifft(fftshift(Srnm_rmc.'))).'; % 1      for range ifft
    Srnm_xphase=exp(-1i*2*pi*C/lambda.*ones(N,1)*tm.*(1-sqrt(1-((lambda*f/2/V).^2)'*ones(1,M)))+1i*phase_cor);
    Srnm_cor=Srnm_yifft.*Srnm_xphase; % Phase compensation
    f_xy=fftshift(ifft(fftshift(Srnm_cor))); % 2        for azimuth ifft
    Ga=abs(f_xy);
    end 
end   
figure(1)
imagesc(Ga);colormap jet;axis tight;colorbar
xlabel('Range sampling point'),ylabel('Azimuth sampling point'),title('Point target')
colorbar;
%%========================echo========================================================================
echo=Ga;
%%========================Two-dimensional interpolation========================
Max=max(max(echo));
[X,Y]=find(echo==Max); %Peak coordinate
DArea=echo(X-64:X+64,Y-64:Y+64); 
E1=fftshift(fft(DArea),1);E2=fftshift(fft(E1.'),1);
D2=4; %D2 times interpolation
A3=[zeros(129,floor(129*(D2-1)/2)) E2 zeros(129,(D2-1)*129-floor(129*(D2-1)/2))];
A4=[zeros(D2*129,floor(129*(D2-1)/2)) A3.' zeros(D2*129,(D2-1)*129-floor(129*(D2-1)/2))];
A4=A4.';
A5=ifft(fftshift(A4,1));
A6=ifft(fftshift(A5.',1));
A7=abs(A6);
%%
figure(3)
imagesc(A7);colormap jet;axis tight;colorbar
xlabel('Range sampling point'),ylabel('Azimuth sampling point')
grid on
%%
figure(4);contour(A7,30);colormap jet;axis tight;
xlabel('Range sampling point'),ylabel('Azimuth sampling point'),title('Two-dimensional contour map')
grid on
%axis([0 400 100 500]);
%axis([0 400 100 500]);xlim([0 400]);ylim([100 500]);xticks(0:50:400);yticks(100:50:500)
%%========================One-dimensional interpolation========================
Max_one=max(max(A7'));[i,j]=find(A7'==Max_one); % Maximum row
A8=A6';
A=A8(i-50:i+50,j);
B=A8(i,j-50:j+50);
A1=fftshift(fft(A.'));
B1=fftshift(fft(B));
D1=4; % times interpolation
R1=[zeros(1,floor(length(A1)*(D1-1)/2)) A1 zeros(1,floor(length(A1)*(D1-1)/2))];
B2=[zeros(1,floor(length(B1)*(D1-1)/2)) B1 zeros(1,floor(length(B1)*(D1-1)/2))];
R2=abs(ifft(fftshift(R1)));
A2=abs(ifft(fftshift(B2)));
r2=20*log10(R2/max(R2)); % Range normalization
a2=20*log10(A2/max(A2)); %Azimuth normalization

figure(5)
plot(r2,'--','LineWidth', 1.5);
xlabel('Range sampling point','FontName','Times New Roman','FontSize',15,'color','k')
ylabel('Amplitude(dB)','FontName','Times New Roman','FontSize',15,'color','k')
colormap jet;axis tight;colorbar;grid on
figure(6)
plot(a2,'-');
xlabel('Azimuth sampling point','FontName','Times New Roman','FontSize',14,'color','k')
ylabel('Amplitude(dB)','FontName','Times New Roman','FontSize',14,'color','k')
colormap jet;axis tight;grid on
hold on 
%legend('J_0','J_1','J_2','J_3','J_4','Location','Best')
%xlim([0 800]);ylim([-30 0]);xticks(0:50:800);yticks(-30:2:0);grid on

%%========================Range resolution========================
[max_r,POS_r]=max(r2);
r3=POS_r;
%% 3DB to the left
while(r2(r3)>-3)  
    r3=r3-1; 
end
rWidth_3db1=r3;
%% Find the first zero on the left
for r4=rWidth_3db1:-1:1
    if R2(r4)<R2(r4-1)
    break
    end
end
zero_left=r4;
%% Find the left side lobe value
for r5=zero_left-1:-1:1
    if R2(r5)>R2(r5-1)
   break
    end
end
%% Left sidelobe energy
r_Sum_sidelobe1=0; 
    for rs1=1:zero_left-1   
        r_Sum_sidelobe1 = r_Sum_sidelobe1 +R2(rs1) * R2(rs1);
    end
%% 3DB to the right
r3=POS_r+1;
    while(r2(r3)>-3)
        r3=r3+1;
    end
rWidth_3db2=r3;
%% Find the first zero on the right
for r6=rWidth_3db2:length(r2)
    if R2(r6)<R2(r6+1)
   break
    end
end
zero_right=r6;
%% Find the right side lobe value
for r7=zero_right+1:length(r2)
    if R2(r7)>R2(r7+1)
    break
    end
end
%% Main-lobe energy
r_Sum_mainlobe=0;
for r8=zero_left:zero_right   
  r_Sum_mainlobe= r_Sum_mainlobe+R2(r8)*R2(r8);
end
%% right sidelobe energy
r_Sum_sidelobe =r_Sum_sidelobe1;
for rs2=zero_right+1:length(r2)
   r_Sum_sidelobe=r_Sum_sidelobe +R2(rs2)*R2(rs2);
end
%% Find the larger value of the first side lobe
rMax_sidelobe=max(R2(r5),R2(r7));
rMax_mainlobe=max(R2);
%%========================Range index calculation========================
r_3db=rWidth_3db2-rWidth_3db1+2;
% integral sidelobe ratio
ISLRr=10*log10(r_Sum_sidelobe/r_Sum_mainlobe);
% Peak sidelobe ratio
PSLRr=20*log10(rMax_sidelobe/rMax_mainlobe);  
% resolution
fenbianlv_r=C*r_3db/(2*Fc*D1*D2);
%%================================================================================================
%%========================Azimuth resolution========================
[max_a,POS_a]=max(a2);
a3=POS_a;
a_Sum_sidelobe1=0; 
%% 3DB to the left
while(a2(a3)>-3)  
    a3=a3-1;
end
aWidth_3db1=a3;
%% Find the first zero on the left
for a4=aWidth_3db1:-1:1
    if A2(a4)<A2(a4-1)
   break
    end
end
a_zero_left=a4;
%% Find the left side lobe value
for a5=a_zero_left-1:-1:1
    if A2(a5)>A2(a5-1)
    break
    end
end
%% Left sidelobe energy
for as1=1:a_zero_left-1   
a_Sum_sidelobe1 = a_Sum_sidelobe1 +A2(as1) * A2(as1);%%%左边旁瓣求和
end
%% 3DB to the right
a3=POS_a+1;
while(a2(a3)>-3)
    a3=a3+1;
end
aWidth_3db2=a3;
%% Find the first zero on the right
for a6=aWidth_3db2:length(a2)
    if A2(a6)<A2(a6+1)
    break
    end
end
a_zero_right=a6;
%% Find the right side lobe value
for a7=a_zero_right+1:length(a2)
    if A2(a7)>A2(a7+1)
    break
    end
end
%% Main-lobe energy
a_Sum_mainlobe=0;
for a8=a_zero_left:a_zero_right    
  a_Sum_mainlobe=a_Sum_mainlobe+A2(a8)*A2(a8);
end
%% right sidelobe energy
a_Sum_sidelobe =a_Sum_sidelobe1 ;
for as2=a_zero_right+1:length(a2)
a_Sum_sidelobe=a_Sum_sidelobe +A2(as2)*A2(as2);
end
%% Find the larger value of the first side lobe
aMax_sidelobe=max(A2(a5),A2(a7));
aMax_mainlobe=max(A2);
%%========================Azimuth index calculation========================
a_3db=aWidth_3db2-aWidth_3db1+2;%3db 带宽
% integral sidelobe ratio
ISLRa=10*log10(a_Sum_sidelobe/a_Sum_mainlobe);
% Peak sidelobe ratio
PSLRa=20*log10(aMax_sidelobe/aMax_mainlobe);
% resolution
fenbianlv_a=V*a_3db/(PRF*D1*D2);

% disp('Range resolution='),disp(fenbianlv_r)     
% disp('Range integral sidelobe ratio='),disp(ISLRr)     
% disp('Range Peak sidelobe ratio='),disp(PSLRr) 
disp('Azimuth resolution='),disp(fenbianlv_a) 
disp('Azimuth integral sidelobe ratio='),disp(ISLRa)            
disp('Azimuth Peak sidelobe ratio='),disp(PSLRa) 
