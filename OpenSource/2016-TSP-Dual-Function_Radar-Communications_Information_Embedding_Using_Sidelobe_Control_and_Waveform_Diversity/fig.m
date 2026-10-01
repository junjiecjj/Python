%%%coherent phase modulation
%%%beam scan phase modulation

%% main function
clc,clear;
close all;

%% parameters

%--------------radar parameters--------------------
N =10; %number of array elements
c = 3e8; %speed of light
fc = 24e9; %carrier frequency
lambda = c/fc; %wavelength
d = lambda/2; %distance between tow elemets
n = 0:N-1; %array element 
PT=10;%transmitted power
T=1e-3;%pulse width
fs=4e3;             %sampling rate     
ts=1/fs;
M=floor(T/ts);      %sampling number in each pulse
MentNum=1e3;  %MenteCarlo simulation number
N_pulse=161e2;    % pulse number
t_pulse=0:ts:T-ts;        %fast time
N_fast=length(t_pulse);
t=0:ts:N_pulse*T-ts;%





theta = -90:0.5:90; %angle
% thetas=-40:40;
theta_main = [-4,4];%mianlobe (-30:30)degree
n_start = find(theta==theta_main(1));% start angle of mainlobe
n_end = find(theta==theta_main(2)); % end angle of mainlobe
theta1 = theta([1:n_start-10,n_end+10:length(theta)]);%(-30:30)sidelobe
theta0 = theta(n_start:n_end);%(-90:-30)&(30:90) mainlobe
a = exp(-1i*2*pi*d/lambda.*n'.*sind(theta));
a1 = exp(-1i*2*pi*d/lambda.*n'.*sind(theta1));%steering vector of sidelobe
a0 = exp(-1i*2*pi*d/lambda.*n'.*sind(theta0));%steering vector of mainlobe
theta_radar = 0;%direction of radar
theta_com =-50;%direction of communication
theta_scan=-40:0.5:40;%radar beam scan range
a_radar = exp(-1i*2*pi*d/lambda.*n*sind(theta_radar)).';%steering vector of radar direction
a_com = exp(-1i*2*pi*d/lambda.*n*sind(theta_com)).';%steering vector of communication direction
a_t=exp(-1i*2*pi*d/lambda.*n'.*sind(theta_scan));
omega=pi/4+(0:4-1)*pi/2;%communication phase symbols
omega(omega>pi)=(omega(omega>pi)-2*pi);
w=zeros(N,length(omega));
w0=(1:length(omega))/T;%
psi=1/sqrt(N_fast)*exp(-1i*2*pi*t_pulse'*w0);% orthogonal waveform
N_scan=floor(N_pulse/length(theta_scan));%scan time in very angle
% Nbits=1e6;%bits number


SNR=-10:20;%signal to noise ratio

%-----------beampattern design--------------------
for ii=1:length(omega)
    
%-------------cvx slover------------------------------
cvx_begin
    variable u(N,1) complex
    minimize(max(abs(u'*a1)))
    subject to
        u'*a_radar ==1;
%        max(abs(u'*a0)) <=1;
        u'*a_com == 0.1*exp(1i*omega(ii));
%         norm(u'*a0-1,1)<=2;
cvx_end
        u=u/norm(u); % normalization
        w(:,ii)=u; 


end

%-----------plot optimal beampattern-------------
bp=zeros(length(theta),length(omega));
mag=zeros(length(theta),length(omega));
linepattern={'c-','r--','g:','b-.'};
figure
hold on
for ii=1:length(omega)
    

bp(:,ii) = w(:,ii)'*a;
% mag(:,ii) = db(abs(bp(:,ii))/max(abs(bp(:,ii))));
mag(:,ii) = db(abs(bp(:,ii)));
hold on
plot(theta,mag(:,ii),linepattern{ii},'linewidth',1.5)
grid minor



end
xlabel('angle (degree)')
ylabel('amplitude(dB)')
% plot([theta_radar theta_radar],[-50 0],'g--','linewidth',2)
plot([theta_com theta_com ],[-50 10],'b--','linewidth',1.5);
ylim([-50,10])
grid on
legend('beampattern 1','beampattern 2','beampattern 3','beampattern 4','communication direction');
%title('dual funcation radar and communication ')
set(gca,'fontsize',10)
hold off

% %% plot phase
% figure
% phase=zeros(1,length(omega));
% val=zeros(1,length(omega));
% for kk=1:length(omega)
%     val(kk)=w(:,kk)'*a_com*10;
%     phase(kk)=angle(w(:,kk)'*a_com);
%     
%     
%     
% end
% scatterplot(val); axis([-1.2,1.2,-1.2,1.2]);%
% grid on
% 
% figure
% rho=ones(1,length(omega));
% hold on
% plot(omega,rho);
% plot(phase,rho);
% hold off
% % grid on
% legend('communication phase symbol','phase in communication direction');

%% beampattern scan

bp2=(a_t(:,1).*w(:,1))'*a;
mag2=db(abs(bp2)/max(abs(bp2)));
figure
plot(theta,mag2,'linewidth',1.5);
grid on

figure

for kk=1:length(theta_scan)
as=exp(-1i*2*pi*d/lambda.*n'.*(sind(theta)-sind(theta_scan(kk))));

bps=w(:,1)'*as;
 mags=db(abs(bps)/max(abs(bps)));
% mags=db(abs(bps));

plot(theta,mags,'linewidth',1.5);
hold on
plot([theta_com theta_com ],[-50 10],'b--','linewidth',1.5);
plot([theta_scan(kk) theta_scan(kk)],[-50 10],'g--','linewidth',1.5);
hold off
xlim([-90 90]);
ylim([-50 10]);
xlabel('angle (degree)');
ylabel('beampattern (dB)')
grid on
pause(1e-1);

end



%% phase embeded strategy

w4=w(:,4).*exp(1i*2*pi*d/lambda*n'*sind(40));%
pha=angle(w4'*as(:,81))*180/pi;%


%% BER VERSUS SNR by MonteCarlo Simulation

ws=zeros(N,length(omega));
st=zeros(N,N_pulse*N_fast);
%temp=zeros(N,N_scan*N_fast);
y_com=zeros(1,N_fast*N_pulse);
phase_est=zeros(1,N_pulse);
y=zeros(1,N_pulse);
B_hat=zeros(2,N_pulse);
T=[0 90 180 270];
ber=zeros(1,length(SNR));
for ind=1:length(SNR)
    biterror=0;
    for cnt=1:MentNum
        B=rand(2,N_pulse);
        B(B>0.5)=1;
        B(B<0.5)=0;
    %% transimtted signal
    for kk=1:length(theta_scan)
%         ws=w.*exp(1i*2*pi*d/lambda*n'*sind(thetas(kk)));
        as=exp(-1i*2*pi*d/lambda.*n'.*(sind(theta)-sind(theta_scan(kk))));
        for mm=1:length(omega)
        dif_phi=angle(w(:,mm)'*as(:,81))-omega(mm);
        ws(:,mm)=w(:,mm).*exp(1i*dif_phi);
        end
        temp=zeros(N,N_scan*N_fast);
        for jj=1:N_scan
            temp(:,1+(jj-1)*N_fast:jj*N_fast)=sqrt(PT)*(floor(sum(B(:,jj+(kk-1)*N_scan)==[0 0]')/2)*conj(ws(:,1))+floor(sum(B(:,jj+(kk-1)*N_scan)==[0 1]')/2)*conj(ws(:,2))...
                +floor(sum(B(:,jj+(kk-1)*N_scan)==[1 0]')/2)*conj(ws(:,3))+floor(sum(B(:,jj+(kk-1)*N_scan)==[1 1]')/2)*conj(ws(:,4)))*psi(:,1).';
            
        end
        st(:,1+(kk-1)*N_scan*N_fast:kk*N_scan*N_fast)=temp;
    end
    
   %% received signal
   for kk=1:length(theta_scan)
       as=exp(-1i*2*pi*d/lambda.*n'.*(sind(theta)-sind(theta_scan(kk))));
       y_com(:,(kk-1)*N_scan*N_fast+1:kk*N_scan*N_fast)=as(:,81).'*st(:,(kk-1)*N_scan*N_fast+1:kk*N_scan*N_fast);
          %% add noise 
         Ps=sum(sum(abs(y_com(:,(kk-1)*N_scan*N_fast+1:kk*N_scan*N_fast)).^2))/(N_scan*N_fast);
        %Ps=sum(abs(y_com).^2)/length(y_com);
        N0=Ps/10^(SNR(ind)/10);
        noise = sqrt(N0/2)*(randn(1,N_scan*N_fast) + 1i*randn(1,N_scan*N_fast));
        y_r(:,(kk-1)*N_scan*N_fast+1:kk*N_scan*N_fast)=y_com(:,(kk-1)*N_scan*N_fast+1:kk*N_scan*N_fast)+noise;
   end
%    %% add noise 
%      Ps=sum(sum(abs(y_com(:,(kk-1)*N_fast+1:kk*N_fast)).^2))/(N*N_fast);
%     %Ps=sum(abs(y_com).^2)/length(y_com);
%     N0=Ps/10^(SNR(ind)/10);
%     noise = sqrt(N0/2)*(randn(1,N_fast*N_pulse) + 1i*randn(1,N_fast*N_pulse));
%     y_r=y_com+noise;
%     y_r=y_com;
       %% matched filter and estimate
       
    for ii=1:N_pulse
        y(ii)=y_r((ii-1)*N_fast+1:ii*N_fast)*conj(psi(:,1));
        phase_est(ii)=angle(y(ii))*180/pi;
        if(phase_est(ii)<0)
            phase_est(ii)=phase_est(ii)+360;
        end
        %         phase_est(phase_est<0)=phase_est(phase_est<0)+360;
        if (phase_est(ii)>=T(1)&&phase_est(ii)<T(2))
            B_hat(:,ii)=[0 0]';
        elseif  (phase_est(ii)>=T(2)&&phase_est(ii)<T(3))
            B_hat(:,ii)=[0 1]'; 
        elseif (phase_est(ii)>=T(3)&&phase_est(ii)<T(4))
            B_hat(:,ii)=[1 0]'; 
        elseif (phase_est(ii)>=T(4)&&phase_est(ii)<360)
            B_hat(:,ii)=[1 1]'; 
        end        
    end    
    biterror=biterror+sum(sum(B_hat~=B))/N_pulse/2;
    
    end
    ber(ind)=biterror/MentNum;
    ind
end
figure
semilogy(SNR,ber,'linewidth',1);
grid on;
xlabel('SNR (dB)');
ylabel('BER'); 
