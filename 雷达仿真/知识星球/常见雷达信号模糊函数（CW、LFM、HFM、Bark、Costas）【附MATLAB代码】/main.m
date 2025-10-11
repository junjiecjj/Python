close all; clear all; clc;

T = 0.26;
fs = 20e3;
fc = 1e3;
B = 1000;
t = 0:1/fs:T-1/fs;

type_index = 1; 
type_name = {'PCW', 'LFM', 'HFM', 'Bark', 'Costas'};

switch type_index
    case 1
        % 1. pcw
        x = exp(1i*2*pi*fc*t);
    case 2
        % 2. lfm
        k = B/T;
        f0 = fc-B/2;
        x = exp(1i*2*pi*(f0*t+k/2*t.^2));
    case 3
        % 3. hfm
        f0 = fc+B/2;
        beta = B/f0/(fc-B/2)/T;
        x = exp(1i*2*pi/beta*log(1+beta*f0*t));
    case 4
        % 4.bark信号
        % 4. hfm + pcw
        %     f0 = fc+B/2;
        %     beta = B/f0/(fc-B/2)/T;
        %     x = sin(2*pi/beta*log(1+beta*f0*t)) + sin(2*pi*fc*t);
        % bark = [1,1,1,1,1,-1,-1,1,1,-1,1,-1,1];
        bark = (randi(2,1,100)-1)*2-1;
        Tbark = T/length(bark);
        tbark = 0:1/fs:Tbark-1/fs;
        s = zeros(1,length(bark)*length(tbark));
        for i = 1:length(bark)
            if bark(i) == 1
                s((i-1)*length(tbark)+1:i*length(tbark))=exp(1j*2*pi*fc*tbark);
            else
                s((i-1)*length(tbark)+1:i*length(tbark))=exp(1j*(2*pi*fc*tbark+pi));
            end
        end
        x = [s, zeros(1,length(t)-length(s))];
    case 5
        % 5.costas信号
        costas = [2,4,8,5,10,9,7,3,6,1];
        f = fc-B/2+(costas-1)*B/(length(costas)-1);
        Tcostas = T/length(costas);
        tcostas = 0:1/fs:Tcostas-1/fs;
        s = zeros(1,length(costas)*length(tcostas));
        for i = 1:length(costas)
            s((i-1)*length(tcostas)+1:i*length(tcostas)) = exp(1j*2*pi*f(i)*tcostas);
        end
        x = [s,zeros(1,length(t)-length(s))]; 
end

re_fs = 0.9*fs:2:1.1*fs;
alpha = re_fs/fs;       % Doppler ratio, alpha = 1-2*v/c
doppler = (1-alpha)*fc;	% Doppler = 2v/c*fc = (1-alpha)*fc

N_a = length( resample(x,fs,min(re_fs)) );
N = N_a + length(x)-1;
afmag = zeros(length(alpha),N);

tic
parfor i = 1:length(alpha)
    if ceil(length(x)/alpha(i)) ~= N_a
        x_alpha = [resample(x,fs,re_fs(i)),zeros(1, N_a-ceil(length(x)/alpha(i)))];
    else
        x_alpha = resample(x,fs,re_fs(i));
    end
    %     x_alpha = Doppler(x,1/re_fs(i),1/fs);
    x_temp = zeros(1,length(x_alpha));
    for j = 1:length(x_alpha)
        x_temp(j) = conj(x_alpha(length(x_alpha)-j+1));
    end
    %     disp(num2str(i))
    afmag_temp = conv(x_temp,x);
    M = length(afmag_temp);
    afmag(i,:) = afmag_temp*sqrt(alpha(i));
end
toc

delay = ([1:N]-N_a)/fs;

tau = 0.2;
fd = 100;
indext = find(delay>=-tau & delay<=tau);
indexf = find(doppler>=-fd & doppler<=fd);
delay1 = delay(indext);
doppler1 = doppler(indexf);

mag = abs(afmag);
mag = mag/max(max(mag));
% mag = 10*log10(mag);
mag1 = mag(:,indext);
mag1 = mag1(indexf,:);
[row,column] = find(mag1<-100);
mag1(row,column)=-60;

figure(1);
mesh(doppler1,delay1,mag1.');
% axis([-100,100,-tau,tau,-50,0]);
colorbar;
set(gcf,'color','w');
xlabel('Doppler (Hz)','FontName','Times New Roman','FontSize',10);
ylabel('Delay (sec)','FontName','Times New Roman','FontSize',10);
zlabel('Level (dB)','FontName','Times New Roman','FontSize',10);
title(['WAF of ',type_name{type_index} ,' Signal'],'FontName','Times New Roman','FontSize',10);

figure(2);
% v=[-3,-3];
% contour(delay1,doppler1,mag1,v,'ShowText','on');grid on;%模糊度图
contour(delay1,doppler1,mag1);grid on;%模糊度图
xlabel('Delay (Sec)','FontName','Times New Roman','FontSize',10);
ylabel('Doppler (Hz)','FontName','Times New Roman','FontSize',10);
title('Contour of AF','FontName','Times New Roman','FontSize',10)

figure(3)
subplot(211);
plot(doppler1,mag1(:,floor(length(indext)/2)),'b')
xlabel('Doppler (Hz)','FontName','Times New Roman','FontSize',10);
ylabel('Amp','FontName','Times New Roman','FontSize',10);
title('Zero Delay','FontName','Times New Roman','FontSize',10)
subplot(212)
plot(delay1,mag1(floor(length(indexf)/2),:),'b')
xlabel('Delay (sec)','FontName','Times New Roman','FontSize',10);
ylabel('Amp','FontName','Times New Roman','FontSize',10);
title('Zero Doppler','FontName','Times New Roman','FontSize',10)

%%
temp=clock;
temp=sum(temp(4:6))*sum(temp(2:3));
temp=round(temp/10);
rand('seed',temp);
%%
%plot([0:length(x_alpha)-1]/length(x_alpha)*fs,abs(fft(x_alpha)));
%hold on;
%plot([0:length(x_alpha2)-1]/length(x_alpha2)*fs,abs(fft(x_alpha2)))





