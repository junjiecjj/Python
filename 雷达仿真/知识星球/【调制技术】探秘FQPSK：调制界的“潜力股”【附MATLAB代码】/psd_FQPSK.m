%------------FQPSK Modulation-------------------
close all;
clear all;

%------------parameters----------------
M = 4;
Rsym = 8000;                  % Input symbol rate
Rbit = Rsym * log2(M);        % Input bit rate
Nos = 60;                      % Oversampling factor
ts = (1/Rsym) / Nos;          % Input sample period
Fcarrier1 =80000;           %carrier
Fcarrier2 = 65000;   

A = 1/sqrt(2);    %base pulse

%------------IJF-Encoder-------------------
%s0 s8
s0 = zeros(1,Nos)+A;
s8 = -s0;
%s1 s9
for i = 1:Nos/2
    s1(i) = A;
end
for i = Nos/2+1:Nos
    s1(i) = 1-(1-A)*cos(pi*(i-Nos/2)/Nos)*cos(pi*(i-Nos/2)/Nos);
end
s9 = -s1;
%s2 s10
for i = 1:Nos/2
    s2(i) = 1-(1-A)*cos(pi*(i-Nos/2)/Nos)*cos(pi*(i-Nos/2)/Nos);
end
for i = Nos/2+1:Nos
    s2(i) = A;
end
s10 = -s2;
%s3 s11
for i = 1:Nos
    s3(i) = 1-(1-A)*cos(pi*(i-Nos/2)/Nos)*cos(pi*(i-Nos/2)/Nos);
end
s11 = -s3;
%s4 s12
for i = 1:Nos
    s4(i) = A*sin(pi*(i-Nos/2)/Nos);
end
s12 = -s4;
%s5 s13
for i = 1:Nos/2
    s5(i) = A*sin(pi*(i-Nos/2)/Nos);
end
for i = Nos/2+1:Nos
    s5(i) = sin(pi*(i-Nos/2)/Nos);
end
s13 = -s5;
%s6 s14
for i = 1:Nos/2
    s6(i) = sin(pi*(i-Nos/2)/Nos);
end
for i = Nos/2+1:Nos
    s6(i) = A*sin(pi*(i-Nos/2)/Nos);
end
s14 = -s6;
%s7 s15
for i = 1:Nos
    s7(i) = sin(pi*(i-Nos/2)/Nos);
end
s15 = -s7;
S = [s0;s1;s2;s3;s4;s5;s6;s7;s8;s9;s10;s11;s12;s13;s14;s15];
i=1:Nos;
i=i-Nos/2;
SourceLen = 40;
%source
source = randi([0 3],SourceLen,1); 

%qpsk modulation
hModulator = comm.QPSKModulator();
hModulator.PhaseOffset = pi/4;
modData = step(hModulator, source);
%modData = pskmod(source,4,pi/4);

modIData = (real(modData)>0)*-1+1;
modQData = (imag(modData)>0)*-1+1;

FQPSK_IData = zeros(SourceLen-4,Nos);
FQPSK_QData = FQPSK_IData;


for i = 3:length(modIData)-1
    I0 = xor(modQData(i),modQData(i-1));
    I1 = xor(modQData(i-1),modQData(i-2));
    I2 = xor(modIData(i),modIData(i-1));
    I3 = (modIData(i)>0);
    Q0 = xor(modIData(i+1),modIData(i));
    Q1 = I2;
    Q2 = I0;
    Q3 = (modQData(i)>0);
    IndexI(i) = I3*8+I2*4+I1*2+I0;
    IndexQ(i) = Q3*8+Q2*4+Q1*2+Q0;
    FQPSK_IData(i-2,:) = S(IndexI(i)+1,:);
    FQPSK_QData(i-2,:) = S(IndexQ(i)+1,:);
end

FQPSK_IData = reshape(FQPSK_IData',[],1);
FQPSK_QData = reshape(FQPSK_QData',[],1);
%delay
    FQPSK_QData = [zeros(Nos/2,1); FQPSK_QData(1:end-Nos/2)];
%product the carrier and sum
cosCarrier1 = cos(2*pi*Fcarrier1*(0:length(FQPSK_IData)-1)*ts);
sinCarrier1 = sin(2*pi*Fcarrier1*(0:length(FQPSK_IData)-1)*ts);

cosCarrier2 = cos(2*pi*Fcarrier2*(0:length(FQPSK_IData)-1)*ts);
sinCarrier2 = sin(2*pi*Fcarrier2*(0:length(FQPSK_IData)-1)*ts);

finalData1 = FQPSK_IData.*cosCarrier1' + FQPSK_QData.*sinCarrier1';
finalData2 = FQPSK_IData.*cosCarrier2' + FQPSK_QData.*sinCarrier2';

%------power spectral density----------
fourier_data1 = fft(finalData1);
fourier_data2 = fft(finalData2);
L= length(fourier_data1);

fourier_data1 = fftshift(fourier_data1)./L;
fourier_data2 = fftshift(fourier_data2)./L;

freq = linspace(-L/2,L/2,L);
f_Rs = 10*freq/Rsym;

psd1 = 10*log10(fourier_data1);
psd2 = 10*log10(fourier_data2);

figure(1);
plot(f_Rs,psd1,'k',f_Rs,psd2,'r')
grid;
xlabel('freq / bit Rate');ylabel('power spectral density , dB / Hz');
legend('F_c_a_r_r_i_e_r = 80 KHz','F_c_a_r_r_i_e_r = 65 KHz');
title('Power Spectral Density FQPSK')
set(gca,'fontsize',16);
axis([-1.5 1.5 -5 -60])