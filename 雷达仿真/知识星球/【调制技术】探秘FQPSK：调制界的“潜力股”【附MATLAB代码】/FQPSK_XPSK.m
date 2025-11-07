
%------------FQPSK Modulation-------------------
close all;
clear all;

%------------parameters----------------
M = 4;
Rsym = 8000;                  % Input symbol rate
Rbit = Rsym * log2(M);        % Input bit rate
Nos = 60;                      % Oversampling factor
ts = (1/Rsym) / Nos;          % Input sample period
Fcarrier =80000;           %carrier

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
%------FQPSK full-symbol waveform---------
subplot(4,4,1),plot(i,s0);title('s0');grid on;axis([i(1),i(Nos),-1,1]);
subplot(4,4,2),plot(i,s8);title('s8');grid on;axis([i(1),i(Nos),-1,1]);
subplot(4,4,3),plot(i,s1);title('s1');grid on;axis([i(1),i(Nos),-1,1]);
subplot(4,4,4),plot(i,s9);title('s9');grid on;axis([i(1),i(Nos),-1,1]);
subplot(4,4,5),plot(i,s2);title('s2');grid on;axis([i(1),i(Nos),-1,1]);
subplot(4,4,6),plot(i,s10);title('s10');grid on;axis([i(1),i(Nos),-1,1]);
subplot(4,4,7),plot(i,s3);title('s3');grid on;axis([i(1),i(Nos),-1,1]);
subplot(4,4,8),plot(i,s11);title('s11');grid on;axis([i(1),i(Nos),-1,1]);
subplot(4,4,9),plot(i,s4);title('s4');grid on;axis([i(1),i(Nos),-1,1]);
subplot(4,4,10),plot(i,s12);title('s12');grid on;axis([i(1),i(Nos),-1,1]);
subplot(4,4,11),plot(i,s5);title('s5');grid on;axis([i(1),i(Nos),-1,1]);
subplot(4,4,12),plot(i,s13);title('s13');grid on;axis([i(1),i(Nos),-1,1]);
subplot(4,4,13),plot(i,s6);title('s6');grid on;axis([i(1),i(Nos),-1,1]);
subplot(4,4,14),plot(i,s14);title('s14');grid on;axis([i(1),i(Nos),-1,1]);
subplot(4,4,15),plot(i,s7);title('s7');grid on;axis([i(1),i(Nos),-1,1]);
subplot(4,4,16),plot(i,s15);title('s15');grid on;axis([i(1),i(Nos),-1,1]);

%------------Signal-Mapping-------------------
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
    
t=((0:ts:length(FQPSK_IData)*ts-ts)*1e-3)/ts;
t=t';
figure(2);
plot(t,FQPSK_IData,'b','linewidth',1.5);
axis([-.1 t(end)+.1 -1.5 1.5]);
xlabel('t/Ts');ylabel('s_I (t)');grid;
title('Ini-phase');

figure(3);
plot(t,FQPSK_QData,'r','linewidth',1.5);
axis([-.1 t(end)+.1 -1.5 1.5]);
xlabel('t/Ts');ylabel('s_Q (t - Ts/2)');grid;
title('Quadrature-phase (DELAYED BY Ts/2)');

%product the carrier and sum
cosCarrier = cos(2*pi*Fcarrier*(0:length(FQPSK_IData)-1)*ts);
sinCarrier = sin(2*pi*Fcarrier*(0:length(FQPSK_IData)-1)*ts);


finalData = FQPSK_IData.*cosCarrier' + FQPSK_QData.*sinCarrier';
time=0:ts:length(finalData)*ts-ts;  % Input time
time=time';


figure(4);
plot(time,finalData,'k');
axis ([-inf inf -2 2]);
xlabel('Time (ms)'); ylabel('FQPSK');grid;
title('FQPSK (XPSK) output')

figure(5)
plot(time,finalData,'k');
axis ([0 1e-3 -2 2]);
xlabel('Time (ms)'); ylabel('FQPSK');grid;
title('FQPSK (XPSK) output')

