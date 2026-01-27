clear
% close
clc
%% SETTINGS
N = 1000;
m = 20;
desiredAngle = 50;
interfereAngle = [40 70 20 80];
jammAngle = [60 90 30 10];
patPoints = 180;
dBSNRin = -12:2;
SNRin = 10.^(dBSNRin/10);
d = 2;
gamma = 1;
trials = 100;
%% DIFFERENT NUMBER OF ANTENNAS
SINRout = zeros(length(dBSNRin), trials);
steerVec = zeros(m,patPoints);
%% STEERING VECTORS
for i = 1:patPoints
    for k = 1:m
        steerVec(k,i) = exp(1i*pi*cos(deg2rad(i))*(k-1));
    end
end
aTilde = 1/sqrt(2)* ...
         [steerVec(:,desiredAngle).', steerVec(:,desiredAngle)'].'; 
%% AVERAGE OVER TRIALS
for trial = 1:trials    
    fprintf("TRIAL: %g (%g PERCENT DONE)\n",trial,(trial-1)/trials*100)
    %% DEFINITIONS
    r = zeros(m,N);
    rDesired = zeros(m,N);
    rInterferencePlusNoise = zeros(m,N);
    noiseIdx = 0;
    %% DATA GENERATION
    sDesired = sign(rand(N,1)-0.5);
    sInterfere1 = sign(rand(N,1)-0.5);
    sInterfere2 = sign(rand(N,1)-0.5);
    sInterfere3 = sign(rand(N,1)-0.5);
    sInterfere4 = sign(rand(N,1)-0.5);
    sJamm1 = (randn(N,1)+1i*randn(N,1))/sqrt(2);
    sJamm2 = (randn(N,1)+1i*randn(N,1))/sqrt(2);
    sJamm3 = (randn(N,1)+1i*randn(N,1))/sqrt(2);
    sJamm4 = (randn(N,1)+1i*randn(N,1))/sqrt(2);
    
    varDesired = var(sDesired);
    sigmaNu = varDesired./SNRin;
    %% DIFFERENT INPUT SNR
    for noiseVar = sigmaNu
        noiseIdx = noiseIdx + 1;
        %% RECEIVED DATA
        for k=1:m
             rDesired(k,:) = sDesired * ...
                        exp(1i*pi*(k-1)*cos(deg2rad(desiredAngle))); 
             noise = (randn(N,1)+1i*randn(N,1))*sqrt(noiseVar/2);       
             rInterferencePlusNoise(k,:) =  ...
                 sInterfere1 * ...
                    exp(1i*pi*(k-1)* cos(deg2rad(interfereAngle(1)))) + ...
                 sInterfere2 * ...
                    exp(1i*pi*(k-1)*cos(deg2rad(interfereAngle(2)))) + ...
                 sInterfere3 * ...
                    exp(1i*pi*(k-1)*cos(deg2rad(interfereAngle(3)))) + ...
                 sInterfere4 * ...
                    exp(1i*pi*(k-1)*cos(deg2rad(interfereAngle(4)))) + ...
                 sJamm1 * exp(1i*pi*(k-1)*cos(deg2rad(jammAngle(1)))) + ...
                 sJamm2 * exp(1i*pi*(k-1)*cos(deg2rad(jammAngle(2)))) + ...
                 sJamm3 * exp(1i*pi*(k-1)*cos(deg2rad(jammAngle(3)))) + ...
                 sJamm4 * exp(1i*pi*(k-1)*cos(deg2rad(jammAngle(4)))) + ...
                 + noise;
             r(k,:) = rDesired(k,:) + rInterferencePlusNoise(k,:); 
        end
        %% ALGORITHM
        Rhat = zeros(m,m);
        RcHat = zeros(m,m);

        for i = 1:N
            wTilde = conj(gamma)*aTilde/norm(aTilde)^2;

            Rhat = Rhat*(i-1)+r(:,i)*r(:,i)';
            RcHat = RcHat*(i-1)+r(:,i)*r(:,i).';
            Rhat = Rhat/i;
            RcHat = RcHat/i;

            Rtilde = [Rhat, RcHat; conj(RcHat), conj(Rhat)];
            for d_prime = 1:d    
                gTilde = (eye(2*m) - aTilde*aTilde'/norm(aTilde)^2)*Rtilde*wTilde;
                muTilde = gTilde'*Rtilde*wTilde/(gTilde'*Rtilde*gTilde);
                wTilde = wTilde - muTilde*gTilde;
            end
        end
        powDesired = mean(real(wTilde'*[rDesired.',rDesired'].'/sqrt(2)).^2);
        powIntNoise = mean(real(wTilde'*[rInterferencePlusNoise.',rInterferencePlusNoise'].'/sqrt(2)).^2);
        SINRout(noiseIdx,trial) = powDesired/powIntNoise;
    end
    clc
end
SINR = mean(pow2db(SINRout),2);
%% PLOT
plot(dBSNRin,SINR,'LineWidth',1,"DisplayName","WL-AVF") 
xlabel("Input SNR")
ylabel("Output SINR (dB)")
title("Steady-state SINR versus the input SNR (N = 1000)")
legend show
grid on
hold on