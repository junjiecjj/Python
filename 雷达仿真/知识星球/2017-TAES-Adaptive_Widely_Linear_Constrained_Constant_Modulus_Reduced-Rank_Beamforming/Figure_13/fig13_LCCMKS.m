clear
% close
clc
%% SETTINGS
N = 2000;
m = 20;
desiredAngle = 50;
interfereAngle = [40 70 20 80];
jammAngle = [60 90 30 10];
patPoints = 180;
dBSNRin = -7:2;
SNRin = 10.^(dBSNRin/10);
d = 3;

% alpha = 0.998;
% gamma = 0.044;
% delta = 0.001;

% alpha = 0.998;
% gamma = 0.0545;
% delta = 0.0001;

alpha = 0.998;
gamma = 0.057;
delta = 0.01;

trials = 300;
steerVec = zeros(m, patPoints);
err = zeros(length(dBSNRin), trials);
%% STEERING VECTORS
for i = 1:patPoints
    for k = 1:m
        steerVec(k,i) = exp(1i*pi*cos(deg2rad(i))*(k-1));
    end
end
%% B MATRIX
B = eye(m) - steerVec(:,desiredAngle)*steerVec(:,desiredAngle)'/...
    (steerVec(:,desiredAngle)'*steerVec(:,desiredAngle));
B1 = eye(m) - steerVec(:,desiredAngle)*steerVec(:,desiredAngle)'/...
     (2*steerVec(:,desiredAngle)'*steerVec(:,desiredAngle));
B2 = -steerVec(:,desiredAngle)*steerVec(:,desiredAngle).'/...
     (2*steerVec(:,desiredAngle)'*steerVec(:,desiredAngle));
Btilde = [B1, B2; conj(B2), conj(B1)];
%% AVERAGE OVER TRIALS
for trial = 1:trials    
    fprintf("TRIAL: %g/%g (%g PERCENT DONE)",trial,trials,(trial-1)/trials*100)
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
        w = [1;zeros(d-1,1)];
        Qinv = delta*eye(d);
        Rhat = zeros(m,m);

        for i = 1:N
            Rhat = Rhat*(i-1)+r(:,i)*r(:,i)';
            Rhat = Rhat/i;

            Tr = zeros(m,d);
            Tr(:,1) = Rhat*steerVec(:,desiredAngle);
            Tr(:,1) = Tr(:,1)/norm(Tr(:,1));

            for d_prime = 2:d
               Tr(:,d_prime) = Rhat*B*Tr(:,d_prime-1);
               Tr(:,d_prime) = Tr(:,d_prime)/norm(Tr(:,d_prime));        
            end

            rb = B*r(:,i);
            rBar = Tr'*rb;
            y = gamma*steerVec(:,desiredAngle)'*r(:,i) - w'*rBar;
            xTilde = conj(y)*rBar;
            dTilde = gamma*conj(y)*steerVec(:,desiredAngle)'*r(:,i)-1;

            kTilde = Qinv*xTilde / (alpha+xTilde'*Qinv*xTilde);
            zetaTilde = dTilde - w'*xTilde;
            Qinv = Qinv/alpha - kTilde*xTilde'*Qinv;
            w = w + kTilde*conj(zetaTilde);
        end
        err(noiseIdx,trial) = zetaTilde;
    end
    clc
end
meanErr = mean(abs(err).^2,2);
%% PLOT
plot(dBSNRin,meanErr,'LineWidth',1,"DisplayName","LCCM-KS") 
xlabel("Input SNR")
ylabel("Output SINR (dB)")
title("Steady-state SINR versus the input SNR (N = 1000)")
legend show
grid on
hold on