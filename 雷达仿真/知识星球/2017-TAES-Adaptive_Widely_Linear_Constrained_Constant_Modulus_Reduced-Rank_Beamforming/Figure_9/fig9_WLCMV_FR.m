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
SNRin = -3;
sigmaN = db2pow(-SNRin);
trials = 200;

SINRout = zeros(N,trials);
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
    fprintf("TRIAL: %g\n",trial)
    %% DEFINITIONS
    r = zeros(m,N);
    rDesired = zeros(m,N);
    rInterferencePlusNoise = zeros(m,N);
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
    %% RECEIVED DATA
    for k=1:m
         rDesired(k,:) = sDesired * ...
                    exp(1i*pi*(k-1)*cos(deg2rad(desiredAngle)));  
         noise = (randn(N,1)+1i*randn(N,1))*sqrt(sigmaN/2);       
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
    Rdesired = zeros(m,m);
    RcDesired = zeros(m,m);
    Rinterference = zeros(m,m);
    RcInterference = zeros(m,m);
    w = aTilde/norm(aTilde)^2;
    for i = 1:N
        Rhat = Rhat*(i-1) + r(:,i)*r(:,i)';
        RcHat = RcHat*(i-1) + r(:,i)*r(:,i).';
        Rhat = Rhat/i;
        RcHat = RcHat/i;
        R = 0.5*[Rhat, RcHat; conj(RcHat), conj(Rhat)];
        g = (eye(2*m) - aTilde*aTilde'/...
            norm(aTilde)^2)*R*w;
        if g == 0
            break
        end
        mu = g'*R*w/(g'*R*g);
        w = w - mu*g;
        %% APPROXIMATION OF AUTO-CORRELATION FUNCTION   
        Rdesired = Rdesired*(i-1) + rDesired(:,i)*rDesired(:,i)';
        RcDesired = RcDesired*(i-1) + rDesired(:,i)*rDesired(:,i).';
        Rinterference = Rinterference*(i-1) + ...
                        rInterferencePlusNoise(:,i)* ...
                        rInterferencePlusNoise(:,i)';
        RcInterference = RcInterference*(i-1) + ...
                         rInterferencePlusNoise(:,i)* ...
                         rInterferencePlusNoise(:,i).';
        Rdesired = Rdesired/i;
        RcDesired = RcDesired/i;
        Rinterference = Rinterference/i;
        RcInterference = RcInterference/i;

        Rs = 0.5*[Rdesired, RcDesired; ...
                  conj(RcDesired), conj(Rdesired)];
        Rin = 0.5*[Rinterference, RcInterference; ...
                   conj(RcInterference), conj(Rinterference)];
        %% SINR OUT 
        SINRout(i,trial) = real((w'*Rs*w)/(w'*Rin*w));
    end
    clc
end
SINR = pow2db(mean(SINRout,2));

plot(1:N,SINR,'LineWidth',1,"DisplayName","WLCMV-FR") 
xlabel("Number of snapshots")
ylabel("Output SINR (dB)")
title("Output SINR convergence performance(SNR= −3dB)")
legend show
grid on
hold on