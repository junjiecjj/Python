clear
% close
clc
%% SETTINGS
N = 1000;
m = 20;
desiredAngle = 50;
interfereAngle = [40 70 20 80];
jammAngle = [10 30];
patPoints = 180;
SNRin = -3;
sigmaN = db2pow(-SNRin);
trials = 100;
angSeparations = 1:8;
%% DIFFERENT NUMBER OF ANTENNAS
SINRout = zeros(length(angSeparations),trials);
steerVec = zeros(m,patPoints);
%% STEERING VECTORS
for i = 1:patPoints
    for k = 1:m
        steerVec(k,i) = exp(1i*pi*cos(deg2rad(i))*(k-1));
    end
end
%% AVERAGE OVER TRIALS
for trial = 1:trials   
    fprintf("TRIAL: %g (%g PERCENT DONE)\n",trial,(trial-1)/trials*100)
    %% DEFINITIONS
    r = zeros(m,N);
    rDesired = zeros(m,N);
    rInterferencePlusNoise = zeros(m,N);
    angSepIdx = 0;
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
    for angSeparation = angSeparations
        angSepIdx = angSepIdx + 1;
        fprintf("ANGULAR SEPARATION: %g (%g PERCENT DONE)\n", angSeparation, (angSepIdx-1)/length(angSeparations)*100)
        for k=1:m
             rDesired(k,:) = sDesired* ...
                        exp(1i*pi*(k-1)*cos(deg2rad(desiredAngle)));  
             noise = (randn(N,1)+1i*randn(N,1))*sqrt(sigmaN/2);       
             rInterferencePlusNoise(k,:) =  ...
                 sInterfere1*exp(1i*pi*(k-1)* ...
                 cos(deg2rad(desiredAngle+angSeparation))) + ...
                 sInterfere2*exp(1i*pi*(k-1)* ...
                 cos(deg2rad(desiredAngle-angSeparation))) + ...
                 sJamm1*exp(1i*pi*(k-1)*cos(deg2rad(jammAngle(1)))) + ...
                 sJamm2*exp(1i*pi*(k-1)*cos(deg2rad(jammAngle(2)))) + ...
                 + noise;

             r(k,:) = rDesired(k,:) + rInterferencePlusNoise(k,:); 
        end
        %% ALGORITHM
        Rhat = zeros(m,m);
        w = steerVec(:,desiredAngle)/norm(steerVec(:,desiredAngle))^2;
        for i = 1:N
            Rhat = Rhat*(i-1) + r(:,i)*r(:,i)';
            Rhat = Rhat/i;
            g = (eye(m) - steerVec(:,desiredAngle)*steerVec(:,desiredAngle)'/...
                norm(steerVec(:,desiredAngle))^2)*Rhat*w;
            if g == 0
                break
            end
            mu = g'*Rhat*w/(g'*Rhat*g);
            w = w - mu*g;
        end
        %% APPROXIMATION OF AUTO-CORRELATION FUNCTION   
        Rdesired = zeros(m,m);
        Rinterference = zeros(m,m);
        for q=1:N
            Rdesired = Rdesired + rDesired(:,q)*rDesired(:,q)';
            Rinterference = Rinterference + ...
                            rInterferencePlusNoise(:,q)* ...
                            rInterferencePlusNoise(:,q)';
        end
        Rdesired = Rdesired/N;
        Rinterference = Rinterference/N;
        %% SINR OUT 
        SINRout(angSepIdx,trial) = real((w'*Rdesired*w)/(w'*Rinterference*w));
    end
    clc
end
SINR = pow2db(mean(SINRout,2));
%%
plot(angSeparations,SINR,'LineWidth',1,"DisplayName","LCMV-FR") 
xlabel("Minimum angular separation (degree)")
ylabel("Output SINR (dB)")
title("Steady-state SINR versus the minimum angular separation between the interferer and the desired user (N = 1000, SNR= −3dB)")
legend show
grid on
hold on
