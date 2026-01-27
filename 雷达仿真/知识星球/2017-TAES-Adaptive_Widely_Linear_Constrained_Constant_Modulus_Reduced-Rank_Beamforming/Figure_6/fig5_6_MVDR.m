clear
% close
clc
%% SETTINGS
N = 50000;
M = linspace(10,30,11);
desiredAngle = 50;
interfereAngle = [40 70 20 80];
jammAngle = [60 90 30 10];
patPoints = 180;
SNRin = -3;
sigmaN = db2pow(-SNRin);
trials = 1;
muW = 0.001;
%% DIFFERENT NUMBER OF ANTENNAS
SINRoutAvg = zeros(length(M),1);
arrayAntennaIdx = 0;
for m = M
    fprintf("ANTENNA NUMBERS: %g",m)
    arrayAntennaIdx = arrayAntennaIdx + 1;
    SINRout = zeros(trials,1);
    steerVec = zeros(m,patPoints);
    %% STEERING VECTORS
    for i = 1:patPoints
        for k = 1:m
            steerVec(k,i) = exp(1i*pi*cos(deg2rad(i))*(k-1));
        end
    end
    %% AVERAGE OVER TRIALS
    for trial = 1:trials    
        fprintf("\nTRIAL: %g",trial)
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
            %% ALG 1
%         Pa = eye(m) - ...
%             (steerVec(:,desiredAngle)*steerVec(:,desiredAngle)')/...
%             (steerVec(:,desiredAngle)'*steerVec(:,desiredAngle));
%         fa = steerVec(:,desiredAngle)/...
%             (steerVec(:,desiredAngle)'*steerVec(:,desiredAngle));
%         
%         w = [1;zeros(m-1,1)];
% 
%         for i = 1:N
%             y = w'*r(:,i);
%             w = Pa*(w - (muW*conj(y)*r(:,i))) + fa;
%         end
        
            %% ALG 2
        Rhat = zeros(m,m);
        for i = 1:N
            Rhat = Rhat + r(:,i)*r(:,i)';
        end
        Rhat = Rhat/N;
%         Rhat = eye(m)*(1+8+sigmaN);
        w = inv(Rhat)*steerVec(:,desiredAngle)/...
                (steerVec(:,desiredAngle)'*inv(Rhat)*steerVec(:,desiredAngle));

            %% ALG 3
%         Rhat = zeros(m,m);
%         w = steerVec(:,desiredAngle)/norm(steerVec(:,desiredAngle))^2;
%         for i = 1:N
%             Rhat = Rhat*(i-1) + r(:,i)*r(:,i)';
%             Rhat = Rhat/i;
%             g = (eye(m) - steerVec(:,desiredAngle)*steerVec(:,desiredAngle)'/...
%                 norm(steerVec(:,desiredAngle))^2)*Rhat*w;
%             if g == 0
%                 break
%             end
%             mu = g'*Rhat*w/(g'*Rhat*g);
%             w = w - mu*g;
%         end
            
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
        SINRout(trial) = real((w'*Rdesired*w)/(w'*Rinterference*w));
    end
    SINRoutAvg(arrayAntennaIdx) = pow2db(mean(SINRout));
    clc
end
plot(M,SINRoutAvg,'LineWidth',1,"DisplayName","MVDR-L") 
xlabel("Number of antennas")
ylabel("Output SINR")
title("Effect of number of antennas on output SINR")
legend show
grid on
hold on