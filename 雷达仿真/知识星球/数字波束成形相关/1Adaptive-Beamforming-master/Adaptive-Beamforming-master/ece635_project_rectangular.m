% Robert Baummer
% A comparison of several adaptive algorithms for beam forming a rectangular array
function ece635_project_rectangular

close all;

%Global variables describing the signal environment
global AOA_i;
global AOA_s;
global nI;
%Global variables describing the rectangular array
global L;
global M;
global dx;
global dy;

%% Define Array Characteristics
%Number of array elements (L odd so origin has an antenna)
%LxM Rectangular Array
L = 5;
M = 5;
%Array spacing d = wavelength/2
dx = 0.5;
dy = 0.5;

%% Angle of arrival for signal of interest (SOI)
%[theta phi]
%AOA_s = [-pi/12 pi/12];
AOA_s = [pi/4 pi/8];
%AOA_s = [0 0];

%% Randomly Generate Interferer Angles of Arrival
%Calculate array beamwidth to prevent interferer from being too close to
%the SOI to resolve
thetap = asin(1/pi*(2.782/L - 2*pi*dx*sin(AOA_s(1))));
thetam = asin(1/pi*(-2.782/L - 2*pi*dx*sin(AOA_s(1))));
Beamwidth_theta = abs(thetap - thetam);

thetap = asin(1/pi*(2.782/L - 2*pi*dy*sin(AOA_s(2))));
thetam = asin(1/pi*(-2.782/L - 2*pi*dy*sin(AOA_s(2))));
Beamwidth_phi = abs(thetap - thetam);

%Number of interferers (# interferers + signal <= L)
nI = 15;
%interfer power (linear)
%note signal of interest power is 0.5
pwr_i = 10;

%Angles of arrival for interferers randomly distributed
rng(17); %keep random angles consistent from run to run
%[theta phi]
AOA_i = rand(nI,2)*pi - pi/2;
%make sure interferers don't have same AOA as SOI
for i = 1:nI
    if AOA_i(i,1) < (AOA_s(1) + Beamwidth_theta) && AOA_i(i,1) > (AOA_s(1) - Beamwidth_theta)
        AOA_i(i,1) = -i*pi/10;
    end
end;

for i = 1:nI
    if AOA_i(i,2) < (AOA_s(2) + Beamwidth_phi) && AOA_i(i,2) > (AOA_s(2) - Beamwidth_phi)
        AOA_i(i,2) = -i*pi/10;
    end
end

disp(['The Desired AOA (theta, phi) = ',num2str(AOA_s*180/pi), ' Degrees'])
for i = 1:nI
    disp(['The Undesired AOAs (theta, phi) = ',num2str(AOA_i(i,:)*180/pi),' Degrees'])
end

%% Signal Definitions
%noise std deviation
sigma = 0.1;
%length of signals
N = 1000;   
%Signal of Interest
S = cos(2*pi*(0:N-1)/100);
%Interferers
I = sqrt(pwr_i)*randn(nI,N);

%array steering vectors for arriving signals centered at origin
index_x = (-floor(L/2):floor(L/2))';
index_y = (-floor(M/2):floor(M/2))';
%SOI Array Steering Vector
AFsx = exp(1i*index_x*2*pi*dx*sin(AOA_s(1))*cos(AOA_s(2)));
AFsy = exp(1i*index_y*2*pi*dy*sin(AOA_s(1))*sin(AOA_s(2)));
%Results in matrix [AFx1AFy1 AFx1AFy2...;AFx2AFy1 AFx2AFy2...]
%AFx in columns, AFy in rows
AFs = AFsx*AFsy.';
%Turn MxL matrix into vector of stacked columns M*Lx1
AFs = AFs(:);
%Interferer Array Steering Vectors
for i = 1:nI
    AFix = exp(1i*index_x*2*pi*dx*sin(AOA_i(i,1))*cos(AOA_i(i,2)));
    AFiy = exp(1i*index_y*2*pi*dy*sin(AOA_i(i,1))*sin(AOA_i(i,2)));
    AF = AFix*AFiy.';
    % turn MxLxnI matrrix into vector of stacked columns M*LxnI
    AFi(:,i) = AF(:);
end

%% Signal Power Estimates
%estimate desired signal correlation matrix from all samples arriving at
%angle steering vector AFs
Xs = AFs*S;
Rxxs = 1/N*(Xs*Xs');
%estimate interfer signals correlation matrix from all samples arriving at
%angle steering vectors AFi
Xi = zeros(L*M,N);
for i = 1:nI
    Xi = Xi + AFi(:,i)*I(i,:);
end
%include noise correlation matrix in calculation
Rxxi = 1/N*Xi*Xi' + sigma^2*diag(ones(L*M,1));

%% Add gaussian noise with sigma^2 power to arriving signals
Snoisy = S + sigma*randn(1,N);
Inoisy = I + sigma*randn(nI,N);

%% LMS algorithm
a = AFs + sum(AFi,2);
%Steering vector correlation matrix
Raa = a*a';
mu = 0.25/real(trace(Raa));
w = zeros(L*M,1);
for i = 1:N
    %The signals arriving from the antennas is equal to the sum of each
    %signal times its array factor plus noise
    X = Snoisy(i)*AFs + AFi*Inoisy(:,i);
    %The output of the weighted antenna array is the antenna weights times
    %to signals arriving from the antennas
    y(i) = w(:,i)'*X;

    %error from SOI
    e(i) = conj(S(i)) - y(i);
    %update weights using error*X as gradient estimate
    w(:,i+1) = w(:,i) + mu*conj(e(i))*X;

    %SIR calculation
    sigma_s_sqd = w(:,i+1)'*Rxxs*w(:,i+1);
    sigma_i_sqd = w(:,i+1)'*Rxxi*w(:,i+1);
    SIR_lms(i) = abs(sigma_s_sqd)/abs(sigma_i_sqd);
end

%Reshape final weights to LxM matrix
w = reshape(w(:,N),M,L)';

%Plot final antenna pattern
rectangular_array_plot(w);
axis auto;
%set(gca, 'xdir', 'reverse', 'ydir', 'reverse');
view([25 45]) %temp debug
title('LMS');

figure
hold on;
title('LMS Learning Curve');
xlabel('Iteration');
ylabel('e^2');
plot(abs(e).^2);
%plot recovered SOI vs SOI
% figure;
% hold on;
% plot(Snoisy,'k');
% plot(real(y),'c');

%% RLS algorithm
clear y e w;
alpha = 0.995;
w = zeros(L*M,1);
%initialize Rxx^-1 to identity matrix
Rxx_hat_inv = diag(ones(L*M,1));
for i = 2:N
    %next value of array input vector
    X = Snoisy(i)*AFs + AFi*Inoisy(:,i);

    %recursive calcuation of Rxx^-1
    %next value of Rxx_hat_inv depends on previous value of Rxx_hat_inv
    Rxx_hat_inv = (1/alpha)*Rxx_hat_inv - ((1/alpha)^2*Rxx_hat_inv*(X*X')*Rxx_hat_inv)/(1 + (1/alpha)*X'*Rxx_hat_inv*X);
    g = Rxx_hat_inv*X;
    %The output of the antenna array
    y(i) = w(:,i-1)'*X;

    e(i) = (S(i)) - y(i);
    w(:,i) = w(:,i-1) + g*(e(i));

    %SIR calculation
    sigma_s_sqd = w(:,i)'*Rxxs*w(:,i);
    sigma_i_sqd = w(:,i)'*Rxxi*w(:,i);
    SIR_rls(i) = abs(sigma_s_sqd)/abs(sigma_i_sqd);
end

%Reshape final weights to LxM matrix
w = reshape(w(:,N),M,L)';

%Plot final antenna pattern
rectangular_array_plot(w);
axis auto;
%set(gca, 'xdir', 'reverse', 'ydir', 'reverse');
view([25 45]) %temp debug
title('RLS');
%plot learning curve
figure
hold on;
title('RLS Learning Curve');
xlabel('Iteration');
ylabel('e^2');
plot(abs(e).^2);

%% Conjugate Gradient Method
clear y e w;
K = 200;    %2 periods of desired signal

w = zeros(L*M,1);
%construct a matrix of K array samples
X = AFs*Snoisy(1:K);
%add in the interferers one at a time
for i = 1:nI
    X = X + AFi(:,i)*Inoisy(i,1:K);
end
%A is K by L matrix of array samples
A = X';
%Desired signal
D = S(1:K)';

%calculate the first residual
r(:,1) = D - A*w(:,1);
DIR = A'*r(:,1);
for i = 1:N
    %calculate the step size mu
    mu = (r(:,i)'*(A*A')*r(:,i))/(DIR'*(A'*A)*DIR);
    %calculate new weight
    w(:,i+1) = w(:,i) + mu*DIR;

    %calculate new residual
    r(:,i+1) = r(:,i) - mu*A*DIR;
    %linear search to determine alpha to minimize the cost function
    alpha = (r(:,i+1)'*(A*A')*r(:,i+1))/(r(:,i)'*(A*A')*r(:,i));
    %calculate new conjugate direction for next pass
    DIR = A'*r(:,i+1) - alpha*DIR;

    %calculate error
    X = Snoisy(i)*AFs + AFi*Inoisy(:,i);
    y(i) = w(:,i)'*X;
    e(i) = S(i) - y(i);

    %SIR calculation
    sigma_s_sqd = w(:,i+1)'*Rxxs*w(:,i+1);
    sigma_i_sqd = w(:,i+1)'*Rxxi*w(:,i+1);
    SIR_cgm(i) = abs(sigma_s_sqd)/abs(sigma_i_sqd);
end

%Reshape final weights to LxM matrix
w = reshape(w(:,N),M,L)';

%Plot final antenna pattern
rectangular_array_plot(w);
axis auto;
%set(gca, 'xdir', 'reverse', 'ydir', 'reverse');
view([25 45]) %temp debug
title('CG');
%plot learning curve
figure
hold on;
title('CG Learning Curve');
xlabel('Iteration');
ylabel('e^2');
plot(abs(e).^2);

%% Kalman Based Normalized LMS
clear y e w;

w = zeros(L*M,1);
M1=6;
sigma_w = .1;
for i = 1:N
    %The signals arriving from the antennas is equal to the sum of each
    %signal times its array factor plus noise
    X = Snoisy(i)*AFs + AFi*Inoisy(:,i);
    %The output of the weighted antenna array is the antenna weights times
    %to signals arriving from the antennas
    y(i) = w(:,i)'*X;

    %error from SOI
    e(i) = S(i) - y(i);

    P = X'*X;
    qv = sigma^2;
    sigma_w(i+1) = sigma_w(i)*(1 - P/(M1-1)/(P + qv/sigma_w(i)));
    %update weights using error*X as gradient estimate
    w(:,i+1) = w(:,i) + X*e(i)/(P + qv/sigma_w(i));

    %SIR calculation
    sigma_s_sqd = w(:,i+1)'*Rxxs*w(:,i+1);
    sigma_i_sqd = w(:,i+1)'*Rxxi*w(:,i+1);
    SIR_klms(i) = abs(sigma_s_sqd)/abs(sigma_i_sqd);
end

%Reshape final weights to LxM matrix
w = reshape(w(:,N),M,L)';

%Plot final antenna pattern
rectangular_array_plot(w);
axis auto;
%set(gca, 'xdir', 'reverse', 'ydir', 'reverse');
view([25 45]) %temp debug
title('KLMS');
%plot learning curve
figure
hold on;
title('KLMS Learning Curve');
xlabel('Iteration');
ylabel('e^2');
plot(abs(e).^2);


%% SINR
figure;
hold on;
SINR = 10*log10(real(Rxxs(1,1)/real(Rxxi(1,1))));
title(['SINR for ',num2str(nI), ' Interfers with SINR ', num2str(SINR),' dB']);
xlabel('Iteration');
ylabel('SINR (dB)');
plot(10*log10(mean(SIR_lms,1)),'b');
plot(10*log10(mean(SIR_rls,1)),'k');
plot(10*log10(mean(SIR_cgm,1)),'c');
plot(10*log10(mean(SIR_klms,1)),'g');
legend('LMS','RLS', 'CGM','KLMS');