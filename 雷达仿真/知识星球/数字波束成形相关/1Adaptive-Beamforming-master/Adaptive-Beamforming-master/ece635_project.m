% Rob Baummer
% A comparison of several adaptive algorithms for beam forming a linear array
function ece635_project

%% Setup
close all;
global AOA_i;
global AOA_s;
global nI;
%True/False whether to plot individual algorithm plots
plot_individual = true;
%Number of array elements (L odd so origin has an antenna)
L = 21;
%Array spacing d = wavelength/2
d = 0.5;
%length of signals
N = 500;   
%Number of runs
nRuns = 1;

e_lms = zeros(1,N);
e_rls = zeros(1,N);
e_cg = zeros(1,N);
e_klms = zeros(1,N);
%Angle of arrival for signal of interest (SOI)
AOA_s = pi/15;

thetap = asin(1/pi*(2.782/L - 2*pi*d*sin(AOA_s)));
thetam = asin(1/pi*(-2.782/L - 2*pi*d*sin(AOA_s)));
Beamwidth = abs(thetap - thetam);

%noise std deviation
sigma = 0.1;

%Number of interferers (# interferers + signal <= L)
nI = 10;
%interfer power (linear)
%note signal of interest power is 0.5
pwr_i = 1;

for q = 1:nRuns
    disp(['Iteration # ', num2str(q)]);
    %Angles of arrival for interferers randomly distributed
    rng(q*19); %keep random angles consistent from run to run
    AOA_i = rand(nI,1)*pi - pi/2;
    %make sure interferers don't have same AOA as SOI
    for i = 1:nI
        if AOA_i(i) < (AOA_s + Beamwidth) && AOA_i(i) > (AOA_s - Beamwidth)
            AOA_i(i) = -i*pi/18;
        end
    end;

    disp(['The Desired AOA = ',num2str(AOA_s*180/pi), ' Degrees'])
    for i = 1:nI
        disp(['The Undesired AOAs = ',num2str(AOA_i(i)*180/pi),' Degrees'])
    end

    %% Signal Definitions
    S = cos(2*pi*(0:N-1)/100);
    I = sqrt(pwr_i)*randn(nI,N);
    % array steering vectors for arriving signals centered at origin
    index = -floor(L/2):floor(L/2);
    AFs = exp(1i*(index)*2*pi*d*sin(AOA_s))';
    for i = 1:nI
        AFi(:,i) = exp(1i*(index)*2*pi*d.*sin(AOA_i(i)))';
    end

    %% Signal Power Estimates
    %estimate desired signal correlation matrix from all samples arriving at
    %angle steering vector AFs
    Xs = AFs*S;
    Rxxs = 1/N*(Xs*Xs');
    %estimate interfer signals correlation matrix from all samples arriving at
    %angle steering vectors AFi
    Xi = zeros(L,N);
    for i = 1:nI
        Xi = Xi + AFi(:,i)*I(i,:);
    end
    %include noise correlation matrix in calculation
    Rxxi = 1/N*Xi*Xi' + sigma^2*diag(ones(L,1));

    %% Add gaussian noise with sigma^2 power to arriving signals
    Snoisy = S + sigma*randn(1,N);
    Inoisy = I + sigma*randn(nI,N);

    %% LMS algorithm
    a = AFs + sum(AFi,2);
    %Steering vector correlation matrix
    Raa = a*a';
    mu = 0.25/real(trace(Raa));
    w = zeros(L,1);
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
        SIR_lms(q,i) = abs(sigma_s_sqd)/abs(sigma_i_sqd);
    end
    
    e_lms = e_lms + e;
    if plot_individual == true
        %Calculate Final Antenna Array Factor
        theta = -pi/2:pi/360:pi/2;
        for i = 1:L
            AF(i,:) = w(i,N)*exp(1i*(index(i))*2*pi*d*sin(theta));
        end
        figure(1)
        hold on;
        title('LMS Antenna Pattern');
        xlabel('Degrees');
        ylabel('Gain');
        plot(theta*180/pi, abs(sum(AF,1)));
        figure(2)
        hold on;
        title('LMS Learning Curve');
        xlabel('Iteration');
        ylabel('e^2');
        plot(abs(e).^2);
        % figure;
        % hold on;
        % plot(S,'k');
        % plot(real(y),'c');
    end
    
    peak_theta_lms(q,:) = track_antenna_pattern(L, d, w);

    %% RLS algorithm
    clear y e w;
    alpha = 0.995;
    w = zeros(L,1);
    %initialize Rxx^-1 to identity matrix
    Rxx_hat_inv = diag(ones(L,1));
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
        SIR_rls(q,i) = abs(sigma_s_sqd)/abs(sigma_i_sqd);
    end

    e_rls = e_rls + e;
    if plot_individual == true
        %Calculate Final Antenna Array Factor
        theta = -pi/2:pi/360:pi/2;
        for i = 1:L
            AF(i,:) = w(i,N)*exp(1i*(index(i))*2*pi*d*sin(theta));
        end
        figure(3)
        hold on;
        title('RLS Antenna Pattern');
        xlabel('Degrees');
        ylabel('Gain');
        plot(theta*180/pi, abs(sum(AF,1)));
        figure(4)
        hold on;
        title('RLS Learning Curve');
        xlabel('Iteration');
        ylabel('e^2');
        plot(abs(e).^2);
        % figure;
        % hold on;
        % plot(S,'k');
        % plot(real(y),'c');
    end
    
    peak_theta_rls(q,:) = track_antenna_pattern(L, d, w);

    %% Conjugate Gradient Method
    clear y e w;
    K = 200;    %2 periods of desired signal

    w = zeros(L,1);
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
        SIR_cgm(q,i) = abs(sigma_s_sqd)/abs(sigma_i_sqd);
    end

    e_cg = e_cg + e;
    if plot_individual == true
        %Calculate Final Antenna Array Factor
        theta = -pi/2:pi/360:pi/2;
        for i = 1:L
            AF(i,:) = w(i,K)*exp(1i*(index(i))*2*pi*d*sin(theta));
        end
        figure(5)
        hold on;
        title('Conjugate Gradient Antenna Pattern');
        xlabel('Degrees');
        ylabel('Gain');
        plot(theta*180/pi, abs(sum(AF,1)));
        figure(6)
        hold on;
        title('Conjugate Gradient Learning Curve');
        xlabel('Iteration');
        ylabel('e^2');
        plot(abs(e).^2);
        % figure;
        % hold on;
        % plot(D,'k');
        % plot(real(y),'c');
    end
    
    peak_theta_cg(q,:) = track_antenna_pattern(L, d, w);

    %% Kalman Based Normalized LMS
    clear y e w;

    w = zeros(L,1);
    M=6;
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
        sigma_w(i+1) = sigma_w(i)*(1 - P/(M-1)/(P + qv/sigma_w(i)));
        %update weights using error*X as gradient estimate
        w(:,i+1) = w(:,i) + X*e(i)/(P + qv/sigma_w(i));

        %SIR calculation
        sigma_s_sqd = w(:,i+1)'*Rxxs*w(:,i+1);
        sigma_i_sqd = w(:,i+1)'*Rxxi*w(:,i+1);
        SIR_klms(q,i) = abs(sigma_s_sqd)/abs(sigma_i_sqd);
    end

    e_klms = e_klms + e;
    if plot_individual == true
        %Calculate Final Antenna Array Factor
        theta = -pi/2:pi/360:pi/2;
        for i = 1:L
            AF(i,:) = w(i,K)*exp(1i*(index(i))*2*pi*d*sin(theta));
        end
        figure(7)
        hold on;
        title('KLMS Antenna Pattern');
        xlabel('Degrees');
        ylabel('Gain');
        plot(theta*180/pi, abs(sum(AF,1)));
        figure(8)
        hold on;
        title('KLMS Learning Curve');
        xlabel('Iteration');
        ylabel('e^2');
        plot(abs(e).^2);
    end
    
    peak_theta_klms(q,:) = track_antenna_pattern(L, d, w);
    
end

%% SIR Comparison
figure;
hold on;
SINR = 10*log10(real(Rxxs(1,1)/real(Rxxi(1,1))));
title(['SINR Comparison for ',num2str(nI), ' Interfers with SINR ', num2str(SINR),' dB']);
xlabel('Iteration');
ylabel('SINR (dB)');
plot(10*log10(mean(SIR_lms,1)),'b');
plot(10*log10(mean(SIR_rls,1)),'k');
plot(10*log10(mean(SIR_cgm,1)),'c');
plot(10*log10(mean(SIR_klms,1)),'g');
legend('LMS','RLS','CGM','KLMS');

%Average e^2
figure;
hold on;
plot(abs(e_lms/nRuns).^2,'k');
plot(abs(e_rls/nRuns).^2,'r');
plot(abs(e_cg/nRuns).^2,'g');
plot(abs(e_klms/nRuns).^2,'b');
title(['Average Learning Curve over ', num2str(nRuns)]);
xlabel('Iterations');
ylabel('e^2');
legend('LMS','RLS','CGM','KLMS');

%Min MSE, Misadjustment
Rxx=Rxxi+Rxxs;
w_opt=.5*(Rxx\AFs);
r=Rxx*w_opt;
min_mse = real(0.5-r'*w_opt);
z_lms = abs(e_lms/nRuns).^2;
z_rls = abs(e_rls/nRuns).^2;
z_cg = abs(e_cg/nRuns).^2;
z_klms = abs(e_klms/nRuns).^2;
excess_mse_lms = mean(z_lms(end-50:end)-min_mse);
excess_mse_rls = mean(z_rls(end-50:end)-min_mse);
excess_mse_cg = mean(z_cg(end-50:end)-min_mse);
excess_mse_klms = mean(z_klms(end-50:end)-min_mse);
misadj_lms = excess_mse_lms/min_mse;
misadj_rls = excess_mse_rls/min_mse;
misadj_cg = excess_mse_cg/min_mse;
misadj_klms = excess_mse_klms/min_mse;

disp(['LMS excess MSE = ',num2str(excess_mse_lms)])
disp(['RLS excess MSE = ',num2str(excess_mse_rls)])
disp(['CGM excess MSE = ',num2str(excess_mse_cg)])
disp(['KLMS excess MSE = ',num2str(excess_mse_klms)])
disp(['LMS Misadjustment = ',num2str(misadj_lms)])
disp(['RLS Misadjustment = ',num2str(misadj_rls)])
disp(['CGM Misadjustment = ',num2str(misadj_cg)])
disp(['KLMS Misadjustment = ',num2str(misadj_klms)])

%Average abs peak theta over nRuns for each algorithm
figure
hold on;
plot(mean((peak_theta_lms(:,1:25)),1)*180/pi,'-')
plot(mean((peak_theta_rls(:,1:25)),1)*180/pi,':')
plot(mean((peak_theta_cg(:,1:25)),1)*180/pi,'.-')
plot(mean((peak_theta_klms(:,1:25)),1)*180/pi,'--')
title('Average |Antenna Direction|');
xlabel('Iterations');
ylabel('Degrees');
legend('LMS','RLS','CGM','KLMS');

%Plot last test case of arriving signals
plot_arriving_signals;

function plot_arriving_signals
global AOA_s;
global AOA_i;
global nI;

figure;

polar([AOA_s AOA_s],[.25 1], 'v-b');
hold on;
for i = 1:nI
    polar([AOA_i(i) AOA_i(i)],[.25 1],'v-r');
end
view(-90,90);
plot(zeros(21,1),-0.5:.05:0.5,'*');
axis([-.1 1 -1 1])
legend('SOI','Interferer');