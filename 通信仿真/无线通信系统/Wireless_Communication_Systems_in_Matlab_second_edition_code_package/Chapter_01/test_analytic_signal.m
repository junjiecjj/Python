%Test routine to check analytic_signal function
t=0:0.001:0.5-0.001;
x = sin(2*pi*10*t); %real-valued f = 10 Hz
subplot(2,1,1); plot(t,x);%plot the original signal
title('x[n] - real-valued signal'); xlabel('n'); ylabel('x[n]');

z = analytic_signal(x); %construct analytic signal
subplot(2,1,2); plot(t, real(z), 'k'); hold on;
plot(t, imag(z), 'r');
title('Components of Analytic signal'); 
xlabel('n'); ylabel('z_r[n] and z_i[n]');
legend('Real(z[n])','Imag(z[n])');