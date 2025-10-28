%figure9
% dual function radar and communication paltform
% wide band
clc,clear;
% close all;
%% parameters
% radar parameters
c = 3e8;%speed of light
f0 = 24e9;%carrier frequency
lambda = c/f0;%wavelength
N = 10;%number of elements
d = lambda/2;%distance between elements
%% variables
theta = -90:0.5:90; %angle range
mainlobe = [-15 15]; %mainlobe
theta_start = find(theta == mainlobe(1));
theta_end = find(theta == mainlobe(2));
theta_main = theta(theta_start:theta_end);% theta in the mainlobe
theta_side = theta([1:theta_start-1-20,theta_end+1+20:length(theta)]);%theta in the sidelobe
theta_com = -50;% communication direction
theta_radar = 0;%radar direction
% n = 0:N-1;
n = -(N-1)/2:(N-1)/2;
a_side = exp(-1i*2*pi*d/lambda.*n'.*sind(theta_side));%steering vector of theta in sidelobe
a_main = exp(-1i*2*pi*d/lambda.*n'.*sind(theta_main));%steering vector of theta in mainlobe
a_t = exp(-1i*2*pi*d/lambda.*n*sind(theta_radar)).';% steering vector of radar direction
a_com = exp(-1i*2*pi*d/lambda.*n'.*sind(theta_com));% steering vector of communication direction
delta = sqrt([0.01 0.0033 0.0066 1e-4]);% sidelobe level at communication direction

% w0 = 1/N*ones(N,1);
pattern_dir  = exp(1i*pi*sind(theta_main*pi/180));% desired beampattern in mainlobe
% pattern_dir = ones(1,length(theta_main));% desired beampattern in mainlobe
u1 = zeros(N,4); % optimal weight vector of four beampattern
%optimize four beampattern weight vector respectivly
for j= 1:4
cvx_begin
    variable u(N,1) 
    variable b(1)
    minimize b
    subject to
        norm(u.'*a_main-pattern_dir,1)<= b;
%         max(abs(u.'*a_main-1))<=b;
%         u'*a_t ==1;
        max(abs(u'*a_side)) <= 0.1;
        u'*a_com == delta(j);
cvx_end
%  
u1(:,j)=u;%save optimial weight vector
% clear u
end
%% plot
% plot four beam pattern
figure
hold on
y1 = u1(:,1)'*exp(-1i*2*pi*d/lambda.*n'.*sind(theta));
mag1 = db(abs(y1));
y2 = u1(:,2)'*exp(-1i*2*pi*d/lambda.*n'.*sind(theta));
mag2 = db(abs(y2));
y3 = u1(:,3)'*exp(-1i*2*pi*d/lambda.*n'.*sind(theta));
mag3 = db(abs(y3));
y4 = u1(:,4)'*exp(-1i*2*pi*d/lambda.*n'.*sind(theta));
mag4 = db(abs(y4));
plot(theta,mag1,'r-.','linewidth',2);
plot(theta,mag2,'g--','linewidth',2);
plot(theta,mag3,'b:','linewidth',2);
plot(theta,mag4,'c-','linewidth',2);
grid minor
xlabel('Angle(degree)')
ylabel('Amplitude(dB)')
ylim([-60 10])
xlim([-90 90])
plot([theta_com(1) theta_com(1)],[-60 10],'m--','linewidth',2)
hold off
set(gca,'fontsize',14)
legend('1st transmitted pattern','2st transmitted pattern','3st transmitted pattern','4st transmitted pattern')
title('beam pattern')
box on