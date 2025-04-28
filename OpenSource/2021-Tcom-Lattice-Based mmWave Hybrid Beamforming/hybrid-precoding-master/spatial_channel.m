function [H]=spatial_channel(Nt ,Nr, Nc)
% the function to generate the channel matrices in hybrid precoding
% these codes are taken from:
% https://github.com/yuxianghao/Alternating-minimization-algorithms-for-hybrid-precoding-in-millimeter-wave-MIMO-systems
 
Nray = 10; % # of rays in each cluster % these are for channel generizations
 

angle_sigma = 10/180*pi; %standard deviation of the angles in azimuth and elevation both of Rx and Tx

gamma = sqrt((Nt*Nr)/(Nc*Nray)); %normalization factor
sigma = 1; %according to the normalization condition of the H
 

    for c = 1:Nc
        AoD_m = unifrnd(0,2*pi,1,2);%uniform random AoD and AoA
        AoA_m = unifrnd(0,2*pi,1,2);
        
        AoD(1,[(c-1)*Nray+1:Nray*c]) = laprnd(1,Nray,AoD_m(1),angle_sigma);
        AoD(2,[(c-1)*Nray+1:Nray*c]) = laprnd(1,Nray,AoD_m(2),angle_sigma);
        AoA(1,[(c-1)*Nray+1:Nray*c]) = laprnd(1,Nray,AoA_m(1),angle_sigma);
        AoA(2,[(c-1)*Nray+1:Nray*c]) = laprnd(1,Nray,AoA_m(2),angle_sigma);
    end
    
    H = zeros(Nr,Nt);
    for j = 1:Nc*Nray
        At(:,j) = array_response(AoD(1,j),AoD(2,j),Nt); %UPA array response
        Ar(:,j) = array_response(AoA(1,j),AoA(2,j),Nr);
        alpha(j) = normrnd(0,sqrt(sigma/2)) + normrnd(0,sqrt(sigma/2))*sqrt(-1);
        H = H + alpha(j) * Ar(:,j) * At(:,j)';% Ar*At'!!!
    end
    H = gamma * H;
    
end

function y  = laprnd(m, n, mu, sigma)
%LAPRND generate i.i.d. laplacian random number drawn from laplacian distribution
%   with mean mu and standard deviation sigma. 
%   mu      : mean
%   sigma   : standard deviation
%   [m, n]  : the dimension of y.
%   Default mu = 0, sigma = 1. 
%   For more information, refer to
%   http://en.wikipedia.org./wiki/Laplace_distribution

%   Author  : Elvis Chen (bee33@sjtu.edu.cn)
%   Date    : 01/19/07

%Check inputs
if nargin < 2
    error('At least two inputs are required');
end

if nargin == 2
    mu = 0; sigma = 1;
end

if nargin == 3
    sigma = 1;
end

% Generate Laplacian noise
u = rand(m, n)-0.5;
b = sigma / sqrt(2);
% y = mu - b.* sign(u).* log(1- 2.* abs(u));
y = mu + sign(u) .* ( pi - b.* log( exp(pi./b) + (2-2.*exp(pi./b)) .* abs(u) ) );

end

function y = array_response(a1,a2,N)
for m= 0:round(sqrt(N))-1
    for n= 0:round(sqrt(N))-1
        y(m*(round(sqrt(N)))+n+1) = exp( 1i* pi* ( m*sin(a1)*sin(a2) + n*cos(a2) ) );
    end
end
y = y.'/round(sqrt(N));
end