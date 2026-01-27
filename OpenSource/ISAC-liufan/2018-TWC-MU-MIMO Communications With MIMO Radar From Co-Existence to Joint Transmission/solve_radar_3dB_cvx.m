function R = solve_radar_3dB_cvx(N, P0, d, theta0, theta1, theta2, theta_sidelobe)
% ------------------------------------------------------------------
% Solve problem (10) to design 3dB main-beam covariance R
% Nt = N, diag(R) = P0/N, 3dB width: [theta1, theta2] around theta0
% ------------------------------------------------------------------

deg2rad = pi/180;

% Steering vectors at key angles
a0  = exp(1j*2*pi*d*sin(theta0*deg2rad)*(0:N-1).').';
a1  = exp(1j*2*pi*d*sin(theta1*deg2rad)*(0:N-1).').';
a2  = exp(1j*2*pi*d*sin(theta2*deg2rad)*(0:N-1).').';

% Sidelobe steering matrix
Msl  = numel(theta_sidelobe);
Asl  = zeros(N, Msl);
for m = 1:Msl
    th = theta_sidelobe(m);
    Asl(:,m) = exp(1j*2*pi*d*sin(th*deg2rad)*(0:N-1).').';
end

cvx_begin sdp quiet
    variable R(N,N) complex semidefinite
    variable t
    
    % Main-beam power at theta0
    P0_main = a0 * R * a0';
    
    % 约束：3dB points
    P1 = a1 * R * a1';
    P2 = a2 * R * a2';
    
    minimize( -t )   % maximize t
    
    subject to
        % 主瓣与旁瓣差 >= t (在 sidelobe 区域)
        for m = 1:Msl
            am = Asl(:,m).';
            Pm = am * R * am';
            P0_main - Pm >= t;
        end
        
        % 3dB constraints (等式)
        P1 == P0_main/2;
        P2 == P0_main/2;
        
        % diag(R) = P0/N
        diag(R) == (P0/N)*ones(N,1);
        
        R == R';   % Hermitian
cvx_end

end
