function a = SterMtx_UPA(N, d_norm, ang_azi, ang_ele)
    % UPA steering matrix, N should be 2-element
    % ang_azi and ang_ele can be vectors with the same length
    % if ang_azi and ang_ele are scalars, the result is steering vector
    
    Nx = N(1);
    Ny = N(2);
    L = size(ang_azi,1) * size(ang_azi,2);
    ang_azi = reshape(ang_azi,1,L);
    ang_ele = reshape(ang_ele,1,L);
    a = kr(exp(-1i*2*pi*d_norm*(0:Nx-1).'*(cos(ang_azi).*sin(ang_ele))),...
            exp(-1i*2*pi*d_norm*(0:Ny-1).'*(sin(ang_azi).*sin(ang_ele)))) / sqrt(Nx*Ny);
end