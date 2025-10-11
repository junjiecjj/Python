function Bm=get_B(m,theta)  %用于产生阻塞矩阵%采用正交法构造阻塞矩阵
    u0=0.5*sin(theta(1)); % 假设阵元间距为半个波长
    a0=exp(-j*2*pi*[0:m-1]'*u0);
    u=u0+[1:m-1];
    B=exp(-j*2*pi*[0:m-1]'*u);
    Bm=conj(B');%% M-1*M 的矩阵
end
