function s_rec=get_s_rec(s,m,p,theta)  %用于产生经过阵元后的信号数据
    A=zeros(m,p);
    j=sqrt(-1);
    %%% 阵元间距为半个波长
    wi=pi*sin(theta);
    A=exp(-j*wi'*[0:(m-1)]);  % 阵列流型
    s_rec=A'*s;
    s_rec=awgn(s_rec,10);  % SNR=10 db
end
