function Bm=get_B(m,theta)  %���ڲ�����������%����������������������
    u0=0.5*sin(theta(1)); % ������Ԫ���Ϊ�������
    a0=exp(-j*2*pi*[0:m-1]'*u0);
    u=u0+[1:m-1];
    B=exp(-j*2*pi*[0:m-1]'*u);
    Bm=conj(B');%% M-1*M �ľ���
end
