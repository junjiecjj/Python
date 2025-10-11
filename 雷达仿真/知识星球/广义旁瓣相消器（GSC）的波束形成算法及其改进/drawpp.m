function y=drawpp(m,wop)
    %figure,plot(e)
    thetas=[-90:90];
    tm=thetas*pi/180;
    am=exp(-j*pi*[0:m-1]'*sin(tm));
    A=abs(wop'*am);  %阵列响应
    A=A/max(A);
    figure,polar(tm,A)
    A=10*log10(A);  %对数图
    hold on,title('归一化阵列响应幅值极坐标图，八阵元，信噪比20db')
    figure,plot(thetas,A);
    hold on,title('八阵元，信噪比20db')
    hold on,xlabel('入射角/度')
    hold on,ylabel('归一化 A=10*log10(A);')
    grid on 
    axis([-90 90 -35 0]);
    hold on,plot(-45,-35:0.1:0,'r');
    hold on,plot(30,-35:0.1:0,'r');
    hold on,plot(0,-35:0.1:0,'r');
    hold on,plot(60,-35:0.1:0,'r');
end