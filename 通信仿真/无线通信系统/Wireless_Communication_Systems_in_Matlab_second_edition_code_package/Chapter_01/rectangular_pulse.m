fs=500; %sampling frequency
T=0.2; %width of the rectangule pulse in seconds
t=-0.5:1/fs:0.5; %time base
g=(t >-T/2) .* (t<T/2) + 0.5*(t==T/2) + 0.5*(t==-T/2);
%g=rectpuls(t,T); %using inbuilt function (signal proc toolbox)
plot(t,g);title(['Rectangular Pulse width=', num2str(T),'s']);