function y=CFAR(x)
%×÷ºãÐé¾¯´¦Àí
%%%%%%%%%%%%%%%%%%%%%%%
n=length(x);
for i=1:n
    if(x(i)>=10^3)
        y(i)=x(i);
    else y(i)=0;
    end
end
w=y(:);
i=1:length(w);
plot(i,w);