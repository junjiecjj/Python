function [x,x_old]=damping(x,x_old,mes)
x=mes*x+(1-mes)*x_old;
x_old=x;
end