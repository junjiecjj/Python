function [variance] =NumMSEQspk(rho)

x=rho;
a =0.2131;
b =-2.129;
c = 0.7841;
d = -0.616;
y = a*exp(b*x) + c*exp(d*x);

variance = y;
end

