function y=blind_iq_compensation(z)
%Function to estimate and compensate IQ impairments for the single-
%branch IQ impairment model
%  y=blind_iq_compensation(z) estimates and compensates IQ imbalance
I=real(z);
Q=imag(z);

theta1=(-1)*mean(sign(I).*Q);
theta2=mean(abs(I));
theta3=mean(abs(Q));

c1=theta1/theta2;
c2=sqrt((theta3^2-theta1^2)/theta2^2);

yI = I;
yQ = (c1*I+Q)/c2;
y= (yI +1i*yQ);
end