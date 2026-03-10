function [Q] = BDISB(R,K,M,eps)


Q=diag(diag(R));

Tt=0;
count =0;

while Tt == 0    
LLw = rem(count,14)+1;
[UU,DDDD]=eig(R,Q);
[~,idx] = sort(real(diag(DDDD))); 
UU1 = UU(:,idx);
UU = UU1(:,1:(M-K));
UU = UU/norm(UU,'fro');


VU = UU;
UU1 = UU*(UU');
UU = R*UU1 + UU1*R;

f0= norm((R-Q)*VU,'fro')^2;

if LLw == 1

c=UU(1:4,1:4);

[Xy,Dy]=eig(c);
dyy=diag(Dy);
I=find(real(dyy) < 0);
if length(I)~=0
    dyy(I)=zeros(length(I),1);
end
Dy=diag(dyy);
c= Xy*Dy*(Xy');
c=c(:);
B1=UU1(1:4,1:4);
B= kron(eye(4),B1)+kron(B1.',eye(4));
B2=B\c; 
XYX = ((reshape(B2,4,4)));
Q(1:4,1:4) = XYX;

elseif LLw == 2
Q(5,5)=0.5*diag(diag(UU(5,5)))*diag(1./diag(UU1(5,5)));
elseif LLw == 3
Q(6,6)=0.5*diag(diag(UU(6,6)))*diag(1./diag(UU1(6,6)));
elseif LLw == 4
Q(7,7)=0.5*diag(diag(UU(7,7)))*diag(1./diag(UU1(7,7)));
elseif LLw == 5
Q(8,8)=0.5*diag(diag(UU(8,8)))*diag(1./diag(UU1(8,8)));
elseif LLw == 6
Q(9,9)=0.5*diag(diag(UU(9,9)))*diag(1./diag(UU1(9,9)));

elseif LLw==7
c=UU(10:12,10:12);

[Xy,Dy]=eig(c);
dyy=diag(Dy);
I=find(real(dyy) < 0);
if length(I)~=0
    dyy(I)=zeros(length(I),1);
end
Dy=diag(dyy);
c= Xy*Dy*(Xy');
c=c(:);
B1=UU1(10:12,10:12);
B= kron(eye(3),B1)+kron(B1.',eye(3));
B2=B\c;
XYX = ((reshape(B2,3,3)));
Q(10:12,10:12) = XYX;

elseif LLw == 8
Q(13,13)=0.5*diag(diag(UU(13,13)))*diag(1./diag(UU1(13,13)));
elseif LLw == 9
Q(14,14)=0.5*diag(diag(UU(14,14)))*diag(1./diag(UU1(14,14)));
elseif LLw == 10
Q(15,15)=0.5*diag(diag(UU(15,15)))*diag(1./diag(UU1(15,15)));
elseif LLw == 11
Q(16,16)=0.5*diag(diag(UU(16,16)))*diag(1./diag(UU1(16,16)));
elseif LLw == 12
Q(17,17)=0.5*diag(diag(UU(17,17)))*diag(1./diag(UU1(17,17)));
elseif LLw == 13
Q(18,18)=0.5*diag(diag(UU(18,18)))*diag(1./diag(UU1(18,18)));


else
c=UU(19:20,19:20);
[Xy,Dy]=eig(c);
dyy=diag(Dy);
I=find(real(dyy) < 0);
if length(I)~=0
    dyy(I)=zeros(length(I),1);
end
Dy=diag(dyy);
c= Xy*Dy*(Xy');
c=c(:);
B1=UU1(19:20,19:20);
B= kron(eye(2),B1)+kron(B1.',eye(2));
B2=B\c;
XYX = ((reshape(B2,2,2)));
Q(19:20,19:20) = XYX;

end

f1 = norm((R-Q)*VU,'fro')^2;

if  abs(f0-f1) < (eps)
    Tt =1;
end

count=count+1;
end



end

