function Rxy=sequence_correlation(x,y,k1,k2)
%Compute correlation between two binary sequences x and y
%k1 and k2 define the limits of the range of desired shifts
x=x(:);y=y(:); %serialize
if length(x) ~= length(y), error('Sequences x and y should be of same length'); end
L=length(x);
rangeOfKs = k1:1:k2; %range of cyclic shifts
Rxy=zeros(1,length(rangeOfKs));

%determine initial shift for sequence x
if k1~=0, %do not shift x if k1 is 0
    start=mod(L+k1,L)+1;%+1 for matlab index
    final=mod(L+k1-1,L)+1;%+1 for matlab index
    x=[x(start:end);x(1:final)];%initial shifted sequence 
end
q=floor(length(x)/length(y)); %assumes x is always larger
r=rem(length(x),length(y)); %remainder part
y=[repmat(y,q,1); y(1:r)];

for i=1:length(rangeOfKs),
    agreements= sum(x==y);%number of agreements
    disagreements = sum(x~=y); %number of disagreements
    x=[x(2:end);x(1)]; %shift left once for each iteration
    Rxy(i)=agreements-disagreements; %sequence correlation
end,end