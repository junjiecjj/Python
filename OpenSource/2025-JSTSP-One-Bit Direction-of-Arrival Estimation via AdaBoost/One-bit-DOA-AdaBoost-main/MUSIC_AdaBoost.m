function [DOA_deg,DOA_rad] = MUSIC_AdaBoost(x0_one,K,M,N,g_num,iter)


g_num_dft=g_num;
LL = [2*K:-1:K+1 K*ones(1,iter-(K))];

kk=0:M-1;
WW=zeros(M,M);
idx=0:M-1;
for ii=1:M
WW(ii,:)=exp(-1j*2*pi*kk(ii)*idx/M);
end

G=M;
A_bar = WW;
S_bar = zeros(G,N);
PHI_R = [real(A_bar) -imag(A_bar);imag(A_bar) real(A_bar)];
XX = PHI_R.';
hh = 2*G;

for iii = 1:N

r = [real(x0_one(:,iii));imag(x0_one(:,iii))];
m = ones(2*M,1)/(2*M);
h_g = zeros(hh,g_num);
alph=zeros(g_num,1);



for jj=1:g_num_dft
    
    h_R = XX * (r .* m);
    h_g(:,jj)=h_R;

    pred_1 = sign(PHI_R*h_R);
    er_1 = r.*pred_1;
    II= find(er_1 < 0);
    z_m = zeros(size(m));
    z_m(II,1) = ones(length(II),1);

    err_m = (((z_m.')*m)/(sum(m))) ;

    if err_m == 0
      alph_m = 1;
      alph(jj) = alph_m;
      break
    else
      alph_m = 0.5*log((1-err_m)/err_m);
      alph(jj) = alph_m;
    end

    m = m .* exp(-(alph_m*er_1));
    m = m/sum(m);

end

h_R_1 = h_R(1:G);
h_R_2 = h_R(1+G:end);
S_bar(:,iii)= h_R_1+1j*h_R_2;

end

s_til = zeros(G,1);

for iii=1:G
    s_til(iii,1) = norm(S_bar(iii,:),2);
end


[~,LOCS] = maxk(s_til,LL(1));


S = S_bar(LOCS,:);
V = S.';
y_one = x0_one.';

for gg=1:iter

G = LL(gg);
A_bar = V;
S_bar = zeros(G,M);

PHI_R = [real(A_bar) -imag(A_bar);imag(A_bar) real(A_bar)];
XX = PHI_R.';
hh = 2*G;

for iii = 1:M

r = [real(y_one(:,iii));imag(y_one(:,iii))];
m = ones(2*N,1)/(2*N);
h_g = zeros(hh,g_num);
alph=zeros(g_num,1);

for jj=1:g_num
        
    h_R = XX * (r .* m);
    h_g(:,jj)=h_R;

    pred_1 = sign(PHI_R*h_R);
    er_1 = r.*pred_1;
    II= find(er_1 < 0);
    z_m = zeros(size(m));
    z_m(II,1) = ones(length(II),1);

    err_m = (((z_m.')*m)/(sum(m))) ;

    if err_m == 0
      alph_m = 1;
      alph(jj) = alph_m;
      break
    else
      alph_m = 0.5*log((1-err_m)/err_m);
      alph(jj) = alph_m;
    end

    m = m .* exp(-(alph_m*er_1));
    m = m/sum(m);

end

h_R = h_g*alph;
h_R_1 = h_R(1:G);
h_R_2 = h_R(1+G:end);

S_bar(:,iii)= h_R_1+1j*h_R_2;

end

GG = S_bar.';
tt = zeros(G,1);
for zz=1:G    
vsh = GG(:,zz)/norm(GG(:,zz));    
GGG = eye(M) - (vsh*(vsh'));   
[~,thet]=music(GGG,1,M);
tt(zz)=thet;
end


thet=tt;
idxR = (0:(M-1))';
d_lambda = 0.5;
A_bar = zeros(M,G);
for ii = 1:G
    A_bar(:,ii) = exp(-1i*2*pi*d_lambda*sin(thet(ii))*idxR);
end

S_bar = zeros(G,N);
PHI_R = [real(A_bar) -imag(A_bar);imag(A_bar) real(A_bar)];
XX = PHI_R.';
hh = 2*G;

for iii = 1:N

r = [real(x0_one(:,iii));imag(x0_one(:,iii))];
m = ones(2*M,1)/(2*M);
h_g = zeros(hh,g_num);
alph=zeros(g_num,1);

for jj=1:g_num
    
    h_R = XX * (r .* m);
    h_g(:,jj)=h_R;

    pred_1 = sign(PHI_R*h_R);
    er_1 = r.*pred_1;
    II= find(er_1 < 0);
    z_m = zeros(size(m));
    z_m(II,1) = ones(length(II),1);

    err_m = (((z_m.')*m)/(sum(m))) ;

    if err_m == 0
      alph_m = 1;
      alph(jj) = alph_m;
      break
    else
      alph_m = 0.5*log((1-err_m)/err_m);
      alph(jj) = alph_m;
    end

    m = m .* exp(-(alph_m*er_1));
    m = m/sum(m);

end

h_R = h_g*alph;
h_R_1 = h_R(1:G);
h_R_2 = h_R(1+G:end);

S_bar(:,iii)= h_R_1+1j*h_R_2;

end

if LL(gg) > K
 [S] = decrem_doa_onebyone(x0_one,A_bar,S_bar,LL(gg+1));   
else   
 S = S_bar;
end

V = S.';
end

DOA_rad = sort(tt);
DOA_deg = DOA_rad * (180/pi);

end



%% ALG 3 
function [S] = decrem_doa_onebyone(x0_one,A_bar,S_bar,K)

    yy = [real(x0_one);imag(x0_one)]; 
    ii = K+1;
    loc = zeros(ii,1);

    for jj=1:ii

        AA = A_bar;
        SS = S_bar;
        AA(:,jj)=[]; 
        SS(jj,:)=[];
        YY = AA*SS;
        YY_0 = sign([real(YY);imag(YY)]);
        
        YY_1 = yy.*YY_0;
        YY_1 = YY_1(:);

       loc(jj)=length(find(YY_1 < 0));
    end

       
     [~,ind] = min(loc);
     S_bar(ind,:)=[];

     S = S_bar;

end


