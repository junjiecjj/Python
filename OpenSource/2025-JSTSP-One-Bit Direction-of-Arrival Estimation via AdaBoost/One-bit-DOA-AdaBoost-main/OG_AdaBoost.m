function [DOA_deg,DOA_rad] = OG_AdaBoost(x0_one,K,M,N,g_num,G)


theta = linspace(-pi/2, pi/2, G); 
idxR = (0:(M-1))';
d_lambda = 0.5;

A_bar = zeros(M,G);
for ii = 1:G
    A_bar(:,ii) = exp(-1i*2*pi*d_lambda*sin(theta(ii))*idxR);  
end

S_bar = zeros(G,N);

PHI_R = [real(A_bar) -imag(A_bar);imag(A_bar) real(A_bar)];
XX = PHI_R.';
hh = 2*G;

for iii = 1:N
        
r = [real(x0_one(:,iii));imag(x0_one(:,iii))];
m = ones(2*M,1)/(2*M); 
m = m/sum(m);


h_g = zeros(hh,g_num);
alph=zeros(g_num,1);

for jj=1:g_num
    
    h_R = XX * (r .* m);
    h_g(:,jj) = h_R;

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
    elseif err_m > 0.5
      h_R = -h_R;
      pred_1 = sign(PHI_R*h_R);
      er_1 = r.*pred_1;
      II= find(er_1 < 0);
      z_m = zeros(size(m));
      z_m(II,1) = ones(length(II),1);

      err_m = (((z_m.')*m)/(sum(m)));
      alph_m = 0.5*log((1-err_m)/err_m);
      alph(jj) = alph_m;
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
S_bar(:,iii) = (S_bar(:,iii))/norm(S_bar(:,iii)); 

end


s_til = zeros(G,1);

for iii=1:G
    s_til(iii,1) = norm(S_bar(iii,:));
end
[~,LOCS]= findpeaks(s_til,'SortStr','descend','NPeaks',K);

DOA_rad = sort(theta(LOCS).');
DOA_deg = DOA_rad * (180/pi);

end