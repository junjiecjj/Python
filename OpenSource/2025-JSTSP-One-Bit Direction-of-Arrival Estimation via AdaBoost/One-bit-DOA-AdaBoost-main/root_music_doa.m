
function [DOA_deg,DOA_rad] = root_music_doa(G,M,K)

Ar = zeros(2*M-1,1);
for i = 1:(M-K)
   Ar = Ar + conv(G(M:-1:1,i),conj(G(:,i))); 
end
fest = zeros(K,1); 
r_A = roots(Ar);

distinct_roots_num = 0;
unit_roots_indx = find(abs(r_A) == 1);
unit_roots_num = length(unit_roots_indx);
if (unit_roots_num~=0)
    unit_roots = r_A(unit_roots_indx);
    distinct_roots_indx = 1:2:unit_roots_num;
    distinct_roots_num = length(distinct_roots_indx);
    fest(1:distinct_roots_num) = angle(unit_roots(distinct_roots_indx)); 
end
[i_min] = find(abs(r_A) < 1);
r_A_min = r_A(i_min);
freqsroot_MUSIC = angle(r_A_min);
[~,index] = sort(abs((abs(r_A_min) -1)));
fest((distinct_roots_num+1):K) = freqsroot_MUSIC(index(1:(K-distinct_roots_num)));  % DOA estimates


[DOA_rad,~] = sort(asin(fest/(pi)));
DOA_deg = DOA_rad * (180/pi);

