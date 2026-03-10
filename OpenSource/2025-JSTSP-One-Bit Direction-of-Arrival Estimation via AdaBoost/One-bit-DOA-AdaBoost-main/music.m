function [DOA_deg,DOA_rad] = music(G,K,M)

theta=-pi/2:(pi/360):(pi/2);
idxR = (0:(M-1))';
gg=zeros(size(theta));
for ii=1:length(theta)
    aaa=exp(-1i*pi*sin(theta(ii))*idxR);
    gg(ii)=1/abs((aaa')*G*(G')*aaa);    
end

[~,LOCS]= findpeaks(gg,'SortStr','descend','NPeaks',K);

if length(LOCS) < K
    LOCS=[LOCS LOCS LOCS];
    LOCS=LOCS(1:K);
end
DOA_rad = sort(theta(LOCS).');
DOA_deg = DOA_rad * (180/pi); 


end