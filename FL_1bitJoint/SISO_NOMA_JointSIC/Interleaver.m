%生成N_u个长为n的随机交织序列
function inteleaver_matrix = Interleaver(N_u,n)
inteleaver_matrix = zeros(N_u,n);
for i=1:N_u

    inteleaver_matrix(i,:) = randperm(n);
%     perm=1:n;
%     for j=1:n
%         r=randi(n-j+1); %pseudorandom scalar integer between 1 and m.
%         inteleaver_matrix(i,j)=perm(r);
%         perm(r)=[]; %removing index to avoid repetition
%     end
end
end