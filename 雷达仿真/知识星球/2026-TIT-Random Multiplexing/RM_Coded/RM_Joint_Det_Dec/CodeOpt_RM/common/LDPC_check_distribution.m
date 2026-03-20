function [dv_out, b_out, dc_out, d_out] = LDPC_check_distribution(H_file_name)

PCM = load(H_file_name);

Nonzero = (PCM ~= 0);
[M, ~]  = size(PCM);        % number of check nodes
N       = max(PCM(:));      % number of variable nodes
E       = sum(Nonzero(:))  % number of edges

RowWeight = sum(Nonzero, 2);
ColWeight = zeros(1, N);
Cols = sort(PCM(:), 'ascend');
[b, m, n] = unique(Cols);
if(Cols(1)~=0)
    ColWeight(b) = [m(1), m(2:end).'-m(1:(end-1)).'];
else
    b = b(b~=0);
    ColWeight(b) = m(2:end)-m(1:(end-1));
end
for col=1:N
	ColWeight(col) = length(find(PCM(:)==col));
end

if(sum(ColWeight)~=sum(RowWeight))
	error('sum(ColWeight)~=sum(RowWeight)!');
end

Lambda_index = min(ColWeight):max(ColWeight);
Lambda  = zeros(size(Lambda_index));
lambda  = zeros(size(Lambda_index));
for cnt = 1:length(Lambda_index)
	num_nodes = length(find(ColWeight==Lambda_index(cnt)));
	Lambda(cnt) = num_nodes / N;
	lambda(cnt) = num_nodes * Lambda_index(cnt) / E;
end

Rho_index = min(RowWeight):max(RowWeight);
Rho       = zeros(size(Rho_index));
rho       = zeros(size(Rho_index));
for cnt = 1:length(Rho_index)
	num_nodes = length(find(RowWeight==Rho_index(cnt)));
	Rho(cnt)  = num_nodes / M;
	rho(cnt)  = num_nodes * Rho_index(cnt) / E;
end

dv_out = Lambda_index(lambda>0.0);
b_out  = lambda(lambda>0.0);
dc_out = Rho_index(rho>0.0);
d_out  = rho(rho>0.0);
end