function LDPC_produce(variable_edge_degree, variable_edge_weight, parity_edge_degree, parity_edge_weight, code_length, output_path)
% function produce_LDPC
% LDPC code
% remove girth-4 loops
% randomly select parity check
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%change edge weight to node weight & compute RATE, degree number,
%node number, edge number, parity number.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(~isempty(output_path))
    if(~exist(output_path, 'dir'))
        mkdir(output_path);
    end
end

variable_edge_weight = variable_edge_weight(:);
variable_edge_degree = variable_edge_degree(:);
parity_edge_weight   = parity_edge_weight(:);
parity_edge_degree   = parity_edge_degree(:);

variable_degree = variable_edge_degree;
variable_weight = variable_edge_weight./variable_edge_degree;
variable_weight = variable_weight/sum(variable_weight);

parity_degree = parity_edge_degree;
parity_weight = parity_edge_weight./parity_edge_degree;
parity_weight = parity_weight/sum(parity_weight);
Rate = 1-sum(parity_edge_weight./parity_edge_degree)/sum(variable_edge_weight./variable_edge_degree);

CodeLen = code_length-1;
N       = 0;
while(N~=code_length && N < code_length)
    var_number = floor(CodeLen*variable_weight);
    N = sum(var_number);
    CodeLen = CodeLen + 0.01;
end
if(N~=code_length)
    error('N~=code_length!');
end
degree_number = max(size(variable_degree));
edge_number = sum(var_number.*variable_edge_degree);
parity_number = ceil(edge_number*parity_edge_weight./parity_edge_degree);
max_parity_degree = max(parity_degree);
max_variable_degree = max(variable_degree);

N = sum(var_number);
L = sum(parity_number);
K = N-L;
display(['Rate = ' num2str(Rate) ', Var_number = ' num2str(N) ', parity_number = ' num2str(L) ', InfoLen = ' num2str(K)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate parity check matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
edge_per_degree= parity_number .* parity_edge_degree;
sum_edge_number= sum(edge_per_degree);
edge_bank      = zeros(1, sum_edge_number);
end_edge_idx   = int32(cumsum(edge_per_degree));
start_edge_idx = int32([1 end_edge_idx(1:(end-1))'+1]);
end_node_idx   = cumsum(parity_number);
start_node_idx = [1 end_node_idx(1:(end-1))'+1];
parity_set = zeros(1,L);
for i = 1:length(parity_edge_degree)
    index_set = start_node_idx(i):end_node_idx(i);
    edge_bank(start_edge_idx(i):end_edge_idx(i)) = index_set.' * ones(1,parity_edge_degree(i));
    parity_set(index_set) = parity_degree(i);
end

parity_set_origin = parity_set;
bsuccess = 0;
ntrial = 0;
max_trial = 100;
%edge_bank = int32(edge_bank); % convert to int
while(bsuccess==0 && ntrial< max_trial)
ntrial = ntrial + 1;
parity_set = parity_set_origin;
% random shuffling
for i=1:sum_edge_number
    temp_index = i-1+ceil(rand(1)*(sum_edge_number-i+1));%randi([i-1, sum_edge_number], 1, 1);
    temp = edge_bank(i);
    edge_bank(i) = edge_bank(temp_index);
    edge_bank(temp_index) = temp;
end
% check repetion and girth-4 loops
count = 1;
pointer = 0;
v2p = zeros(N,max_variable_degree);
v2p_sparse = sparse([1], [1], [0], L, N, N*max_variable_degree);
num_loop4 = 0;
for i = 1:degree_number
    for j = 1:var_number(i)
        if (sum_edge_number-pointer)<2
            break;
        end
        step = min(variable_edge_degree(i),sum_edge_number-pointer);
        tmp_edges = edge_bank(1+pointer:step+pointer);
        pointer = pointer+step;
        
        % remove the edges from the same parity node
        idx = [];
        for k = 1:length(tmp_edges)
            for t = k+1:length(tmp_edges)
                if tmp_edges(k)==tmp_edges(t)
                    parity_set(tmp_edges(k)) = parity_set(tmp_edges(k))-1;
                    idx = [idx t];
                    break;
                end
            end
        end
        tmp_edges(idx) = [];
        
        % remove girth-4 loops
        k = 1;
        while k<length(tmp_edges)
            [~,cols] = find(v2p_sparse(tmp_edges(k),:)~=0);
            t = k+1;
            while t<=length(tmp_edges)
                if isempty(find(v2p_sparse(tmp_edges(t),cols)~=0))
                    t = t+1;
                else
                    parity_set(tmp_edges(t)) = parity_set(tmp_edges(t))-1;
                    tmp_edges(t) = [];
                    num_loop4 = num_loop4+1;
                end
            end
            k = k+1;
        end

        % degree not less than 2
        while length(tmp_edges)<2
            tmp_edges = [tmp_edges edge_bank(pointer+1)];
            pointer = pointer+1;
            if length(tmp_edges) == 2
                if tmp_edges(1) == tmp_edges(2)
                    parity_set(tmp_edges(1)) = parity_set(tmp_edges(1))-1;
                    tmp_edges(2) = [];
                end
            end
            % girth-4 loop check
            k = 1;
            while k<length(tmp_edges)
                [~,cols] = find(v2p_sparse(tmp_edges(k),:)~=0);
                t = k+1;
                while t<=length(tmp_edges)
                    if isempty(find(v2p_sparse(tmp_edges(t),cols)~=0))
                        t = t+1;
                    else
                        parity_set(tmp_edges(t)) = parity_set(tmp_edges(t))-1;
                        tmp_edges(t) = [];
                        num_loop4 = num_loop4+1;
                    end
                end
                k = k+1;
            end
        end
        v2p(count,1:length(tmp_edges)) = tmp_edges;
        v2p_sparse(tmp_edges,count) = 1;
        count = count+1;
    end
end
bsuccess = isempty(find(parity_set<2));
end
if(bsuccess==0)
    error('Code Construction Failed! Please Regenerate.');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% output 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_new = count-1;
L_new = sum(parity_number);
K_new = N_new-L_new;
display(['Rate = ' num2str(Rate) ', OutVar_num = ' num2str(N_new) ', Outparity_num = ' num2str(L_new) ', OutInfL = ' num2str(K_new)]);
display(['num_loop4 = ' num2str(num_loop4) ', ntrail = ' num2str(ntrial)]);

%--------------------------------------------------------------------------
% The above parity matrix is stored from the view of variable nodes; change
% the storage from the view of check node. Junjie Ma, 19-03-2013
% Note: the indices follow a Matlab-style, i.e., starts from 1.
%--------------------------------------------------------------------------
p2v = zeros(max_parity_degree,L);
Parity_num = zeros(N,1);
for n = 1:N
    for j = 1:max_variable_degree
        if(v2p(n,j)>0)
            Parity_num(v2p(n,j)) = Parity_num(v2p(n,j)) + 1;
            p2v(Parity_num(v2p(n,j)),v2p(n,j)) = n;
        end
    end
end
fid=fopen('H_Fine.txt','w');
fprintf(fid, [repmat('%d\t',[1 max_parity_degree]), '\r\n'], p2v);
fclose(fid);

fid=fopen('LDPC_param.txt','w');
fprintf(fid,'<ParityCheckMatrix_RowNum(ParityCheckNum)>\n');
fprintf(fid,'%d\n', L);
fprintf(fid,'<CodedWordLength>\n');
fprintf(fid,'%d\n', max(p2v(:)));
fprintf(fid,'<ParityCheckMatrix_ColNum(MaxCNDDegreeNum)>\n');
fprintf(fid,'%d\n', max(parity_degree));
fprintf(fid,'<LDPCDecIterNum>\n');
fprintf(fid,'%d\n', 1);
fprintf(fid,'<Reset\\Yes\\No>\n');
fprintf(fid,'No\n');
fprintf(fid,'<AdaptiveIteration\\Yes\\No>\n');
fprintf(fid,'Yes\n');
fprintf(fid,'<Display\\Yes\\No>\n');
fprintf(fid,'No\n');
fprintf(fid,'<DispGap>\n');
fprintf(fid,'1\n');
fprintf(fid,'<HFileName>\n');
fprintf(fid,'H_Fine.txt\n');
fprintf(fid,'<IfFastEncode\\Yes\\No>\n');
fprintf(fid,'Yes\n');
fclose(fid);
%--------------------------------------------------------------------------
if(~isempty(output_path))
	movefile('H_Fine.txt', output_path);
	movefile('LDPC_param.txt', output_path);
end
end