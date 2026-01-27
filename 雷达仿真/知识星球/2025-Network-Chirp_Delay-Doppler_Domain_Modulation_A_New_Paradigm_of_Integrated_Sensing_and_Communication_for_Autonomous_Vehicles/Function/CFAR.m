function X2_Detected = CFAR(dim, X2, CFARset, SNRdB, flag)
%% OS-CFAR
% input
%       dim                 1 or 2, dimension of X2
%       X2                  X.*conj(X)
%       CFARset             setting of CFAR
% output
%       X2_Detected         either 0 or 1, the same size that of X2
% 
% time:2023.9.27
%% Determine the threshold
% vec = rand(1,100);
% n = 21;
% findNthSmallest(vec, n)
% tmp = sort(vec,'ascend');
% tmp(21)
quantile = 95/100;
factor = 1;
% in order to avoid target whose reflect energy is low being lost
% ThreFactor = 1/100000;% for RT
% ThreFactor = 1/100;% for CT
if strcmp(flag, 'OS')
    if dim == 1
        Ntot = length(X2);
        Nguard = CFARset.Nguard;
        Nrefer = CFARset.Nrefer;
        Thres = zeros(1, Ntot);   
        Pfa = CFARset.Pfa;  
        N = 2*Nrefer;
        alpha = N*(Pfa^(-1/N)-1);
        for i_index = 0:Ntot-1
            tmp = sort([X2(mod(i_index-Nguard-Nrefer:i_index-Nguard-1, Ntot)+1) ...
                        X2(mod(i_index+Nguard+1:i_index+Nguard+Nrefer, Ntot)+1)]);
            sigma2_est = tmp(floor(N*quantile));
            Thres(i_index+1) = alpha*sigma2_est;
        end
    elseif dim == 2    
        Nrow = size(X2, 1);
        Ncol = size(X2, 2);
        Nguard = CFARset.Nguard;
        Nrefer = CFARset.Nrefer;
        Thres = zeros(Nrow, Ncol);   
        Pfa = CFARset.Pfa;    
        N = (2*Nrefer(1)+2*Nguard(1)+1)*(2*Nrefer(2)+2*Nguard(2)+1) - (2*Nguard(1)+1)*(2*Nguard(2)+1);
        alpha = N*(Pfa^(-1/N)-1);
        parfor i_row = 0:Nrow-1
            tmp_parfor = zeros(1, Ncol);
            for i_col = 0:Ncol-1
                 tmp = [reshape(    X2(mod(i_row-Nguard(2)-Nrefer(2) : i_row+Nguard(2)+Nrefer(2), Nrow) + 1, ...
                                    mod(i_col-Nguard(1)-Nrefer(1) : i_col-Nguard(1)-1, Ncol) +1), ...
                                1, (2*Nrefer(2)+2*Nguard(2)+1)*Nrefer(1)) ...
                                ...
                        reshape(    X2(mod(i_row-Nguard(2)-Nrefer(2) : i_row+Nguard(2)+Nrefer(2), Nrow) + 1, ...
                                    mod(i_col+Nguard(1)+1 : i_col+Nguard(1)+Nrefer(1), Ncol) +1), ...
                                1, (2*Nrefer(2)+2*Nguard(2)+1)*Nrefer(1)) ...
                                ...
                        reshape(    X2(mod(i_row-Nguard(2)-Nrefer(2) : i_row-Nguard(2)-1, Nrow) +1, ...
                                    mod(i_col-Nguard(1):i_col+Nguard(1), Ncol) +1), ...
                                1, Nrefer(2)*(2*Nguard(1)+1)) ...
                                ...
                        reshape(    X2(mod(i_row+Nguard(2)+1 : i_row+Nguard(2)+Nrefer(2), Nrow) +1, ...
                                    mod(i_col-Nguard(1):i_col+Nguard(1), Ncol) +1), ...
                                1, Nrefer(2)*(2*Nguard(1)+1))];
                sigma2_est = findNthSmallest(tmp, floor(N*quantile));
                tmp_parfor(i_col+1) = alpha*sigma2_est;
    %             Thres(i_row+1, i_col+1) = alpha*ThreFactor*Nrow*Ncol*sigma2_est;
            end
            Thres(i_row+1,:) = tmp_parfor;
        end
    else 
        error('Dimension of X2 is neither 1 nor 2!');
    end
else% strcmp(flag, 'CA')
    if dim == 1
        Ntot = length(X2);
        Nguard = CFARset.Nguard;
        Nrefer = CFARset.Nrefer;
        Thres = zeros(1, Ntot);   
        Pfa = CFARset.Pfa;  
        N = 2*Nrefer;
        alpha = N*(Pfa^(-1/N)-1);
        for i_index = 0:Ntot-1
            sigma2_est = mean([X2(mod(i_index-Nguard-Nrefer:i_index-Nguard-1, Ntot)+1) ...
                        X2(mod(i_index+Nguard+1:i_index+Nguard+Nrefer, Ntot)+1)]);
            Thres(i_index+1) = alpha*factor*sigma2_est;
        end
    elseif dim == 2    
        Nrow = size(X2, 1);
        Ncol = size(X2, 2);
        Nguard = CFARset.Nguard;
        Nrefer = CFARset.Nrefer;
        Thres = zeros(Nrow, Ncol);   
        Pfa = CFARset.Pfa;    
        N = (2*Nrefer(1)+2*Nguard(1)+1)*(2*Nrefer(2)+2*Nguard(2)+1) - (2*Nguard(1)+1)*(2*Nguard(2)+1);
        alpha = N*(Pfa^(-1/N)-1);
        parfor i_row = 0:Nrow-1
            tmp_parfor = zeros(1, Ncol);
            for i_col = 0:Ncol-1
                 tmp = [reshape(    X2(mod(i_row-Nguard(2)-Nrefer(2) : i_row+Nguard(2)+Nrefer(2), Nrow) + 1, ...
                                    mod(i_col-Nguard(1)-Nrefer(1) : i_col-Nguard(1)-1, Ncol) +1), ...
                                1, (2*Nrefer(2)+2*Nguard(2)+1)*Nrefer(1)) ...
                                ...
                        reshape(    X2(mod(i_row-Nguard(2)-Nrefer(2) : i_row+Nguard(2)+Nrefer(2), Nrow) + 1, ...
                                    mod(i_col+Nguard(1)+1 : i_col+Nguard(1)+Nrefer(1), Ncol) +1), ...
                                1, (2*Nrefer(2)+2*Nguard(2)+1)*Nrefer(1)) ...
                                ...
                        reshape(    X2(mod(i_row-Nguard(2)-Nrefer(2) : i_row-Nguard(2)-1, Nrow) +1, ...
                                    mod(i_col-Nguard(1):i_col+Nguard(1), Ncol) +1), ...
                                1, Nrefer(2)*(2*Nguard(1)+1)) ...
                                ...
                        reshape(    X2(mod(i_row+Nguard(2)+1 : i_row+Nguard(2)+Nrefer(2), Nrow) +1, ...
                                    mod(i_col-Nguard(1):i_col+Nguard(1), Ncol) +1), ...
                                1, Nrefer(2)*(2*Nguard(1)+1))];
                sigma2_est = mean(tmp);
                tmp_parfor(i_col+1) = alpha*factor*sigma2_est;  
            end
            Thres(i_row+1,:) = tmp_parfor;
        end
    else 
        error('Dimension of X2 is neither 1 nor 2!');
    end
end
%% Compare the threshold with X2
X2_Detected = X2>Thres;
%% Debug
% figure;mesh(X2)
% figure
% if dim == 2
%     mesh(X2_Detected)
% %     image(X2_Detected,'CDataMapping','scaled')
%     colorbar
% else
%     plot(X2_Detected)
% end
% figure;mesh(X2)
% figure;mesh(Thres)
% figure;mesh(X2_Detected)
end

%% From ChatGPT
function nth_smallest = findNthSmallest(vec, n)
    % Return empty if n is out of vector bounds
    if n < 1 || n > numel(vec)
        nth_smallest = [];
        return;
    end

    % Create a max heap and insert the first n elements of the vector into it
    heap = vec(1:n);
    heap = heap(:);  % Ensure heap is a column vector
    heap = heap';

    % Convert the heap to a max heap
    heap = heap(end:-1:1);
    for i = floor(n/2):-1:1
        heap = maxHeapify(heap, i, n);
    end

    % Process the remaining elements
    for i = n+1:numel(vec)
        if vec(i) < heap(1)
            heap(1) = vec(i);
            heap = maxHeapify(heap, 1, n);
        end
    end

    nth_smallest = heap(1);
end

function heap = maxHeapify(heap, i, n)
    left = 2 * i;
    right = 2 * i + 1;
    largest = i;

    if left <= n && heap(left) > heap(i)
        largest = left;
    end

    if right <= n && heap(right) > heap(largest)
        largest = right;
    end

    if largest ~= i
        temp = heap(i);
        heap(i) = heap(largest);
        heap(largest) = temp;
        heap = maxHeapify(heap, largest, n);
    end
end
