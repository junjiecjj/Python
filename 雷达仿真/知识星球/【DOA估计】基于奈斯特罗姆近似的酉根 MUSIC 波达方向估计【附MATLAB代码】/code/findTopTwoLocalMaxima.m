function [rows,cols] = findTopTwoLocalMaxima(A)
    % Find local maxima using imregionalmax
    localMax = imregionalmax(A);
    
    % Extract the values and their linear indices
    maximaValues = A(localMax);
    maximaIndices = find(localMax);
    
    % Sort the maxima values and get indices of top two
    [sortedValues, sortIdx] = sort(maximaValues, 'descend');
    
    topTwoValues = sortedValues(1:min(2, end));
    topTwoIndices = maximaIndices(sortIdx(1:min(2, end)));
    
    % Convert linear indices to subscripts (row, col)
    [rows, cols] = ind2sub(size(A), topTwoIndices);
end