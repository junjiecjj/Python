function vArray = helperVirtualArray(txArray, rxArray)
% Combine txArray and rxArray into a virtual array by convolving their
% apertures
    txElementPositions = getElementPosition(txArray);
    N = getNumElements(txArray);

    rxElementPositions = getElementPosition(rxArray);
    M = getNumElements(rxArray);

    vElemenetPositions = zeros(3, N*M);

    for m = 1:M
        vElemenetPositions(:, (m-1)*N+1:m*N) = rxElementPositions(:, m) + txElementPositions;
    end

    vArray = phased.ConformalArray('ElementPosition', vElemenetPositions);
end