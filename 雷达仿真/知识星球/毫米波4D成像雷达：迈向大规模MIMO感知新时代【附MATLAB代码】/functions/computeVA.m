function VA = computeVA(TX, RX)

VA = zeros(size(TX,1) * size(RX,1), 2);
counter = 0;

for i = 1:size(TX,1)
    for j = 1:size(RX,1)
        counter = counter + 1;
        VA(counter,:) = TX(i,:) + RX(j,:);
    end
end

end