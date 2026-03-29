function binVal = ps64qam_gray2bin(grayVal)
% Gray码转二进制
grayVal = uint32(grayVal(:));
binVal  = grayVal;

shiftVal = bitshift(binVal, -1);
while any(shiftVal > 0)
    binVal   = bitxor(binVal, shiftVal);
    shiftVal = bitshift(shiftVal, -1);
end

binVal = double(binVal);
end
