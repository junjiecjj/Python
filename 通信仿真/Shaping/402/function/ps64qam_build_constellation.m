function [constellation, bitLabels, axisLevels] = ps64qam_build_constellation()
    % 构造Gray映射的64QAM星座
    % 输出：
    % constellation : 64x1 复数星座点
    % bitLabels     : 64x6 比特标签，每行对应一个星座点
    % axisLevels    : 8x1 星座轴电平

    axisLevels = [-7; -5; -3; -1; 1; 3; 5; 7];

    % Gray标签 0~7 转换到二进制顺序，再映射到8PAM幅度
    grayIdx = (0:7).';
    binIdx  = ps64qam_gray2bin(grayIdx);
    mappedLevels = axisLevels(binIdx + 1);

    constellation = zeros(64,1);
    bitLabels     = zeros(64,6);

    n = 1;
    for iIdx = 0:7
        bitsI = ps64qam_int2bits(iIdx, 3);
        for qIdx = 0:7
            bitsQ = ps64qam_int2bits(qIdx, 3);
            constellation(n) = mappedLevels(iIdx + 1) + 1i * mappedLevels(qIdx + 1);
            bitLabels(n,:) = [bitsI bitsQ];
            n = n + 1;
        end
    end
end
