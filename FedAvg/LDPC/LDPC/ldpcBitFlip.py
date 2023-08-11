





import numpy as np

def dec2bin(num):
    return np.binary_repr(num, width = 8)

def match_cols(H, C):
    #在H中匹配C校验和，返回匹配到的列数，该列对应需要翻转的比特
    H_tmp = np.hstack((H, np.zeros((H.shape[0], 1))))*2 - 1
    C_tmp = C*2 - 1
    maxMat = np.matmul(C_tmp, H_tmp)
    flip = np.argmax(maxMat, axis=1)
    return flip

def flip_function(flip, modulatedRe):
    #根据翻转位进行翻转
    flip_mask = flip!=max(flip)
    flip_mask = flip_mask.squeeze()
    modulatedRe[flip_mask, flip[flip_mask].squeeze()] = np.logical_not(modulatedRe[flip_mask, flip[flip_mask].squeeze()])
    return modulatedRe

def bitFlip(received, H):
    #received: 接收信号
    #H：校验矩阵

    ## 接收序列硬判决
    demodulated = np.array(received > 0)
    demodulatedRe = demodulated.reshape(-1, H.shape[1])  # (32768, 7)
    checkSum = np.mod(np.matmul(demodulatedRe, H.T), 2)  # (32768, 3)
    flip = match_cols(H, checkSum)
    decoded = flip_function(flip, demodulatedRe.copy())
    return decoded.flatten()

if __name__ == '__main__':
    vec_dec2bin = np.vectorize(dec2bin)

    P = np.array([[1,1,1,0],[0,1,1,1],[1,0,1,1]])    ## n - k = 3, k = 4
    H = np.concatenate((P, np.eye(3)), axis=1)       ## n - k = 3, n = 7
    G = np.concatenate((np.eye(4), P.T), axis=1)     # k = 4, n = 7
    Kdec = np.random.randint(0, 256, size=(16384))
    Kbin = np.array([int(_) for _ in ''.join(vec_dec2bin(Kdec))])  #  131072 = 16384 * 8
    KbinRe = Kbin.reshape(-1, 4)                     # (32768, 4)

    encoded = np.mod(np.matmul(KbinRe, G), 2)        # (32768, 7)
    encodedRe = encoded.reshape(1,-1)                # (1, 229376)
    modulated = encodedRe*2-1                        # (1, 229376)
    noise = 0.5*np.random.randn(modulated.shape[1])  # (1, 229376)
    received = modulated+noise                       # (1, 229376)

    #译码算法 比特翻转算法
    decoded = bitFlip(received, H)
    #译码算法 校验和算法
