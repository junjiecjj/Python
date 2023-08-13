import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from ldpcBitFlip import bitFlip

def dec2bin(num):
    return np.binary_repr(num, width=8)
vec_dec2bin = np.vectorize(dec2bin)

def pfunction(received, sigma=1):
    return 1/(1+np.exp(2*received/sigma**2))

def check2bit(qNeg, H_mask):
    shape = qNeg.shape# 3,7
    rPos,rNeg = np.zeros(shape),np.zeros(shape)
    qTmp = qNeg[H_mask].reshape(shape[0],4) #4为列重
    qTmp = 1-2*qTmp
    colProduct = np.product(qTmp, axis=1)
    rTmpPos = 0.5+0.5*np.divide(np.full(qTmp.shape, colProduct.reshape(shape[0],1)),qTmp)#
    rTmpNeg = 1-rTmpPos
    rPos[H_mask] = rTmpPos.flatten()
    rNeg[H_mask] = rTmpNeg.flatten()
    return rPos, rNeg

def bit2check(rPos, receivedPzero, neg=False):
    rPosMask = rPos == 0
    rPos[rPosMask] = 1
    rowProductPos = np.product(rPos,axis=0)
    rPostmp = np.divide(np.full(rPos.shape, rowProductPos), rPos)
    rPos[rPosMask] = 0
    rPostmp[rPosMask] = 0
    if not neg:
        qPostmp2 = np.multiply(1-receivedPzero, rPostmp)
    else:
        qPostmp2 = np.multiply(receivedPzero, rPostmp)
    return  rowProductPos, qPostmp2

def sumProduct(received, H, sigma, maxiter=10):
    #initial
    decoded = np.zeros(received.shape[1])
    receivedP = pfunction(received, sigma)
    receivedP = receivedP.reshape(-1, H.shape[1])
    #循环words
    for j in range(len(receivedP)):
        receivedPzero = receivedP[j]
        qNeg = np.multiply(receivedPzero,H) #3,7
        #qPos = np.multiply(np.full((qNeg.shape),1) - receivedPzero,H)

        #Pass information from check nodes to bit nodes
        H_mask= H!= 0
        k = np.zeros(qNeg.shape)
        for _ in range(maxiter):
            rPos, rNeg = check2bit(qNeg,H_mask)
            # bit nodes to check nodes
            rowProductPos, qPostmp2 = bit2check(rPos, receivedPzero)
            rowProductNeg, qNegtmp2 = bit2check(rNeg, receivedPzero, neg=True)

            k[H_mask] = np.divide(1,qPostmp2[H_mask]+qNegtmp2[H_mask])
            #qPos = np.multiply(k,qPostmp2)
            qNeg = np.multiply(k,qNegtmp2)

        QtmpPos = np.multiply(1-receivedPzero, rowProductPos)
        QtmpNeg = np.multiply(receivedPzero, rowProductNeg)
        K = np.divide(1,QtmpPos+QtmpNeg)
        QPos = np.multiply(K,QtmpPos)
        #QNeg = np.multiply(K,QtmpNeg)
        decoded[j*H.shape[1]:j*H.shape[1]+H.shape[1]] = QPos>0.5
    return decoded

A = np.array([[1,1,1,0],[0,1,1,1],[1,0,1,1]])#3,4
H = np.concatenate((A,np.eye(3)),axis=1)#3,7
G = np.concatenate((np.eye(4),A.T),axis=1)#4,7
Kdec = np.random.randint(0, 256,size=(1024))
Kbin = np.array([int(_) for _ in ''.join(vec_dec2bin(Kdec))])#8192
KbinRe = Kbin.reshape(-1, 4)

encoded = np.mod(np.matmul(KbinRe, G), 2)
encodedRe = encoded.reshape(1,-1)
modulated = encodedRe*2-1
snrs = list((range(-5,8)))
sigmas = [1/(10**(snr/10)) for snr in snrs]
bitErrRatio = np.zeros((4,len(snrs)))
for i,sigma in enumerate(sigmas):
    noise = sigma*np.random.randn(modulated.shape[1])
    received = modulated+noise
    #sum-product
    decodedSumP = sumProduct(received, H, sigma)
    bitErrRatioSumP = sum(sum(decodedSumP!=encodedRe))/encodedRe.shape[1]
    #bitFlip
    decodedBitFlip = bitFlip(received, H)
    bitErrRatioBitFlip = sum(sum(decodedBitFlip!=encodedRe))/encodedRe.shape[1]
    #硬判决
    bitErrRatioHard = sum(sum((received>0)!=encodedRe))/encodedRe.shape[1]
    bitErrRatio[:,i] = [np.log10(bitErrRatioSumP),np.log10(bitErrRatioBitFlip),
                        np.log10(bitErrRatioHard),np.log10(0.5*erfc(np.sqrt(10**(snrs[i]/10))))]

    print('snr',snrs[i],'\n硬判决误码率：',bitErrRatioHard,'\n比特翻转误码率：', bitErrRatioBitFlip,'\n和积译码误码率：',bitErrRatioSumP)

# 绘制图像
plt.plot(snrs, bitErrRatio[0], color='red',marker='o', label='sum product')
plt.plot(snrs, bitErrRatio[1], color='blue',marker='s', label='bit flip')
plt.plot(snrs, bitErrRatio[2], color='green',marker='^', label='hard descion')
#plt.plot(snrs, bitErrRatio[3], color='black', label='theoretical')
# 设置图像属性
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.title('SNR vs. BER')
plt.legend()

# 显示图像
plt.show()
