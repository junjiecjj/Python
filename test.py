
import numpy as np
import matplotlib.pyplot as plt

NSTEPS = 8

N = 16
dVsLambda = 0.5

n = np.arange(0,N,1)
k=0

# beam 方向与天线的夹角，天线是从上到下的方向看， beam 是从天线射向 UE 方向看，
#  这两个射线的夹角
#          |
#          |
#          |
#          |
#          |
#          |
#          |\
#          | \                      s e^{-j (N-1) phi}             s e^{j (N-1) theta}
#          |  \
#          |   \
#          |    \
#          |     \                        ...                       ...
#          |      \
#          |       \                      s e^{-j 2 phi}           s e^{j 2 theta}
#          |        \
#          |         \
#          | beamangle\                   s e^{-j phi}             s e^{j theta}
#          |           \
#          |            \                  s                         s
#
#  负相位，是把波形向前推， phi = 2 pi d/lambda cos(beamAngle), cos 是递减函数
#  beamAngle 越小，则 phi越大，则把波形向前移动的越大
#  当 beamAngle 从 0 逐渐增大到 PI/2，cos(beamAngle) 从 1 递减到 0，所以，波形向前
#  推的相位越来越小，则 beam 方向逆时针旋转
#  当 beamAngle 从 PI/2 逐渐增大到 PI，cos(beamAngle) 从 0 递减到 -1，所以 -phi 是从0递增到 1，
#  波形向后拉的相位越来越大，则 beam 方向继续逆时针旋转
for k in range(0,NSTEPS):
    beamAngle = k*np.pi/NSTEPS
    phi = 2*np.pi*dVsLambda*np.cos(beamAngle)

    phaseRotateVector = np.exp(-1j*phi*n)

    thetaAll = np.arange(0,2*np.pi,0.01)
    beamAmp = []
    for theta in thetaAll:
        ePhi = np.exp(1j*2*np.pi*dVsLambda*np.cos(theta)*n)
        BeamAmpForOneTheta = np.dot(phaseRotateVector,ePhi)
        beamAmp.append(BeamAmpForOneTheta)
    beamAmp = np.array(beamAmp)
    fig = plt.figure()
    ax = plt.subplot(111, polar=True)
    ax.plot(thetaAll, abs(beamAmp))
    ax.vlines(beamAngle,0,16,'r')
    ax.grid(True)
    ax.set_theta_offset(-np.pi/2)
    fig.suptitle(str(k) + "*PI/" + str(NSTEPS))



