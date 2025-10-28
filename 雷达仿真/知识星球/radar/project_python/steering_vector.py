#%%

import numpy as np

def steering_vector(pos, ang, wavelength):
    """
    计算导向矢量
    :param pos: 位置矩阵，形状为 (3, N)，N 是阵元的数量，每个阵元的坐标用列向量表示
    :param ang: 方向角矩阵，可以是 1xM 或 2xM 矩阵，M 是输入信号的数量。
                如果是 2xM 矩阵，每列表示 [方位角; 俯仰角]。
    :param wavelength: 信号波长
    :return: 导向矢量，形状为 (N, M), 每一列是一个导向矢量
    """
    # 方位角
    azimuth = ang[0,:]
    # 俯仰角
    elevation = ang[1,:]
    # 目标方向的单位方向向量
    unit_direction_vector = np.array([
        np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth)),
        np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth)),
        np.sin(np.radians(elevation))
    ]) 
    
    # 信号到达不同阵元的距离差，参考阵元坐标 (0, 0, 0)
    distance = np.matmul(pos.T,unit_direction_vector)
    
    # 计算导向矢量
    sv = np.exp(1j * 2 * np.pi * distance / wavelength) 
    
    return sv 
#%%
if __name__ == '__main__':
    np.set_printoptions(precision=4)

    wavelength = 1.25e-2
    pos = np.array([[0, 1, 0], [0, 5, 0], [0, 2, 0], [0, 3, 0]]).T * 0.5 * wavelength
    ang = np.array([[10, 0], [20, 10] ]).T
    my_sv = steering_vector(pos, ang, wavelength)
    print(my_sv)

#%%
