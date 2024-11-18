from .base import Denoiser

def pad(a,size):
    out=np.zeros(size)
    out[:len(a)] = a
    return out

def LMMSE_denoiser(r_2, gamma_2, noise_precision=0,A=0,y=0):
    # global variables A, y, noise_precision are assumed to be defined
    
    if type(A) is tuple:
        # SVD provided
        U, S, VT = A
        n,p=U.shape[0],VT.shape[1]
        S_v = pad(S,VT.shape[0]) # compatible with V
        S_u = pad(S,U.shape[1]) # compatible with U
        alpha = gamma_2 * sum(1/(noise_precision * S_v**2+ gamma_2 ))/p
        eta_2 = gamma_2/alpha
        matrix = (VT.T/(noise_precision*S_v**2+gamma_2))
        xhat_2 = matrix @( noise_precision *(pad(((y.T @U) * S_u).T,(p,1)))  +  gamma_2*VT@r_2 )
    else:
        matrix = inv(noise_precision * A.T @ A + gamma_2 * np.eye(A.shape[1]))
        
        xhat_2 = matrix @ (noise_precision * A.T@ y + gamma_2 * r_2)
        alpha = gamma_2 * np.trace(matrix)/matrix.shape[1]
        eta_2 = gamma_2/alpha
    
    return xhat_2, eta_2, alpha

class Lmmse(Denoiser):
    def __init__(self, *args, **kwargs):
        super().__init__(LMMSE_denoiser, *args, **kwargs)
        

