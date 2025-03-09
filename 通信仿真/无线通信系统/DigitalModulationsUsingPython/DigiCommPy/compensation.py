"""
Module: DigiCommPy.compensation.py

@author: Mathuranathan Viswanathan
Created on Sep 6, 2019
"""
import numpy as np
from numpy import mean,real,imag,sign,abs,sqrt,sum

def dc_compensation(z):
    """
    Function to estimate and remove DC impairments in the IQ branch
    Parameters:
        z: DC impaired signal sequence (numpy format)
    Returns:
        v: DC removed signal sequence    
    """
    iDCest=mean(real(z)) # estimated DC on I branch
    qDCest=mean(imag(z)) # estimated DC on I branch
    v=z-(iDCest+1j*qDCest) # remove estimated DCs
    return v

def blind_iq_compensation(z):
    """
    Function to estimate and compensate IQ impairments for the
    single-branch IQ impairment model
    Parameters:
        z: DC impaired signal sequence (numpy format)
    Returns:
        y: IQ imbalance compensated signal sequence 
    """
    I=real(z);Q=imag(z)
    theta1=(-1)*mean(sign(I)*Q)
    theta2=mean(abs(I)); theta3=mean(abs(Q))
    c1=theta1/theta2
    c2=sqrt((theta3**2-theta1**2)/theta2**2)
    return I +1j*(c1*I+Q)/c2

class PilotEstComp():
    # Class: PilotEstComp (Pilot based estimation and compensation)
    # Attribute definitions:
    #    self.impObj: reference to the object implementing the impairment model
    #    self.Kest : estimated gain imbalance
    #    self.Pest : estimated phase mismatch
    #    self.preamble : time domain representation of IEEE 802.11a defined long preamble
    def __init__(self,impObj): # constructor
        self.impObj = impObj
        self.Kest = 1
        self.Pest = 0
        # Length 64 - long preamble as defined in the IEEE 802.11a (frequency domain representation)
        preamble_freqDom = np.array([0,0,0,0,0,0,1,1,
                                            -1,-1,1,1,-1,1,-1,1,\
                                            1,1,1,1,1,-1,-1,1,1,\
                                            -1,1,-1,1,1,1,1,0,1,\
                                            -1,-1,1,1,-1,1,-1,1,\
                                            -1,-1,-1,-1,-1,1,1,\
                                            -1,-1,1,-1,1,-1,1,1,\
                                            1,1,0,0,0,0,0])
        from scipy.fftpack import ifft
        self.preamble=ifft(preamble_freqDom,n = 64)
        
    def pilot_est(self):
        """
        IQ imbalance estimation using Pilot transmission
        Computes:
            Kest -  estimated gain imbalance
            Pest - estimated phase mismatch
        """
        # send known preamble through the impairments model
        r_preamb = self.impObj.receiver_impairments(self.preamble)
        
        # remove DC imbalance before IQ imbalance estimation
        z_preamb= r_preamb - (mean(real(r_preamb)) + 1j* mean(imag(r_preamb)))
        # IQ imbalance estimation
        I=real(z_preamb); Q=imag(z_preamb)
        self.Kest = sqrt(sum((Q*Q))/sum(I*I)) # estimated gain imbalance
        self.Pest = sum(I*Q)/sum(I*I) # estimated phase mismatch
    
    def pilot_iqImb_compensation(self,d):
        """
        Function to compensate IQ imbalance during the data transmission
        Parameters:
            d : The impaired received complex signal sequence
        Returns:
            w : IQ imbalance compensated complex signal sequence
        Usage:
            from compensation import PilotEstComp
            pltEstCompObj = PilotEstComp(impObj) #initialize
            pltEstCompObj.pilot_iqImb_compensation(d) #call function
        """
        # estimate the Kest, Pest for the given model using pilot transmission
        self.pilot_est()
        d_dcRemoved = d - (mean(real(d)) + 1j* mean(imag(d)))
        I=real(d_dcRemoved); Q=imag(d_dcRemoved)
        wi= I;
        wq = (Q - self.Pest*I)/sqrt(1-self.Pest**2)/self.Kest
        return wi + 1j*wq