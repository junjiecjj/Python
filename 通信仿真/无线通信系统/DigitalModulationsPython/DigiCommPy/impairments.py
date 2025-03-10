"""
Module: DigiCommPy.impairments.py

@author: Mathuranathan Viswanathan
Created on Sep 5, 2019
"""
import numpy as np
from numpy import sin,cos,pi

class ImpairmentModel():
    # Class: ImpairmentModel
    # Attribute definitions:
    #    self.g : gain mismatch between I,Q branches
    #    self.phi : phase mismatch of the local oscillators(in degrees)
    #    self.dc_i : DC bias on I branch
    #    self.dc_q : DC bias on Q branch
    def __init__(self,g=1,phi=0,dc_i=0,dc_q=0): # constructor
        self.g    = g
        self.phi  = phi
        self.dc_i = dc_i
        self.dc_q = dc_q
        
    def receiver_impairments(self,r):
        """
        Add receiver impairments to the IQ branches. Introduces DC and IQ
        imbalances between the inphase and quadrature components of the
        complex baseband signal r. 
        Parameters:
            r : complex baseband signal sequence to impair
        Returns:
            z : impaired signal sequence
        """
        if isinstance(r,list):
            r = np.array(r) #convert to numpy array if in list format
        
        k = self.__iq_imbalance(r)
        z = self.__dc_impairment(k)
        return z
        
    def __iq_imbalance(self,r): #private member function
        """
        Add IQ imbalance impairments in a complex baseband. Introduces IQ imbalance
        and phase mismatch between the inphase and quadrature components of the
        complex baseband signal r. 
        Parameters:
            r : complex baseband signal sequence to impair
        Returns:
            z : impaired signal sequence
        """
        Ri=np.real(r); Rq=np.imag(r);
        Zi= Ri # I branch
        Zq= self.g*(-sin(self.phi/180*pi)*Ri + cos(self.phi/180*pi)*Rq) # Q branch crosstalk
        return Zi+1j*Zq
    
    def __dc_impairment(self,r): #private member function
        """
        Add DC impairments in a complex baseband model.Introduces DC imbalance
        between the inphase and quadrature components of the complex baseband
        signal r.
        Parameters:
            r: complex baseband signal sequence to impair
        Returns:
            z : impaired signal sequence
        """
        return r + (self.dc_i+1j*self.dc_q)