import numpy as np

def sine_wave(f,overSampRate,phase,nCyl):
    """
    Generate sine wave signal with the following parameters
    Parameters:
        f : frequency of sine wave in Hertz
        overSampRate : oversampling rate (integer)
        phase : desired phase shift in radians
        nCyl : number of cycles of sine wave to generate
    Returns:
        (t,g) : time base (t) and the signal g(t) as tuple    
    Example:
        f=10; overSampRate=30;
        phase = 1/3*np.pi;nCyl = 5;
        (t,g) = sine_wave(f,overSampRate,phase,nCyl)
    """
    fs=overSampRate*f # sampling frequency
    t = np.arange(0,nCyl*1/f-1/fs,1/fs) # time base    
    g=np.sin(2*np.pi*f*t+phase) # replace with cos if a cosine wave is desired    
    return (t,g) # return time base and signal g(t) as tuple

def square_wave(f,overSampRate,nCyl):
    """
    Generate square wave signal with the following parameters
    Parameters:
        f : frequency of square wave in Hertz
        overSampRate : oversampling rate (integer)
        nCyl : number of cycles of square wave to generate
    Returns:
        (t,g) : time base (t) and the signal g(t) as tuple    
    Example:
        f=10; overSampRate=30;nCyl = 5;
        (t,g) = square_wave(f,overSampRate,nCyl)
    """
    fs=overSampRate*f # sampling frequency
    t = np.arange(0,nCyl*1/f-1/fs,1/fs) # time base    
    g=np.sign(np.sin(2*np.pi*f*t)) # replace with cos if a cosine wave is desired    
    return (t,g) # return time base and signal g(t) as tuple

def rect_pulse(A,fs,T):
    """
    Generate isolated rectangular pulse with the following parameters
    Parameters:
        A : amplitude of the rectangular pulse
        fs : sampling frequency in Hz
        T : duration of the pulse in seconds
    Returns:
        (t,g) : time base (t) and the signal g(t) as tuple 
    Example:
        A = 1; fs=500;T = 0.2;
        (t,g) = rect_pulse(A,fs,T)
    """
    t=np.arange(-0.5,0.5,1/fs) # time base
    rect=(t >-T/2) * (t<T/2) + 0.5*(t==T/2) + 0.5*(t==-T/2)
    g = A*rect
    return (t,g)

def gaussian_pulse(fs,sigma):
    """
    Generate isolated Gaussian pulse with the following parameters
    Parameters:
        fs : sampling frequency in Hz
        sigma : pulse width in seconds
    Returns:
        (t,g) : time base (t) and the signal g(t) as tuple
    Example:
        fs = 80; sigma = 0.1;
        (t,g) = gaussian_pulse(fs,sigma)
    """
    t=np.arange(-0.5,0.5,1/fs) # time base
    g=1/(np.sqrt(2*np.pi)*sigma)*(np.exp(-t**2/(2*sigma**2)))
    return(t,g) # return time base and signal g(t) as tuple