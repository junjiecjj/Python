'''
Test scripts described in Chapter 1

@author: Mathuranathan Viswanathan
Created on Jul 16, 2019
'''
def sine_wave_demo():
    """
    Simulate a sinusoidal signal with given sampling rate    
    """
    import numpy as np
    import matplotlib.pyplot as plt # library for plotting
    from signalgen import sine_wave # import the function
    
    f = 10 #frequency = 10 Hz
    overSampRate = 30 #oversammpling rate
    phase = 1/3*np.pi #phase shift in radians
    nCyl = 5 # desired number of cycles of the sine wave
    (t,g) = sine_wave(f,overSampRate,phase,nCyl) #function call 
    
    plt.plot(t,g) # plot using pyplot library from matplotlib package
    plt.title('Sine wave f='+str(f)+' Hz') # plot title 
    plt.xlabel('Time (s)') # x-axis label
    plt.ylabel('Amplitude') # y-axis label
    plt.show() # disply the figure

def scipy_square_wave():
    """
    Generate a square wave with given sampling rate
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal
    
    f = 10 # f = 10Hz 
    overSampRate = 30 # oversampling rate
    nCyl = 5 # number of cycles to generate
    
    fs = overSampRate*f # sampling frequency
    t = np.arange(start=0,stop=nCyl*1/f,step=1/fs)# time base
    g = signal.square(2 * np.pi * f * t, duty = 0.2)
    plt.plot(t,g); plt.show()

def chirp_demo():
    """
    Generating and plotting a chirp signal
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import chirp
    
    fs = 500 # sampling frequency in Hz
    t =np.arange(start = 0, stop = 1,step = 1/fs) #total time base from 0 to 1 second
    g = chirp(t, f0=1, t1=0.5, f1=20, phi=0, method='linear')
    plt.plot(t,g); plt.show()

def interpret_fft_demo():
    """
    Demonstrate how to interpret FFT results
    """
    from scipy.fftpack import fft
    import numpy as np
    import matplotlib.pyplot as plt
    np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})
    
    fc=10 # frequency of the carrier
    fs=32*fc # sampling frequency with oversampling factor=32
    t=np.arange(start = 0,stop = 2,step = 1/fs) # 2 seconds duration
    x=np.cos(2*np.pi*fc*t) # time domain signal (real number)
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    ax1.plot(t,x) #plot the signal
    ax1.set_title('$x[n]= cos(2 \pi 10 t)$')
    ax1.set_xlabel('$t=nT_s$')
    ax1.set_ylabel('$x[n]$')
    
    N=256 # FFT size
    X = fft(x,N) # N-point complex DFT, output contains DC at index 0
    # Nyquist frequency at N/2 th index positive frequencies from
    # index 2 to N/2-1 and negative frequencies from index N/2 to N-1
    print(X[0]) #output approximately equal to zero
    print(abs(X[7:10])) #display samples 7 to 9
    
    # calculate frequency bins with FFT
    df=fs/N # frequency resolution
    sampleIndex = np.arange(start = 0,stop = N) # raw index for FFT plot
    f=sampleIndex*df # x-axis index converted to frequencies
    
    ax2.stem(sampleIndex,abs(X),use_line_collection=True) # sample values on x-axis
    ax2.set_title('X[k]');ax2.set_xlabel('k');ax2.set_ylabel('|X(k)|');
    ax3.stem(f,abs(X),use_line_collection=True); # x-axis represent frequencies
    ax3.set_title('X[f]');ax3.set_xlabel('frequencies (f)');ax3.set_ylabel('|X(f)|');
    fig.show()
    
    nyquistIndex=N//2 #// is for integer division
    print(X[nyquistIndex-2:nyquistIndex+3, None]) #None argument to print array as column
    
    from scipy.fftpack import fftshift
    #re-order the index for emulating fftshift
    sampleIndex = np.arange(start = -N//2, stop = N//2) # // for integer division
    X1 = X[sampleIndex] #order frequencies without using fftShift
    X2 = fftshift(X) # order frequencies by using fftshift
    df=fs/N # frequency resolution
    f=sampleIndex*df # x-axis index converted to frequencies
    
    #plot ordered spectrum using the two methods
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)#subplots creation
    ax1.stem(sampleIndex,abs(X1), use_line_collection=True)# result without fftshift
    ax1.stem(sampleIndex,abs(X2),'r', use_line_collection=True) # result with fftshift
    ax1.set_xlabel('k');ax1.set_ylabel('|X(k)|')
    
    ax2.stem(f,abs(X1), use_line_collection=True)
    ax2.stem(f,abs(X2),'r' , use_line_collection=True)
    ax2.set_xlabel('frequencies (f)'),ax2.set_ylabel('|X(f)|')
    fig.show()

def magnitude_phase_info_from_fft():
    """
    Demonstrate how to extract magnitude and phase using FFT/IFFT
    """
    from scipy.fftpack import fft, ifft, fftshift, ifftshift
    import numpy as np
    import matplotlib.pyplot as plt
    
    A = 0.5 # amplitude of the cosine wave
    fc=10 # frequency of the cosine wave in Hz
    phase=30 # desired phase shift of the cosine in degrees
    fs=32*fc # sampling frequency with oversampling factor 32
    t=np.arange(start = 0,stop = 2,step = 1/fs) # 2 seconds duration
    
    phi = phase*np.pi/180; # convert phase shift in degrees in radians
    x=A*np.cos(2*np.pi*fc*t+phi) # time domain signal with phase shift
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)
    ax1.plot(t,x) # plot time domain representation
    ax1.set_title(r'$x(t) = 0.5 cos (2 \pi 10 t + \pi/6)$')
    ax1.set_xlabel('time (t seconds)');ax1.set_ylabel('x(t)')
    
    N=256 # FFT size
    X = 1/N*fftshift(fft(x,N)) # N-point complex DFT
    
    df=fs/N # frequency resolution
    sampleIndex = np.arange(start = -N//2,stop = N//2) # // for integer division
    f=sampleIndex*df # x-axis index converted to ordered frequencies
    ax2.stem(f,abs(X), use_line_collection=True) # magnitudes vs frequencies
    ax2.set_xlim(-30, 30)
    ax2.set_title('Amplitude spectrum')
    ax2.set_xlabel('f (Hz)');ax2.set_ylabel(r'$ \left| X(k) \right|$')
    
    phase=np.arctan2(np.imag(X),np.real(X))*180/np.pi # phase information
    ax3.plot(f,phase) # phase vs frequencies
    ax3.set_title('Phase spectrum')
    ax3.set_ylabel(r"$\angle$ X[k]");ax3.set_xlabel('f(Hz)')
    
    X2=X #store the FFT results in another array
    # detect noise (very small numbers (eps)) and ignore them
    threshold = max(abs(X))/10000; # tolerance threshold
    X2[abs(X)<threshold]=0 # maskout values below the threshold
    phase=np.arctan2(np.imag(X2),np.real(X2))*180/np.pi # phase information
    ax4.stem(f,phase, use_line_collection=True) # phase vs frequencies
    ax4.set_xlim(-30, 30)
    ax4.set_title('Phase spectrum')
    ax4.set_ylabel(r"$\angle$ X[k]");ax4.set_xlabel('f(Hz)')
    fig.show()
    
    x_recon = N*ifft(ifftshift(X),N) # reconstructed signal
    t = np.arange(start = 0,stop = len(x_recon))/fs # recompute time index
    fig2, ax5 = plt.subplots()
    ax5.plot(t,np.real(x_recon)) # reconstructed signal
    ax5.set_title('reconstructed signal')
    ax1.set_xlabel('time (t seconds)');ax1.set_ylabel('x(t)')
    fig2.show()

def sine_wave_psd_demo():
    """
    Computation of power of a signal - simulation and verification
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    A=1 #Amplitude of sine wave
    fc=100 #Frequency of sine wave
    fs=3000 # Sampling frequency - oversampled by the rate of 30
    nCyl=3 # Number of cycles of the sinewave
    t=np.arange(start = 0,stop = nCyl/fc,step = 1/fs) #Time base
    x=-A*np.sin(2*np.pi*fc*t) # Sinusoidal function
    
    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols = 2)
    ax1.plot(t,x)
    ax1.set_title('Sinusoid of frequency $f_c=100 Hz$')
    ax1.set_xlabel('Time(s)');ax1.set_ylabel('Amplitude')
    fig.show()
    
    from numpy.linalg import norm
    L=len(x)
    P=(norm(x)**2)/L; #norm from numpy linear algo package
    print('Power of the Signal from Time domain {:0.4f}'.format(P))
    
    from scipy.fftpack import fft,fftshift
    NFFT=L
    X=fftshift(fft(x,NFFT))
    Px=X*np.conj(X)/(L**2) #Power of each freq components
    fVals=fs*np.arange(start = -NFFT/2,stop = NFFT/2)/NFFT
    ax2.stem(fVals,Px,'r')
    ax2.set_title('Power Spectral Density')
    ax2.set_xlabel('Frequency (Hz)');ax2.set_ylabel('Power')

def compare_convolutions():
    """
    Comparing different methods for computing convolution
    """
    import numpy as np
    from scipy.fftpack import fft,ifft
    from essentials import my_convolve #import our function from essentials.py
    
    x = np.random.normal(size = 7) + 1j*np.random.normal(size = 7) #normal random complex vectors
    h = np.random.normal(size = 3) + 1j*np.random.normal(size = 3) #normal random complex vectors
    L = len(x) + len(h) - 1 #length of convolution output
       
    y1=my_convolve(h,x) #Convolution Using Toeplitz matrix
    y2=ifft(fft(x,L)*(fft(h,L))).T #Convolution using FFT
    y3=np.convolve(h,x) #Numpy's standard function
    print(f' y1 : {y1} \n y2 : {y2} \n y3 : {y3} \n')

def analytic_signal_demo():
    """
    Investigate components of an analytic signal
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from essentials import analytic_signal #import our function from essentials.py
    
    t=np.arange(start=0,stop=0.5,step=0.001); # time base
    x = np.sin(2*np.pi*10*t) # real-valued f = 10 Hz
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(t,x) # plot the original signal
    ax1.set_title('x[n] - real-valued signal')
    ax1.set_xlabel('n');
    ax1.set_ylabel('x[n]')
    
    z = analytic_signal(x) # construct analytic signal
    
    ax2.plot(t, np.real(z), 'k',label='Real(z[n])')
    ax2.plot(t, np.imag(z), 'r',label='Imag(z[n])')
    ax2.set_title('Components of Analytic signal');
    ax2.set_xlabel('n');
    ax2.set_ylabel(r'$z_r[n]$ and $z_i[n]$')
    ax2.legend();fig.show()

def extract_envelope_phase():
    """
    Demonstrate extraction of instantaneous amplitude and phase from 
    the analytic signal constructed from a real-valued modulated signal
    """
    import numpy as np
    from scipy.signal import chirp
    import matplotlib.pyplot as plt
    from essentials import analytic_signal
    
    fs = 600 # sampling frequency in Hz
    t = np.arange(start=0,stop=1, step=1/fs) # time base
    a_t = 1.0 + 0.7 * np.sin(2.0*np.pi*3.0*t) # information signal
    c_t = chirp(t, f0=20, t1=t[-1], f1=80, phi=0, method='linear')
    x = a_t * c_t # modulated signal
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(x) # plot the modulated signal
    z = analytic_signal(x) # form the analytical signal
    inst_amplitude = abs(z) # envelope extraction
    inst_phase = np.unwrap(np.angle(z)) # inst phase
    inst_freq = np.diff(inst_phase)/(2*np.pi)*fs # inst frequency
    
    # Regenerate the carrier from the instantaneous phase
    extracted_carrier = np.cos(inst_phase)
    ax1.plot(inst_amplitude,'r') # overlay the extracted envelope
    ax1.set_title('Modulated signal and extracted envelope')
    ax1.set_xlabel('n');ax1.set_ylabel(r'x(t) and $|z(t)|$')
    ax2.plot(extracted_carrier)
    ax2.set_title('Extracted carrier or TFS')
    ax2.set_xlabel('n');ax2.set_ylabel(r'$cos[\omega(t)]$')
    fig.show()

def hilbert_phase_demod():
    """
    Demonstrate simple Phase Demodulation using Hilbert transform
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import hilbert
    
    fc = 210 # carrier frequency
    fm = 10 # frequency of modulating signal
    alpha = 1 # amplitude of modulating signal
    theta = np.pi/4 # phase offset of modulating signal
    beta = np.pi/5 # constant carrier phase offset
    # Set True if receiver knows carrier frequency & phase offset 
    receiverKnowsCarrier= False
        
    fs = 8*fc # sampling frequency
    duration = 0.5 # duration of the signal
    t = np.arange(start = 0, stop = duration, step = 1/fs) # time base
    
    #Phase Modulation
    m_t = alpha*np.sin(2*np.pi*fm*t + theta) # modulating signal
    x = np.cos(2*np.pi*fc*t + beta + m_t ) # modulated signal
    
    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(t,m_t) # plot modulating signal
    ax1.set_title('Modulating signal')
    ax1.set_xlabel('t');ax1.set_ylabel('m(t)')
    
    ax2.plot(t,x) # plot modulated signal
    ax2.set_title('Modulated signal')
    ax2.set_xlabel('t');ax2.set_ylabel('x(t)');fig1.show()
    
    #Add AWGN noise to the transmitted signal
    mu = 0; sigma = 0.1  # noise mean and sigma
    n = mu + sigma*np.random.normal(len(t)) # awgn noise
    r = x + n # noisy received signal
    
    # Demodulation of the noisy Phase Modulated signal
    z= hilbert(r) # form the analytical signal from the received vector
    inst_phase = np.unwrap(np.angle(z)) # instaneous phase
    
    if receiverKnowsCarrier: # If receiver knows the carrier freq/phase perfectly
        offsetTerm = 2*np.pi*fc*t+beta
    else: # else, estimate the subtraction term
        p = np.polyfit(x =t, y =inst_phase, deg =1)#linear fit instantaneous phase
        #re-evaluate the offset term using the fitted values
        estimated = np.polyval(p,t)
        offsetTerm = estimated
    
    demodulated = inst_phase - offsetTerm
    
    fig2, ax3 = plt.subplots()
    ax3.plot(t,demodulated) # demodulated signal
    ax3.set_title('Demodulated signal')
    ax3.set_xlabel('n')
    ax3.set_ylabel(r'$\hat{m(t)}$');fig2.show()