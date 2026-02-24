#from Autoencoder.imports import *
from Autoencoder.NN_classes import * #Encoder,Decoder,Radar_receiver,Radar_detect,Angle_est
from Autoencoder.functions import *
from Autoencoder.training_routine_SNRsweep import system_runner
import sys
#import pickle
torch.set_default_device(device)
#enc =NN_classes.Encoder(M=4)
logging.info("detection_eval.py")
#enc, dec, beam, rad_rec = pickle.load( open( "figures/trained_NNs.pkl", "rb" ) )
softmax = nn.Softmax(dim=1)
sigmoid = nn.Sigmoid()

from mpmath import invertlaplace

#mp.dps = 15; mp.pretty = True




def CFAR_detect(sigIN,N_valid, noise, Pf,cpr):
    sig_power = np.sum(np.sum(np.abs(np.reshape(sigIN, (N_valid, cpr,-1)))**2,axis=1),axis=1)
    #threshold = ncx2.ppf(1-Pf,cpr*16,beta)
    threshold =noise/2*float(chi2.ppf(1-Pf,cpr*K*2))
    detect = (sig_power > threshold)
    return detect, sig_power

def detect_eval(Pf,cpr,beta=1,sigmans=1,xtl=1, K=16):

    threshold =sigmans**2/2*float(chi2.ppf(1-Pf,cpr*K*2))
    
    sigmas=1
    mu = (cpr*K*sigmas**2*np.mean(np.abs(xtl)**2) + sigmans**2*cpr*K)
    sigma = np.real(np.sqrt(cpr*K**2*sigmas**4*np.mean(np.abs(xtl)**4)+cpr*K*sigmans**4+(2*np.mean(np.abs(xtl)**2)*cpr*K*sigmans**2*1**2)))
    detect_lim = 1-norm.cdf(threshold,mu,sigma)

    t2 = np.linspace(-0.1*threshold,1.5*threshold,200)
    prob_plot1 = norm.pdf(t2,mu,sigma)

    mu = (cpr*K*sigmas**2*np.mean(np.abs(xtl)**2) )
    sigma = np.real(np.sqrt(cpr*K**2*sigmas**4*np.mean(np.abs(xtl)**4)))
    prob_plot_sig = norm.pdf(t2,mu,sigma)

    sigmas=0
    mu = (cpr*K*sigmas**2*np.mean(np.abs(xtl)**2) + sigmans**2*cpr*K)
    sigma = np.real(np.sqrt(cpr*K**2*sigmas**4*np.mean(np.abs(xtl)**4)+cpr*K*sigmans**4))
    pf = 1-norm.cdf(threshold,mu,sigma)

    prob_plot = norm.pdf(t2,mu,sigma)

    sigmamix = np.sqrt((2*np.mean(np.abs(xtl)**2)*cpr*K*sigmans**2*1**2))
    prob_plot_mix= norm.pdf(t2,0,sigmamix)

    mgf_mu = (K-1)*cpr*sigmans**2+cpr*(sigmans**2+np.mean(np.abs(xtl)**2)*K*sigmas**2)
    mgf_sig = (-1 + K) *cpr * (1 + (-1 + K) *cpr) *sigmans**4 - 2 *(-1 + K)* cpr**2* sigmans**2 *(-sigmans**2 - np.mean(np.abs(xtl)**2)* K *sigmas**2) + cpr *(1 + cpr)* (sigmans**4 +2*sigmans**2*np.mean(np.abs(xtl)**2)* K *sigmas**2 +np.mean(np.abs(signal)**4)* K**2 *sigmas**4)

    prob_plot_mgf= norm.pdf(t2,mgf_mu,np.sqrt(mgf_sig))
    
    return np.array((detect_lim, pf)), (t2,prob_plot1,prob_plot, prob_plot_mix, prob_plot_sig, prob_plot_mgf)


size = 30
SNR_s = 10.0**(-5/10) # dB to linear
sigma_s = np.sqrt(1)
sigma_c = 1
sigma_ns = 1/np.sqrt(SNR_s)


K=16

M=torch.tensor([16]).to(device)
qam = QAM_encoder(M)
beta= np.linspace(0,1,11,endpoint=True)
pf = 10**np.linspace(-8,0,72,endpoint=True)

try:
    plot_l = int(sys.argv[1])
except:
    plot_l = 8
print(plot_l)

#k_a=[16,1]
N_valid = 100000
cpr=[50] #np.arange(20)+1
N_angles = N_valid
syms = torch.randint(16,(cpr[0],N_valid)).reshape(N_valid,cpr[0])
kappa = np.mean(np.abs((np.outer(qam.forward(syms).detach().cpu().numpy(),np.sqrt(beta)) + np.outer(np.exp(1j*np.random.uniform(-np.pi,np.pi,(cpr[0],N_valid))),np.sqrt(1-beta)))**4), axis=0)
cl = (np.abs((np.outer(qam.forward(syms).detach().cpu().numpy(),np.sqrt(beta[plot_l])) + np.outer(np.exp(1j*np.random.uniform(-np.pi,np.pi,(cpr[0],N_valid))),np.sqrt(1-beta[plot_l])))**2)).reshape(N_valid, cpr[0])


cl1 = np.abs(qam.forward(syms).detach().cpu().numpy())**2
cl2 = (np.abs(np.exp(1j*np.random.uniform(-np.pi,np.pi,(N_valid,cpr[0])))))**2
stats= np.zeros((len(kappa), len(cpr), len(pf),2))
rate_detect = np.zeros((len(kappa), len(cpr), len(pf)))



def MGF_signal_plus_noise(s):
    return ((1/(1+sigma_ns**2*s))**((K-1)*cpr[0]))*np.mean(np.prod(1/(1+(sigma_ns**2+K*sigma_s**2*np.abs(cl[0:50]))*s), axis=1),axis=0)

def MGF_signal_plus_noise0(s):
    return ((1/(1+sigma_ns**2*s))**((K)*cpr[0]))

def detect_pdf_eval():
    t = np.linspace(1,1500,750)
    pdf_0 = [invertlaplace(MGF_signal_plus_noise0,ti, method='dehoog') for ti in t]
    pdf_1 = [invertlaplace(MGF_signal_plus_noise, ti, method='dehoog') for ti in t]
    return t, pdf_0, pdf_1


t, pdf_0, pdf_1 = detect_pdf_eval()


for i in range(len(cpr)):
    signal = ((qam.forward(torch.randint(M,(N_valid* cpr[i],))).detach().cpu().numpy()*np.sqrt(beta[plot_l]))+ np.exp(1j*np.random.uniform(-np.pi,np.pi,N_valid*cpr[i]))*np.sqrt(1-beta[plot_l])).reshape(N_valid, cpr[i])
    noise = ((np.random.randn(N_valid*cpr[i]*K)+1j*np.random.randn(N_valid*cpr[i]*K))*sigma_ns/np.sqrt(2)).reshape(N_valid,K, cpr[i])
    rcs = ((np.random.randn(N_valid*cpr[i])+1j*np.random.randn(N_valid*cpr[i]))*1/np.sqrt(2)).repeat(K).reshape(N_valid, cpr[i],K).mT
    signaln = signal.repeat(K).reshape(N_valid, cpr[i],K).mT*rcs + noise

    for k in range(len(pf)):
        detect, power_stat = CFAR_detect(signaln, N_valid, 1/SNR_s, pf[k], cpr[i])
        rate_detect[plot_l,i,k] = np.sum(detect)/N_valid

        

        stats[plot_l,i,k], det_stat = detect_eval(pf[k], cpr[i], beta[plot_l], sigma_ns, signal, K=K)
        if (k==3 and i==0):
            #ti, pdf_0, pdf_1 = detect_pdf_eval()
            
            plt.figure()
            t1, x1, _ = plt.hist(power_stat, density=True,bins=50, alpha=0.2, label="zs^2")
            t2, x2, _ = plt.hist(np.sum(np.sum(np.abs(noise)**2,1),1), density=True,bins=50, alpha=0.2, label="noise^2")
            plt.plot(t,fx[0],'o', label="Saddlepoint approx")
            plt.plot(t,fx_cm[0],'o', label="Saddlepoint approx constant modulus")
            plt.plot(t,fx_0[0],'o', label="Saddlepoint approx no target")
            
            plt.plot(t, pdf_0,'x', label="noise")
            plt.plot(t, pdf_1,'x', label="target")
            plt.legend()
            plt.show()

            l1=[]
            l1.append("p")
            l1.append("t0")
            l1.append("t1")
            #save_to_txt(np.stack((t,pdf_0,pdf_1)),"pd_mgf"+str(l),label=l1)
            save_to_txt(np.stack((t1,x1[:len(x1)-1],t2,x2[:len(x1)-1])), "hist_data"+str(beta[plot_l]),label=['bin1','v1', 'bin2', 'v2'])
            

plt.figure()
plt.plot(pf,rate_detect[0,0,:], label="Rate PSK")
plt.plot(pf,stats[0,0,:,0], label="prob Pd PSK")
plt.plot(pf,stats[0,0,:,1], label="prob Pf PSK")

plt.plot(pf,rate_detect[10,0,:], label="Rate QAM")
plt.plot(pf,stats[10,0,:,0], label="prob Pd QAM")
plt.plot(pf,stats[10,0,:,1], label="prob Pf QAM")
plt.xlabel("Pf")
plt.ylabel("Pd")
plt.legend()

#np.savetxt("rate_detect_pf", rate_detect.reshape(-1,len(pf)), comments="")
np.save("rate_detect_all",rate_detect)
np.save("stats", stats)

plt.figure()
plt.plot(beta[:],rate_detect[:,0,45], label="Rate PSK+QAM")
plt.plot(beta[:],stats[:,0,45,0], label="prob Pd PSK+QAM")
plt.plot(beta[:],stats[:,0,45,1], label="prob Pf PSK+QAM")
plt.xlabel("beta")
plt.ylabel("Pd")
plt.legend()

plt.figure()
plt.plot(cpr,rate_detect[1,:,45], label="Rate")
plt.plot(cpr,stats[1,:,45,0], label="prob Pd")
plt.plot(cpr,stats[1,:,45,1], label="prob Pf")
plt.xlabel("N")
plt.ylabel("Pd")
plt.legend()

plt.show()





    