import datetime
import pickle
import sys
import imports



begin_time = datetime.datetime.now()


## MIMO 
from training_routine_SNRsweep import *
logging.info("One simulation with SNR sweep and 1 targ, QAM enc, 2 UEs")
M=torch.tensor([16], dtype=int).to(device)
wr = 0.2
beta = float(sys.argv[1])
num_ue = int(sys.argv[2])
SNR_s = torch.pow(10.0,torch.tensor([-10,0],device=device)/10)
SNR_c = torch.pow(10.0,torch.tensor([15,20],device=device)/10)

#training of exact beamform
logging.info("Modulation Symbols: "+str(M))
logging.info("SNR sensing = "+str(SNR_s))
logging.info("SNR Communication = "+str(SNR_c))
logging.info("beta = "+str(beta))

enc_best,dec_best, beam_best, rad_rec_best = pickle.load( open( 'protosystem2.pkl', "rb" ) )

#enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,30,0.001,1,5]),weight_sens=wr,max_target=1,stage=1, plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=num_ue, beta_corr=beta)
enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,20,0.005,1,1]),weight_sens=wr,max_target=1,stage=1,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=num_ue, beta_corr=beta)
enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,30,0.001,1,5]),weight_sens=wr,max_target=1,stage=3,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=num_ue, beta_corr=beta)
enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,50,0.001,1,5]),weight_sens=wr,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=num_ue, beta_corr=beta)
enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([50,50,0.001,1,15]),weight_sens=wr,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=num_ue, beta_corr=beta)

#enc_best,dec_best, beam_best, rad_rec_best, validation_SERs1,gmi_exact1, P_r1, const = train_network(M,SNR_s,SNR_c,train_params=cp.array([80,50,0.0001,1,15]),weight_sens=0.7,max_target=1,stage=None,NNs=[enc_best,dec_best, beam_best, rad_rec_best], plotting=True,setbehaviour="none", namespace=namespace, enctype="QAM",num_ue=2)
beta = str(round(beta, 3)).translate(None, '.,')
with open('/'+ beta +'.pkl', 'wb') as fh:
    pickle.dump([enc_best,dec_best, beam_best, rad_rec_best], fh)

logging.info("Training duration is" + str(datetime.datetime.now()-begin_time))