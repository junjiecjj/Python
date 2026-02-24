#from aifc import Error
from imports import *
from functions import *
from NN_classes import *


def train_network(M=4,SNR_s = [1,1], SNR_c = [100,200],train_params=[50,100,0.01,1,1],weight_sens = 0.1,max_target=1,stage=None,NNs=None, plotting=True, setbehaviour="none", namespace="",loss_beam=0,enctype="QAM", num_ue=2, beta_corr=0):
    """ 
    Training process of NN autoencoder

    M : number of constellation symbols
    sigma_n : noise standard deviation 
    train_params=[num_epochs,batches_per_epoch, learn_rate]
    weight_sens : Impact of the radar receiver; impact of communicationn is (1-weight_sens)
    stage : NNs should be trained serially; therefore there are 3 training stages:
        stage = 1: training of encoder, decoder, beamformer and angle estimation
        stage = 2: training of encoder, decoder, beamformer and estimation of angle uncertainty
        stage = 3: training of encoder, decoder, beamformer and target detection 
    
    M, sigma_n, modradius are lists of the same size 
    setbehaviour (str) : encompasses methods to take care of permutations;
        "setloss" -> use hausdorff distance for loss; permute in validation set
        "permute"
        "sortall"
        "sortphi"
        "none" : do nothing
        "ESPRIT" -> use esprit algorithm instead of NN 
    plotting (bool): toggles Plotting of PDFs into folder /figures 
    """
    encoding = ['sum of onehot', 'onehot'][0] #onehot only for a single target implemented for now
    #enctype = "QAM" or "NN"
    benchmark = 1
    canc = 0
    k_a = [16,1]
    sens_beam = 1
    sens_input = ["ryx","ryy"][1]

    torch.set_default_device(device)
    #torch.autograd.set_detect_anomaly(True)
    #if M.size()!=1:
    #    raise error("M, sigma_n, need to be of same size (float)!")
    
    num_epochs=int(train_params[0])
    batches_per_epoch=int(train_params[1]) #np.ceil(2000/num_epochs)on Validation Dataset
    cpr_min = int(train_params[3]) # communication per radar: integer of how many communication symbols are transmitted for the same radar estimation
    cpr_max = int(train_params[4])
    learn_rate =train_params[2]
    N_valid = 100000

    logging.info("Running training in training_routine_SNRweep.py")
    logging.info("Maximum target number is %i" % max_target)
    logging.info("Set behaviour is %s" % setbehaviour )
    logging.info("loss_beam is %s" % str(loss_beam))
    logging.info("ws is %s" % str(weight_sens))
    logging.info("input sensing %s" % str(sens_input))
    logging.info("sensing beam %s" % str(sens_beam))
    lambda_txr = 0.1

    sigma_nc_all=1/torch.sqrt(SNR_c).to(device)
    sigma_ns_all=1/torch.sqrt(SNR_s).to(device)
    sigma_c=1
    sigma_s=1

    sigmoid = torch.nn.Sigmoid()
    printing=False #suppresses all printed output but GMI
    
    # Generate Validation Data
    #y_valid = torch.zeros(N_valid,dtype=int, device=device).to(device)
    y_valid = torch.randint(0,int(M),(N_valid*cpr_max,num_ue)).to(device)

    if plotting==True:
        # meshgrid for plotting
        ext_max = 2  # assume we normalize the constellation to unit energy than 1.5 should be sufficient in most cases (hopefully)
        mgx,mgy = cp.meshgrid(cp.linspace(-ext_max,ext_max,200), cp.linspace(-ext_max,ext_max,200))
        meshgrid = cp.column_stack((cp.reshape(mgx,(-1,1)),cp.reshape(mgy,(-1,1))))
    
    if NNs==None:
        #enc = QAM_encoder(M)
        if enctype=="NN":
            enc=[]
            for i in range(num_ue):
                enc.append(Encoder(M).to(device))
        else:
            enc=[]
            for i in range(num_ue):
                enc.append(QAM_encoder(M).to(device))
        dec=[]
        for i in range(num_ue):                   
            dec.append(Decoder(M).to(device))
        beam = Beamformer(kx=k_a[0],ky=k_a[1],n_ue=num_ue+sens_beam).to(device)
        rad_rec = Radar_receiver(kx=k_a[0],ky=k_a[1],max_target=max_target,cpr_max=15, encoding=encoding, num_ue=num_ue+sens_beam).to(device)
        #rad_rec = Joint_radar_receiver(kx=k_a[0],ky=k_a[1],max_target=max_target, encoding=encoding).to(device)
    else:
        if enctype=="NN":
            enc = NNs[0]
        else:
            enc=[]
            for i in range(num_ue):
                enc.append(QAM_encoder(M).to(device)) 
        dec = NNs[1]
        beam = NNs[2]
        rad_rec = NNs[3]
        encoding = rad_rec.encoding
        
    
    # Adam Optimizer
    # List of optimizers in case we want different learn-rates
    optimizer=[]
    
    for i in range(num_ue):
        if enctype=="NN":
            optimizer.append(optim.Adam(enc[i].parameters(), lr=float(learn_rate)))
        optimizer.append(optim.Adam(dec[i].parameters(), lr=float(learn_rate)))
    optimizer.append(optim.Adam(rad_rec.parameters(), lr=float(learn_rate)))
    optimizer.append(optim.Adam(beam.parameters(), lr=float(learn_rate)))

    softmax = nn.Softmax(dim=1).to(device)

    # Cross Entropy loss
    loss_fn = nn.CrossEntropyLoss()
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss(reduction="none")
    bce_loss_target = nn.BCEWithLogitsLoss(reduction="none")
    mse_loss =nn.MSELoss(reduction="none")
    gauss_nll_loss = nn.GaussianNLLLoss()

    # fixed batch size of 10000
    batch_size_per_epoch = np.zeros(num_epochs, dtype=int)+10000*2

    validation_BER = torch.zeros((int(num_epochs),int(torch.log2(M)),num_ue))
    validation_SERs = torch.zeros((int(num_epochs),num_ue))
    validation_received = []
    det_error=[[],[]]
    sent_all=[]
    rmse_angle=[]
    rmse_benchmark=[]
    rmse_benchmark_2=[]
    d_angle_uncertain=[]
    CRB_azimuth = 100 # init high value for CRBs
    CRB_elevation = 100
    print('Start Training stage '+str(stage))
    logging.info("Start Training stage %s" % stage)

    bitnumber = int(torch.sum(torch.log2(M)))
    gmi = torch.zeros(int(num_epochs),device=device)
    gmi_exact = torch.zeros((int(num_epochs), bitnumber,num_ue), device=device)
    #m_enc = torch.zeros((batch_size_per_epoch[epoch]),device=device)
    SNR = np.zeros(int(num_epochs))
    #mradius =[]
    k = torch.arange(beam.kx)+1
    # if stage==1 or stage==3:
    #     if sens_beam==1:
    #         P = torch.zeros(sens_beam+num_ue)
    #         P[:num_ue] = np.sqrt((1-weight_sens)/num_ue)
    #         P[num_ue:] = np.sqrt((weight_sens))
    #     else:
    #         P = torch.zeros(sens_beam+num_ue) + np.sqrt(1/num_ue)
    #     beam.norm(P)
    # #elif stage==3:
    # #    beam.norm(P="randomize")
    # else:
    beam.norm(P=None)
        
    system = system_runner(M,SNR_s , SNR_c ,train_params,weight_sens,max_target,[enc,dec,beam,rad_rec], setbehaviour, namespace,enctype, num_ue, sens_beam, rc= sens_input, beta_c = beta_corr)
    alph=1

    for epoch in range(int(num_epochs)):
        cpr= torch.randint(cpr_min,cpr_max+1,(batch_size_per_epoch[epoch],1),device=device)
        
        
        help1=0
        for step in range(int(batches_per_epoch)):
            decoded, batch_cw, t_NN, target, angle_shrunk, permuted_angle_shrunk, angle_corr, _, modulated, _ = system.run_system(train=True, help1=help1)
            if stage==1: # max kappa while comm
                closs = 1/num_ue*comm_loss_ra(decoded,batch_cw,alpha=alph)#*cpr_max#*torch.unsqueeze((cpr_ex),1))
                #angle_loss = torch.mean(mse_loss(torch.squeeze(angle_shrunk), torch.squeeze(permuted_angle_shrunk))*(angle_corr).repeat(2,1).T.reshape(-1,2))
                kappa = torch.min(torch.mean(torch.abs(radiate(beam.kx,torch.linspace(-20,20,40, device=device)*np.pi/180,beam.ky) @ modulated.mT)**4,axis=1))
                loss = (1-weight_sens)*closs-kappa #+ weight_sens*angle_loss #+ beamloss
            elif stage==2: # Training of communication
                loss = torch.mean(torch.sum(bce_loss(decoded,batch_cw),1))
                loss = (1-weight_sens)*loss.clone() 
            elif stage==3: # max kappa while comm + target detection
                closs = 1/num_ue*comm_loss_ra(decoded,batch_cw,alpha=alph)
                kappa = torch.min(torch.mean(torch.abs(radiate(beam.kx,torch.linspace(-20,20,40, device=device)*np.pi/180,beam.ky) @ modulated.mT)**4,axis=1))
                loss =  (1-weight_sens)*closs + (weight_sens)*(torch.mean(bce_loss_target(torch.squeeze(t_NN),torch.squeeze(target))))-kappa*0.1#+ beamloss # combine with radar detect loss

            else: # joint comm + detect loss
                closs = 1/num_ue*comm_loss_ra(decoded,batch_cw,alpha=alph)
                loss =  (1-weight_sens)*closs + (weight_sens)*(torch.mean(bce_loss_target(torch.squeeze(t_NN),torch.squeeze(target))))#+angle_loss)# combine with radar detect loss
                
            
            # compute gradients
            loss.backward() 

            # run optimizer
            for elem in optimizer:
                elem.step()
            

            # reset gradients
            for elem in optimizer:
                elem.zero_grad()

        
        with torch.no_grad(): #no gradient required on validation data
            beam.norm(P=None)
            decoded, batch_cw, t_NN, target, angle_shrunk, permuted_angle_shrunk, _, benchmark_angle_nn, sent_all,t_bm = system.run_system(train=False,cpr=[cpr_max,cpr_max],SNR_c=1/sigma_nc_all[1].repeat(2),SNR_s=1/sigma_ns_all[0].repeat(2))
            y_valid = system.valid_labels

            if plotting==True:
                cvalid = torch.zeros(N_valid*cpr_max)
            #decoded_valid=torch.zeros((N_valid*cpr_max,int(torch.max(M)),num_ue), dtype=torch.float32, device=device)
            
            decoded_hard=[]
            decoded_symbol=[]
            #bits=[]
            s=[]
            for i in range(num_ue):
                gmi_exact[epoch,:,i]=GMI(M,decoded[i], batch_cw[i])
                decoded_hard.append(torch.round(torch.sigmoid(decoded[i])).type(torch.int16).to(device))
                decoded_symbol_h= torch.zeros((N_valid*cpr_max),dtype=torch.long).to(device)
                code_ints = torch.zeros(M).to(device)
                for b in range(int(torch.log2(M))):
                    decoded_symbol_h += decoded_hard[i][:,b]*2**b
                    code_ints += gray_code(M).to(device)[:,b]*2**b
                decoded_symbol.append(code_ints[decoded_symbol_h.detach().cpu()])
                #bits.append(BER(decoded_hard[i], batch_cw[i],M))
                validation_SERs[epoch,i] = SER(decoded_symbol[i], y_valid[:,i])
                validation_BER[epoch,:,i]= BER(decoded_hard[i], batch_cw[i],M)

            #t_hard = 0.5+0.5*torch.sign(t_NN)
            t_hard = torch.round(sigmoid(t_NN).type(torch.float))
            # detection probability
            prob_e_d = torch.sum(torch.squeeze(t_hard)*torch.squeeze(target))/torch.sum(torch.squeeze(target)).to(device)
            # false alarm rate
            prob_f_d = torch.sum(torch.squeeze(t_hard)*(1-torch.squeeze(target)))/torch.sum(1-torch.squeeze(target)).to(device)
         
                
            if printing==True:
                print('Detect Probability of radar detection after epoch %d: %f' % (epoch, prob_e_d))            
                print('False Alarm rate of radar detection after epoch %d: %f' % (epoch, prob_f_d))
            logging.info('Detect Probability of radar detection after epoch %d: %f' % (epoch, prob_e_d))
            logging.info('False Alarm rate of radar detection after epoch %d: %f' % (epoch, prob_f_d))
            logging.info("BEnchmark detection rate and detection probability are: "+ str(t_bm))
            #if l_target==max_target-1:
            det_error[0].append(prob_e_d.detach().cpu().numpy())
            det_error[1].append(prob_f_d.detach().cpu().numpy())

            ## Angle estimation
            x_detect = torch.nonzero(t_hard*target > 0.5).to(device)[:,0] # targets that were present and were detected
            rmse_benchmark.append(torch.sqrt(torch.mean(torch.abs((benchmark_angle_nn[x_detect,0,:] - permuted_angle_shrunk[x_detect,0,:]))**2)).detach().cpu().numpy())
            rmse_angle.append(torch.sqrt(torch.mean(torch.abs(torch.squeeze(angle_shrunk[x_detect,0,:])-torch.squeeze(permuted_angle_shrunk[x_detect,0,:]))**2)).detach().cpu().numpy())
  
            if printing==True:
                print('Angle estimation error after epoch %d: %f (rad)' % (epoch, rmse_angle[epoch]))
            logging.info('Angle estimation error after epoch %d: %f (deg) | %f (rad)' % (epoch, 180/np.pi*rmse_angle[epoch],rmse_angle[epoch]))

            # color map for plot
            if plotting==True:
                cvalid=y_valid
            

            if printing==True:
                print('Validation BER after epoch %d: ' % (epoch) + str(validation_BER[epoch]) +' (loss %1.8f)' % (loss.detach().cpu().numpy()))  
                print('Validation SER after epoch %d: %f (loss %1.8f)' % (epoch, validation_SERs[epoch], loss.detach().cpu().numpy()))              
            
            logging.debug('Validation BER after epoch %d: ' % (epoch) + str(validation_BER[epoch]) +' (loss %1.8f)' % (loss.detach().cpu().numpy()))
            logging.debug('Validation SER after epoch %d: ' % (epoch) + str(validation_SERs[epoch]) +' (loss %1.8f)' % (loss.detach().cpu().numpy()))
        
            if printing==True:
                print("GMI is: "+ str(torch.sum(gmi_exact[epoch]).item()) + " bit after epoch %d (loss: %1.8f)" %(epoch,loss.detach().cpu().numpy()))
            logging.info("GMI is: "+ str(torch.sum(gmi_exact[epoch]).item()) + " bit after epoch %d (loss: %1.8f)" %(epoch,loss.detach().cpu().numpy()))

            # Choose best training epoch to save NNs or keep training further
            loss_ev = (-torch.sum(gmi_exact[epoch])-prob_e_d+prob_f_d).detach().cpu().numpy() + rmse_angle[epoch] #+ torch.mean((torch.log10(uncertainty+1e-9)+(torch.abs(angle*np.pi - phi_valid)**2)/(2*uncertainty**2+1e-9))*t)


            if epoch==0:
                enc_best=system.enc
                dec_best=system.dec
                best_epoch=0
                beam_best=system.beam
                rad_rec_best = system.rad_rec
                loss_b = 100
            elif epoch==num_epochs-1:
                enc_best=system.enc
                dec_best=system.dec
                best_epoch=epoch
                beam_best=system.beam
                rad_rec_best = system.rad_rec
                loss_b = loss_ev
            elif loss_ev<loss_b:
                enc_best=system.enc
                dec_best=system.dec
                best_epoch=epoch
                beam_best=system.beam
                rad_rec_best = system.rad_rec
                loss_b = loss_ev

    if enctype=="NN":
        constellations = cp.zeros((M,num_ue),dtype=np.complex128)
        for i in range(num_ue):
            constellations[:,i] = cp.asarray(enc_best[i](torch.eye(int(M), device=device), noise=torch.unsqueeze(sigma_nc_all[1],0).repeat(M,1)).cpu().detach().numpy())
    else:
        constellations = cp.zeros((M,num_ue),dtype=np.complex128)
        for i in range(num_ue):
            constellations[:,i] = enc[i](torch.arange(int(M))).cpu().detach().numpy()
    logging.info("Constellation is: %s" % (str(constellations)))

    ### Plot & collect the results in log:
    print('Training finished')
    logging.info('Training finished')

    logging.info("SER obtained: %s" % (str(validation_SERs)))
    logging.info("GMI obtained: %s" % str(np.sum(gmi_exact.detach().cpu().numpy(),axis=1)))
      

    logging.info("CRB in azimuth is: "+str(CRB_azimuth))
    logging.info("CRB in elevation is: "+str(CRB_elevation))

    beam_tensor = np.pi/180*torch.tensor([[-20.0,20.0,50,70]])
    direction = system.beam(beam_tensor)
    
    if plotting==True:
        if device=='cpu':
            plot_training(validation_SERs.cpu().detach().numpy(),validation_BER,M, constellations, gmi_exact.detach().cpu().numpy(), direction, det_error, np.asarray(rmse_angle),benchmark = rmse_benchmark, stage=stage,namespace=namespace, CRB=CRB_azimuth, antennas=rad_rec.k,enctype=enctype) 
        else:
            try:
                plot_training(validation_SERs.cpu().detach().numpy(), validation_BER,M, cp.asnumpy(constellations),  gmi_exact.detach().cpu().numpy(), direction, det_error, np.asarray(rmse_angle),benchmark = rmse_benchmark, stage=stage, namespace=namespace, CRB=CRB_azimuth,antennas=rad_rec.k,enctype=enctype) 
            except:
                pass
                

    if device=='cpu':
        return(enc_best,dec_best, beam_best, rad_rec_best, validation_SERs,gmi_exact, det_error, cp.array(constellations))
    else:
        try:
            return(enc_best,dec_best, beam_best, rad_rec_best, validation_SERs,gmi_exact, det_error, cp.asnumpy(constellations))
        except:
            return(enc_best,dec_best, beam_best, rad_rec_best, validation_SERs,gmi_exact, det_error, (constellations))



class system_runner():
    def __init__(self,M=4,SNR_s = [1,1], SNR_c = [100,200],train_params=[50,100,0.01,1,1],weight_sens = 0.1,max_target=1,NNs=None, setbehaviour="none", namespace="",enctype="QAM", num_ue=2, sens_beam=0, rc="ryy",N_valid=10000, beta_c=0):
        self.enctype = enctype
        self.M = M
        self.SNR_s = SNR_s
        self.SNR_c = SNR_c
        self.train_params=train_params
        self.weight_sens = weight_sens
        self.max_target = max_target
        self.setbehaviour=setbehaviour
        self.namespace= namespace
        self.num_ue = num_ue
        self.sens_beam = sens_beam
        self.sens_input = rc
        self.beta_c = beta_c

        k_a = [16,1] # Antenna Array 

        if NNs==None:
            #enc = QAM_encoder(M)
            if enctype=="NN":
                self.enc=[]
                for i in range(num_ue):
                    self.enc.append(Encoder(M).to(device))
            else:
                self.enc=[]
                for i in range(num_ue):
                    self.enc.append(QAM_encoder(M).to(device))
            self.dec=[]
            for i in range(num_ue):                   
                self.dec.append(Decoder(M).to(device))
            self.beam = Beamformer(kx=k_a[0],ky=k_a[1],n_ue=num_ue+self.sens_beam).to(device)
            self.rad_rec = Radar_receiver(kx=k_a[0],ky=k_a[1],max_target=max_target,cpr_max=15, num_ue=num_ue+sens_beam).to(device)
    
        else:
            if enctype=="NN":
                self.enc = NNs[0]
            else:
                self.enc=[]
                for i in range(num_ue):
                    self.enc.append(QAM_encoder(M).to(device)) 
            self.dec = NNs[1]
            self.beam = NNs[2]
            self.rad_rec = NNs[3]
        
        self.N_valid = N_valid
        batch_labels = torch.zeros((self.N_valid*int(self.train_params[4])*self.num_ue),dtype=torch.long, device=device)
        batch_labels.random_(int(self.M)).to(device)
        
        self.labels_onehot = torch.zeros(int(self.N_valid*int(self.train_params[4])*self.num_ue), int(self.M), device=device)
        self.labels_onehot[range(self.labels_onehot.shape[0]), batch_labels.long()]=1
        self.labels_onehot = self.labels_onehot.reshape(int(self.N_valid*int(self.train_params[4])), int(self.M),self.num_ue)
        batch_labels = batch_labels.reshape(-1,self.num_ue)
        self.valid_labels=batch_labels

        
        self.sensing_beam = beam_opt(torch.linspace(-np.pi,np.pi, 360),-20/180*np.pi,20/180*np.pi,k_a)
        



    def run_system(self,train = True, cpr=None, SNR_s=None, SNR_c=None, help1=0):
        if train==True:
            N_valid=100000 #np.ceil(2000/num_epochs)on Validation Dataset
            cpr_min = int(self.train_params[3]) # communication per radar: integer of how many communication symbols are transmitted for the same radar estimation
            cpr_max = int(self.train_params[4])

            if self.enctype=="NN":
                for i in range(self.num_ue):
                    self.enc[i].train()
            for i in range(self.num_ue):
                self.dec[i].train()
            self.rad_rec.train()
            self.beam.train()
            
            if SNR_c==None:
                sigma_nc_all=1/torch.sqrt(self.SNR_c).to(device)
            else:
                sigma_nc_all=1/torch.sqrt(SNR_c).to(device)
            if SNR_s==None:
                sigma_ns_all=1/torch.sqrt(self.SNR_s).to(device)
            else:
                sigma_ns_all=1/torch.sqrt(SNR_s).to(device)
            
            
        else:
            N_valid = self.N_valid
            self.rad_rec.eval()
            self.beam.eval()
            if self.enctype=="NN":
                for i in range(self.num_ue):
                    self.enc[i].eval()
            cpr_min=cpr[0]
            cpr_max=cpr[1]
            sigma_nc_all=1/torch.sqrt(SNR_c).to(device)
            sigma_ns_all=1/torch.sqrt(SNR_s).to(device)
            
        
        #### random parameters
        cpr= torch.randint(cpr_min,cpr_max+1,(N_valid,1),device=device)
        
        
        sigma_nc = torch.rand((int(N_valid*cpr_max),self.num_ue)).to(device)*(sigma_nc_all[1]-sigma_nc_all[0])+sigma_nc_all[0]
        sigma_ns = (torch.rand(int(N_valid),1).to(device)*(sigma_ns_all[1]-sigma_ns_all[0])+sigma_ns_all[0])
        sigma_ns_ex = sigma_ns.repeat(1,cpr_max).reshape(int(N_valid)*cpr_max)
        # sample new mini-batch directory on the GPU (if available)
        decoded=torch.zeros(int(N_valid*cpr_max),(torch.max(self.M)), device=device)
        
        ### Initialize random communication symbols, receiver angles, target numbers and target angles ###
        if train==True:
            batch_labels = torch.zeros((N_valid*cpr_max*self.num_ue),dtype=torch.long, device=device)
            batch_labels.random_(int(self.M)).to(device)
            batch_labels_onehot = torch.zeros(int(N_valid*cpr_max*self.num_ue), int(self.M), device=device)
            batch_labels_onehot[range(batch_labels_onehot.shape[0]), batch_labels.long()]=1
            batch_labels_onehot = batch_labels_onehot.reshape(int(N_valid*cpr_max), int(self.M),self.num_ue)
            batch_labels = batch_labels.reshape(-1,self.num_ue)
        else:
            batch_labels = self.valid_labels
            batch_labels_onehot = self.labels_onehot
        
        batch_cw = [] 
        if self.enctype!="QAM":
            for i in range(self.num_ue):
                batch_cw.append(gray_code(self.M)[batch_labels[:,i]].to(device))
        else:
            for i in range(self.num_ue):
                batch_cw.append(self.enc[i].coding()[batch_labels[:,i]])

        theta_valid = torch.zeros((int(N_valid),2,self.num_ue), device=device)
        theta_valid[:,0,:] = np.pi/180*torch.tensor([50,70]).to(device)
        theta_valid[:,1,:] = np.pi/2
        theta_valid = torch.repeat_interleave(theta_valid,cpr_max,dim=0)
        
        target_labels = torch.randint(self.max_target+1,(int(N_valid),)).to(device) # Train in each epoch first 1 target, then 2, then ...

        ##encoding: [1,1,0,...] means 2 targets are detected
        target = torch.zeros((int(N_valid),self.max_target)).to(device)
        label_tensor = torch.zeros(self.max_target+1,self.max_target).to(device)
        for x in range(self.max_target+1):
            label_tensor[x] = torch.concat((torch.ones(x), torch.zeros(self.max_target-x)))
        target += label_tensor[target_labels] 
        target_onehot = torch.zeros((int(N_valid),self.max_target+1)).to(device)
        target_onehot[np.arange(int(N_valid)),target_labels] = 1


        phi_valid = torch.zeros((int(N_valid),2,self.max_target)).to(device)
        phi_valid[:,0,:] = np.pi/180*(torch.rand((int(N_valid),self.max_target))*40-20) # paper: [-20 deg, 20 deg]
        phi_valid[:,1,:] = np.pi/2
        phi_valid[:,0,:] *= target
        phi_valid[:,1,:] *= target

        # Only applies for multiple targets
        if (self.setbehaviour=="sortall" or self.setbehaviour=="sortphi") and self.max_target>1:
            for l in range(int(N_valid)):
                idx1 = torch.nonzero(phi_valid[l,0,:]).to(device)
                idx0 = torch.nonzero(phi_valid[l,0,:] == 0).to(device)
                if idx1.numel() > 0:
                    idx = torch.argsort(phi_valid[l,0,torch.squeeze(idx1,1)], descending=True).to(device)
                    if idx0.numel() > 0:
                        idx = torch.squeeze(torch.cat((idx,torch.squeeze(idx0,1))))
                    else:
                        idx = torch.squeeze(idx)
                else:
                    idx = torch.squeeze(idx0).to(device)
                phi_valid[l,0,:] = phi_valid[l,0,idx]
                phi_valid[l,1,:] = phi_valid[l,1,idx]

        # enable oversampling
        phi_valid_ex = phi_valid.repeat(1,cpr_max,1).reshape(cpr_max*N_valid,2,self.max_target)

        input_beam = torch.zeros((int(N_valid*cpr_max),4)).to(device)
        input_beam[:,0:2] = np.pi/180*torch.tensor([[-20.0,20.0]]).repeat(int(N_valid*cpr_max),1).to(device)
        input_beam[:,2:4] = theta_valid[:,0,0:2]
        direction = self.beam(input_beam).to(device) # give the angles in which targets/receivers are to be expected
        

        # Propagate (training) data through transmitter
        if self.enctype!="QAM":
            encoded = torch.zeros((N_valid*cpr_max,self.num_ue+self.sens_beam,1)).type(torch.complex64)
            for i in range(self.num_ue):
                encoded[:,i] = torch.unsqueeze(self.enc[i](batch_labels_onehot[:,:,i],noise=sigma_nc[:,i]),1)
            if self.sens_beam==1:
                encoded[:,i+1] = 1
            del batch_labels_onehot
        else:
            encoded = torch.zeros((N_valid*cpr_max,self.num_ue+self.sens_beam,1)).type(torch.complex64)
            for i in range(self.num_ue):
                encoded[:,i] = torch.unsqueeze(self.enc[i](batch_labels[:,i]),1).to(device)
            if self.sens_beam==1:
                encoded[:,i+1] = (1*torch.exp(1j*torch.rand(batch_labels[:,i].shape[0])*2*np.pi)).reshape(-1,1)

        #modulated = torch.sum(encoded.expand(N_valid*cpr_max,num_ue,k_a[0]*k_a[1]).mT @ direction.mT,1)
        #modulated = torch.matmul(encoded, torch.unsqueeze(torch.transpose(direction,0,1),0)) # Apply Beamforming 
        modulated = torch.squeeze(direction @ encoded).to(device)

        # radar target detection
        target_ex = target.repeat(1,1,cpr_max).reshape(cpr_max*N_valid,self.max_target)

        # Propagate through channel
        received = [] #torch.zeros(N_valid*cpr_max,num_ue).type(torch.complex64)
        decoded = [] #torch.zeros(N_valid*cpr_max,int(np.log2(M)),num_ue)
        CSI=[]
        to_receiver = torch.squeeze(modulated.unsqueeze(2).transpose(1,2) @ radiate(self.beam.kx,theta_valid[:,0].reshape(-1), self.beam.ky, theta_valid[:,1].reshape(-1)).reshape(-1,self.num_ue,self.beam.kx).transpose(1,2))
        for i in range(self.num_ue):
            #to_receiver = torch.sum(modulated * radiate(self.beam.kx,theta_valid[:,0,i], self.beam.ky, theta_valid[:,1,i]), axis=1)
            if i==0:
                # always reflect in whole angle range
                tr_2 = torch.sum(modulated * radiate(self.beam.kx,phi_valid_ex[:,0,i], self.beam.ky, phi_valid_ex[:,1,i]), axis=1)*(phi_valid_ex[:,0,i]*torch.squeeze(target_ex)>10*np.pi/180)
                alph = self.beta_c
                r1, beta = two_path_channel(torch.stack((to_receiver[:,i],tr_2)).mT, torch.tensor([1,alph]), sigma_nc[:,i])
                CSI1 = beta[0] * torch.sum(direction[0,:,i] * radiate(self.beam.kx,theta_valid[:,0,i], self.beam.ky, theta_valid[:,1,i]), axis=1)
                CSI2 = beta[1] * torch.sum(direction[0,:,i] * radiate(self.beam.kx,phi_valid_ex[:,0,i], self.beam.ky, phi_valid_ex[:,1,i]), axis=1)*(phi_valid_ex[:,0,i]*torch.squeeze(target_ex)>10*np.pi/180)
                CSI.append(CSI1+CSI2)
                received.append(r1)
            else:
                r2, beta1 = rayleigh_channel(to_receiver[:,i], 1, sigma_nc[:,i], 0.1)
                CSI.append(beta1 * torch.sum(direction[0,:,i] * radiate(self.beam.kx,theta_valid[:,0,i], self.beam.ky, theta_valid[:,1,i]), axis=1))
                received.append(r2)
            # calculate specific channel state information 
            decoded.append(self.dec[i](received[i], CSI[i],noise=sigma_nc[:,i]))

        

        received_rad,_ = radar_channel_swerling1(modulated,1, sigma_ns_ex, 0.1,self.rad_rec.k, phi_valid=phi_valid_ex, target=target_ex)
        #received_rad,_ = radar_channel_swerling1(modulated,1, sigma_ns_ex, 0.1,self.rad_rec.k, phi_valid=phi_valid_ex, target=target_ex)
        
        cpr_tensor = torch.tril(torch.ones(cpr_max,cpr_max))
        x_j = torch.transpose(cpr_tensor[cpr.detach().cpu().numpy()-1],0,1).reshape((cpr_max*N_valid,1)).repeat((1,self.rad_rec.k[0])).to(device)
        x_i = torch.reshape(received_rad * x_j, (N_valid, cpr_max, self.rad_rec.k[0]*self.rad_rec.k[1])).to(device)
        x_i = torch.transpose(x_i, 2,1)
        #x_i = torch.transpose(x_i, 2,1)
        #x_i = torch.transpose(torch.reshape(x_i,(N_valid,cpr_max,self.rad_rec.k[0]*self.rad_rec.k[1])),1,2)

        if self.sens_input=="ryx":
            x_e = torch.reshape(modulated * x_j,(N_valid,cpr_max, self.rad_rec.k[0]*self.rad_rec.k[1])).to(device)
            R = torch.matmul(x_i,torch.conj(x_e))/(cpr*sigma_ns).reshape(-1,1,1)
        else:
            R = (x_i @ torch.transpose(torch.conj(x_i),1,2))/(cpr*sigma_ns**2).reshape(-1,1,1)
        
        received_radnn = (R.reshape(N_valid,-1)).to(device)

        angle= torch.zeros_like(phi_valid)

        t_NN = CFAR_detect_train(x_i,N_valid,sigma_ns,0.01,cpr, self.sensing_beam)

        bm_detect= torch.zeros(4)
        if train==False:
            Pf=0.01
            d = CFAR_detect(x_i,N_valid,sigma_ns[0],0.01,cpr_max, self.sensing_beam)
            bm_detect[0] = torch.sum(torch.squeeze(d)*torch.squeeze(target))/torch.sum(torch.squeeze(target)).to(device)
            bm_detect[1] = torch.sum(torch.squeeze(d)*torch.squeeze(1-target))/torch.sum(torch.squeeze(1-target)).to(device)
            a = radiate(self.beam.kx,torch.tensor(0, device=device),self.beam.ky)
            beta1 = (a @ direction[0])**2
            xtl = encoded #torch.mean(torch.abs(encoded)**4, axis=0)bm_detect[2], bm_detect[3]= detect_eval(0.01,cpr_max, beta1, sigma_ns[0], xtl)
            bm_detect[2], bm_detect[3] = detect_eval(0.01,cpr_max,beta1,sigma_ns[0],xtl)

            values = torch.sum(torch.sum(torch.abs(torch.reshape(x_i, (N_valid, cpr_max,-1)))**2,axis=1),axis=1)
            #scale = float(chi2.ppf(1-Pf,cpr[0]*16*2))/float(16*chi2.ppf(1-Pf,cpr[0]*2))
            #values2 = torch.sum(torch.abs(torch.reshape(self.sensing_beam @ x_i, (N_valid, cpr_max,-1)))**2,axis=1)[:,0]*scale
            
            t1 = values[torch.nonzero(target[:,0]*values)].detach().cpu().numpy()
            t0 = values[torch.nonzero((1-target[:,0])*values)].detach().cpu().numpy()

            #t1rb = values2[torch.nonzero(target[:,0]*values2)].detach().cpu().numpy()
            #t0rb = values2[torch.nonzero((1-target[:,0])*values2)].detach().cpu().numpy()

            # plt.hist(t1[:,0],bins=20,density=True)
            # plt.hist(t0[:,0],bins=20,density=True)

            # #plt.hist(t1rb[:,0],bins=20,density=True)
            # #plt.hist(t0rb[:,0],bins=20,density=True)
            # plt.savefig("detection_stat.png")

        # only relevant for multiple targets
        if (self.setbehaviour=="sortall" and self.max_target>1):
            for l in range(int(N_valid*1)):
                t1 = torch.nonzero(torch.abs(angle[l,1,:])>0.1).to(device)
                t0 = torch.nonzero(torch.abs(angle[l,1,:]) <= 0.1).to(device)
                if t1.numel() > 0:
                    idx = torch.argsort(angle[l,0,torch.squeeze(t1,1)], descending=True).to(device)
                    if t0.numel() > 0:
                        idx = torch.squeeze(torch.cat((idx,torch.squeeze(t0,1))))
                    else:
                        idx = torch.squeeze(idx)
                else:
                    idx = torch.squeeze(t0).to(device)
                angle[l,0,:] = angle[l,0,idx]
                angle[l,1,:] = angle[l,1,idx]

        if (self.setbehaviour=="permute" or self.setbehaviour=="ESPRIT" and self.max_target>1):
            permuted_angle = permute( phi_valid, angle, self.max_target,target_labels)
        else:
            permuted_angle = phi_valid
        #
        permuted_angle_shrunk = permuted_angle#torch.mean(permuted_angle.reshape(N_valid,cpr_max,2,max_target),1)
        angle_shrunk = torch.mean(angle.reshape(N_valid,1,2,self.max_target),1)
        #targ = torch.squeeze(torch.nonzero(permuted_angle_shrunk[:,0,0]))
        benchmark_angle_nn = torch.zeros(N_valid,2,self.max_target).to(device)
        #benchmark_angle_nn2 = torch.zeros(N_valid,2,self.max_target).to(device)
        if train==False:
            x_e = torch.reshape(modulated * x_j,(N_valid,cpr_max, self.rad_rec.k[0]*self.rad_rec.k[1])).to(device)
            R = torch.matmul(x_i,torch.conj(x_e))/(cpr*sigma_ns).reshape(-1,1,1)
            R2 = torch.matmul(x_i,torch.transpose(torch.conj(x_i),1,2))/(cpr*sigma_ns).reshape(-1,1,1)
            for i in range(N_valid):
                if target_labels[i]!=0:
                    #c = received_rad[i*cpr_max:i*cpr_max+cpr[i],:]
                    #time_esprit = datetime.datetime.now()
                    #benchmark_angle_nn[i,0,0:target_labels[i]] = esprit_angle_nns(c,self.rad_rec.k,target_labels[i], cpr[i],0)
                    #time_esprit = datetime.datetime.now() -time_esprit
                    #benchmark_angle_nn[i,1,0:target_labels[i]] = esprit_angle_nns(c,self.rad_rec.k[[1,0]],target_labels[i], cpr[i],0)+np.pi/2

                    benchmark_angle_nn[i,0] = ml_est_angle(Ryx=R[i])
                    #benchmark_angle_nn2[i,0] = angle_est_ML(x_i[i])
        #kappa = torch.sum(radiate(self.beam.kx,torch.linspace(-20,20,40, device=device)*np.pi/180,self.beam.ky) @ modulated)
        #del x_i
        #del x_j
        if self.sens_input=="ryx":
            return decoded, batch_cw, t_NN, target, angle_shrunk, permuted_angle_shrunk, (cpr/sigma_ns**2), benchmark_angle_nn, modulated, bm_detect
        else:
            return decoded, batch_cw, t_NN, target, angle_shrunk, permuted_angle_shrunk, (cpr/sigma_ns**2), benchmark_angle_nn, modulated, bm_detect




