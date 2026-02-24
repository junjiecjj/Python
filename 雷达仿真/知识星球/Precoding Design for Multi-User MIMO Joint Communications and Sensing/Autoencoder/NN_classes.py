from numpy import empty
from imports import *
from functions import cdist, gray_code

class Encoder(nn.Module):
    def __init__(self,M):
        super(Encoder, self).__init__()
        self.M = torch.as_tensor(M, device=device)
        self.K = 16
        self.enctype = "NN"
        # Define Transmitter Layer: Linear function, M icput neurons (symbols), 2 output neurons (real and imaginary part)        
        self.fcT1 = nn.Linear(self.M,8*self.M, device=device) 
        #self.fcT1s = nn.Linear(1,2*self.M, device=device) 
        self.fcT2 = nn.Linear(8*self.M, 8*self.M,device=device)
        self.fcT3 = nn.Linear(8*self.M, 8*self.M,device=device) 
        self.fcT5 = nn.Linear(8*self.M, 2,device=device)
        #if mradius==1:
        #    self.modradius = nn.Parameter(mradius.clone().detach(), requires_grad=False).cuda()
        #else:

        # Non-linearity (used in transmitter and receiver)
        self.activation_function = nn.ReLU()  # in paper: LeakyReLU for hidden layers    


    def forward(self, x, noise):
        # compute output
        #AP = torch.zeros(len(x),1)+0.1
        #x_con = torch.cat((x,AP),1)
        out = self.activation_function(self.fcT1(x))
        
        out = self.activation_function(self.fcT2(out))
        out = self.activation_function(self.fcT3(out))
        encoded = self.fcT5(out)
        # compute normalization factor and normalize channel output
        norm_factor = torch.sqrt(torch.mean(torch.abs((torch.view_as_complex(encoded)).flatten())**2)) # normalize mean squared amplitude to 1
        lim = torch.max(torch.tensor([norm_factor,1])) # Allow for energy reduction
        #norm_factor = torch.max(0.1, norm_factor)
        #norm_factor = torch.max(torch.abs(torch.view_as_complex(encoded)).flatten())
        #if norm_factor>1:        
        #norm_factor = torch.sqrt(torch.mean(torch.mul(encoded,encoded)) * 2 ) # normalize mean amplitude in real and imag to sqrt(1/2)
        modulated = torch.view_as_complex(encoded)/norm_factor
        return modulated

class Decoder(nn.Module):
    def __init__(self,M):
        super(Decoder, self).__init__()
        # Define Receiver Layer: Linear function, 2 icput neurons (real and imaginary part), M output neurons (symbols)
        self.M = torch.as_tensor(M, device=device)
        self.fcR1 = nn.Linear(5,10*self.M,device=device)
        self.dropout1 = nn.Dropout()
        #self.fcR1s = nn.Linear(1,2*self.M, device=device)  
        self.fcR2 = nn.Linear(10*self.M,10*self.M,device=device)
        self.fcR3 = nn.Linear(10*self.M,10*self.M,device=device)
        self.dropout2 = nn.Dropout()
        #self.fcR3b = nn.Linear(20*self.M,20*self.M,device=device)
        self.fcR4 = nn.Linear(10*self.M,10*self.M,device=device) 
        self.fcR5 = nn.Linear(10*self.M, int(torch.log2(self.M)),device=device) 
        #self.alpha=torch.tensor([alph,alph])
        # Non-linearity (used in transmitter and receiver)
        self.activation_function = nn.ELU()
        #self.CSI = CSI #channel state information kappa      

    def forward(self, x, CSI,noise):
        # compute output

        # MMSE equalizer
        mmse = torch.conj(CSI)/(torch.conj(CSI)*CSI+torch.squeeze(noise**2)).to(device)
        x_prep = x*mmse # MMSE equalizer approach
        x_real = torch.cat((torch.view_as_real(x_prep).float(),torch.view_as_real(CSI),(noise**2).reshape(-1,1)),1)#noise**2
        out = self.activation_function(self.fcR1(x_real))
        out = self.activation_function(self.fcR2(out))
        out = self.activation_function(self.fcR3(out))
        out = self.activation_function(self.fcR4(out)) 
        logits = self.fcR5(out)
        
        return logits

class Beamformer(nn.Module):
    """ Transforms the single transmit signal into a tensor of transmit signals for each Antenna
    beamforming is necessary if multiple receivers are present or additional Radar detection is implemented.
    Learning of an appropriate beamforming:
    
    input paramters:
        [theta_min, theta_max] : Angle Interval of Radar target
        [phi_min, phi_max] : Angle interval of receiver
        theta_last : last detected angle of target

    output parameters:
        out : phase shift for transmit antennas
     """
    def __init__(self,kx,ky=1,n_ue=1):
        super(Beamformer, self).__init__()
        self.kx =torch.as_tensor(kx) # Number of transmit antennas in x-direction;
        self.ky =torch.as_tensor(ky) # Number of transmit antennas in y-direction;
        self.n_ue = n_ue
        
        #self.d1 = nn.Dropout(p=0.2)
        self.fcB1 = nn.Linear(4,self.kx, device=device)
        self.fcB2 = nn.Linear(self.kx,self.kx, device=device)
        self.fcB3 = nn.Linear(self.kx,self.kx*2, device=device)
        self.fcB4 = nn.Linear(self.kx*2,self.kx*2*n_ue, device=device) # linear output layer

        if ky>1:
            self.fcA1 = nn.Linear(4,self.ky, device=device)
            self.fcA2 = nn.Linear(self.ky,self.ky, device=device)
            self.fcA3 = nn.Linear(self.ky,self.ky*2, device=device)
            self.fcA4 = nn.Linear(self.ky*2,self.ky*2*n_ue, device=device) # linear output layer
        # Non-linearity (used in transmitter and receiver) ?
        self.activation_function = nn.ELU() # ReLU
        self.P = None
        self.icounter=0
    
    def norm(self,P):
        self.P = P

    def forward(self, Theta):
        try:
            out = self.activation_function(self.fcB1(Theta)).to(device)
        except:
            out = self.activation_function(self.fcB1(torch.hstack((Theta,torch.zeros((len(Theta),1),device=device))))).to(device)
        #out = self.activation_function(self.fcB2(out)).to(device)
        out_2 = self.activation_function(self.fcB3(out)).to(device)
        outx = torch.view_as_complex(torch.reshape(self.activation_function(self.fcB4(out_2)),(-1,self.kx,self.n_ue,2))).to(device)
        #outx = (torch.exp(1j*out)/torch.sqrt(self.kx)).to(device) # output is transformed from angle to complex number; normalize with antenna number?
        
        outy = torch.tensor([[1+0j]]).to(device)
        if self.ky>1:
            out = self.activation_function(self.fcA1(Theta)).to(device)
            out = self.activation_function(self.fcA2(out)).to(device)
            out_2 = self.activation_function(self.fcA3(out)).to(device)
            outy = torch.view_as_complex(torch.reshape(self.activation_function(self.fcA4(out_2)),(-1,self.ky,self.n_ue,2))).to(device)
            #outy = (torch.exp(1j*out)/torch.sqrt(self.ky)).to(device)
            out_allp = outx @ outy #torch.kron(outx,outy).to(device)  # Power norm so that we can compare Antenna configurations
        else:
            out_allp = outx
        #print(torch.sum(torch.abs(out_all)**2))
        #out_allp =torch.ones((len(Theta),self.kx*self.ky)).to(device) # for QAM beam comparison
        
        if self.P=="self":
            P = torch.zeros(self.n_ue)+1/torch.sqrt(torch.tensor(self.n_ue).to(device))
            d = (torch.sum(torch.abs(out_allp)**2,dim=1)).to(device)
            out_all = torch.reshape(out_allp,(-1,self.kx*self.ky,self.n_ue))/torch.unsqueeze(torch.sqrt(d),1)*P
        elif self.P=="randomize":
            P = (torch.abs(torch.randn((out_allp.shape)))**2+self.icounter/150)/self.n_ue
            P = P/torch.sqrt(torch.sum(P,axis=2)).reshape(P.shape[0],P.shape[1],1).repeat_interleave(out_allp.shape[2],dim=2)
            d = torch.sum(torch.sum(torch.abs(out_allp)**2,dim=1),dim=1).to(device)
            out_all = torch.reshape(out_allp,(-1,self.kx*self.ky,self.n_ue))/torch.reshape(torch.sqrt(d),(-1,1,1))*P
            self.icounter +=1
        elif self.P!=None:
            d = (torch.sum(torch.abs(out_allp)**2,dim=1)).to(device)
            P= self.P
            out_all = torch.reshape(out_allp,(-1,self.kx*self.ky,self.n_ue))/torch.unsqueeze(torch.sqrt(d),1)*P
        else:
            d = torch.sum(torch.sum(torch.abs(out_allp)**2,dim=1),dim=1).to(device)
            out_all = torch.reshape(out_allp,(-1,self.kx*self.ky,self.n_ue))/torch.reshape(torch.sqrt(d),(-1,1,1))

        #.repeat(1,self.kx*self.ky,self.n_ue)
        #t = torch.sum(torch.abs(out_all)**2)
        
        return out_all

class Radar_receiver(nn.Module):
    """ Detects radar targets and estimates positions (angles) at which the targets are present.
    
    input paramters:
        k: number of antennas of radar receiver (linear array kx1)

    output parameters:
        detect: bool whether radar target is present
        angle: estimated angle(s)
        uncertain: uncertainty of angle estimate

     """
    def __init__(self,kx,ky,max_target=1, cpr_max=1, encoding="counting", num_ue=1):
        super(Radar_receiver, self).__init__()
        self.k =torch.as_tensor([kx,ky]) # Number of transmit antennas; Paper k=16
        self.detect_offset = torch.zeros((cpr_max),device=device,requires_grad=False)
        self.targetnum = max_target
        self.encoding=encoding
        self.rad_detect = Radar_detect(k=self.k,max_target=max_target, encoding=encoding, num_ue=num_ue).to(device)
        self.rad_angle_est = Angle_est(k=self.k,max_target=max_target, num_ue=num_ue).to(device)
        self.num_ue= num_ue
        #self.rad_angle_est = Radar_tracking(k=self.k).to(device)
        #self.rad_angle_uncertain = Angle_est_uncertainty(k=self.k).to(device)
    
    def forward(self, c_k, targets=None, cpr=torch.tensor([1], device=device, dtype=torch.float32), noise=1):
        #cpr_max = int(torch.max(cpr))
        #cpr = torch.zeros(c_k.size()[1],1) + cpr
        detect = self.rad_detect(c_k).to(device)
        #detect = detect/cpr #And63 -> log-likelihood scales with number of samples
        #else:
            #detect = self.rad_detect(c_k,cpr)
        Pf =0.01 # maximum error probability
        if targets!=None:
            with torch.no_grad():
                for c in range(torch.max(cpr)):
                    if (c+1) in cpr:
                        select_detect = (torch.squeeze(cpr)==c+1).to(device) # cycle for each cpr
                        #xi = torch.nonzero((1-targets))
                        xi = torch.nonzero((1-targets)*torch.unsqueeze(select_detect,1).type(torch.int16))
                        #xj = torch.nonzero(select_detect.type(torch.int16))
                        t = detect[xi[:,0],xi[:,1]]
                        sorted_nod, idx = torch.sort(torch.squeeze(t))#/noise[x[:,0]]**2)
                        off = sorted_nod[int((1-Pf)*len(sorted_nod))].to(device) # highest LLR that needs pushing to 0 (+1)
                        if idx.numel():
                            self.detect_offset[c] = off 
                
            angle_est = self.rad_angle_est(c_k,targets,cpr=cpr,noise=noise)
        else:
            if noise!=None:
                try:
                    detect = detect - self.detect_offset[cpr.type(torch.long)-1].to(device)
                except:
                    detect = detect - self.detect_offset[cpr.detach().cpu()-1].to(device)
                #ylim = torch.tensor([np.sqrt(2*np.log2(1/Pf))],device=device)*noise
                #detect = detect -ylim
            else:
                detect = detect - self.detect_offset[cpr.type(torch.long)-1]
            #detect = torch.sigmoid(detect)
            angle_est = self.rad_angle_est(c_k,cpr=cpr,noise=noise)
        
        return(detect, angle_est)#, #angle_uncertain)


class Radar_detect(nn.Module):
    def __init__(self,k,max_target,encoding, num_ue=1):
        super(Radar_detect, self).__init__()
        self.k =torch.as_tensor(k) # Number of transmit antennas; Paper k=16
        self.d = torch.prod(k)
        self.targetnum = max_target
        self.num_ue= num_ue
        #layers target_detection
        self.fcB1a = nn.Linear(self.d*self.d*2+1,self.d*self.d*2*4, device=device)
        self.fcB1b = nn.Linear(self.d*self.d*2*4,self.d*self.d*2, device=device)
        self.fcB1c = nn.Linear(self.d*self.d*2,self.d*2, device=device)
        #self.fcB1d = nn.Linear(1,self.d*2, device=device)
        self.fcB2 = nn.Linear(self.d*2,self.d*2, device=device)
        self.fcB3 = nn.Linear(self.d*2,self.d, device=device)
        if encoding=='onehot':
            self.fcB4 = nn.Linear(self.d,self.targetnum+1, device=device)
        else:
            self.fcB4 = nn.Linear(self.d,self.targetnum, device=device) # linear output layer, add one for onehot encoding
        # Non-linearity (used in transmitter and receiver) ?
        self.activation_function = nn.ELU()
    
    def forward(self, c_k,cpr=torch.tensor([[1]], device=device, dtype=torch.float32), noise=None):
        if len(c_k)!=len(cpr):
            cpr=cpr.clone().repeat(len(c_k),1)
        detect = self.target_detection(c_k, cpr, noise)
        # fix false alarm rate to 0.01 in receiver
        return detect

    def target_detection(self, c_k, cpr, noise=None):
        x_in = torch.cat((torch.real(c_k),torch.imag(c_k),cpr),1)
        out = self.activation_function(self.fcB1a(x_in.type(torch.float32))).to(device)
        #out = torch.add(out,self.activation_function(self.fcB1b(torch.imag(c_k).type(torch.float32))).to(device))
        out = self.activation_function(self.fcB1b(out))
        out = self.activation_function(self.fcB1c(out))
        #out = self.activation_function(self.fcB2(out))
        out_2 = self.activation_function(self.fcB3(out))
        outx = (self.activation_function(self.fcB4(out_2)))
        #outx = torch.sigmoid(outx) # integrate sigmoid layer into loss function
        return outx

class Angle_est(nn.Module):
    def __init__(self,k,num_ue=1,max_target=1):
        super(Angle_est, self).__init__()
        self.k =torch.as_tensor(k) # Number of transmit antennas; Paper k=16
        self.d = torch.prod(k)
        self.num_targets = max_target # prevent large network
        self.num_ue = num_ue
        t=max_target
        #layers AoA est
        self.fcA1a = nn.Linear(self.d*self.d*2+2,self.d**2*4, device=device)
        self.fcA1b = nn.Linear(self.d**2*4,self.d**2*4*t, device=device)
        #self.fcA1b = nn.Linear(self.d**2,self.d*4*t, device=device)
        #self.fcA1c = nn.Linear(1,self.d*4*t, device=device)
        #self.fcA1d = nn.Linear(1,self.d*4*t, device=device)
        self.fcA2x = nn.Linear(self.d**2*4*t,self.d**2*4*t, device=device)
        self.fcA2 = nn.Linear(self.d**2*4*t,self.d*4*t, device=device)
        #self.fcA3 = nn.Linear(self.d*4*t,self.d*t*2, device=device)
        self.fcA3 = nn.Linear(self.d*4*t,self.d*t, device=device)
        self.fcA4 = nn.Linear(self.d*t,2*max_target, device=device) # linear output layer 
        # Non-linearity (used in transmitter and receiver) ?
        self.activation_function = nn.ELU()
    
    def angle_est(self, c_k,cpr, noise=None):
        x_in = torch.cat((torch.real(c_k),torch.imag(c_k),cpr,noise),1)
        #c_k = c_k.clone()/torch.sum(torch.abs(c_k)**2,dim=0)
        out = self.activation_function(self.fcA1a(x_in.type(torch.float32))).to(device)
        out = self.activation_function(self.fcA1b(out)).to(device)
        out = self.activation_function(self.fcA2x(out)).to(device)
        out = self.activation_function(self.fcA2(out))
        out_2 = self.activation_function(self.fcA3(out))
        outx = (self.activation_function(self.fcA4(out_2)))
        outx = np.pi/2*torch.tanh(outx) # now two angles, elevation and azimuth
        out_all = torch.reshape(outx,(-1,2,self.num_targets))
        return(out_all)
    
    def forward(self, c_k, targets=None, cpr=1, noise=None):
        if targets==None:
            angle = self.angle_est(c_k,cpr,noise=noise)
        else:
            targ = torch.nonzero(torch.squeeze(targets))
            angle = torch.zeros((targets.size()[0],2, self.num_targets)).to(device)
            cpr = cpr[targ[:,0]]
            if noise==None:
                pass
            else:
                noise = noise[targ[:,0]]
            if targ.numel():
                angle[targ[:,0]] = self.angle_est(c_k[targ[:,0]],cpr, noise=noise)

        return angle

class Joint_radar_receiver(nn.Module):
    """ Detects radar targets and estimates positions (angles) at which the targets are present.
    
    input paramters:
        k: number of antennas of radar receiver (linear array kx1)

    output parameters:
        detect: bool whether radar target is present
        angle: estimated angle(s)
        uncertain: uncertainty of angle estimate

     """
    def __init__(self,kx,ky,max_target=1, encoding="counting"):
        super(Joint_radar_receiver, self).__init__()
        self.k =torch.as_tensor([kx,ky]) # Number of transmit antennas; Paper k=16
        self.detect_offset = 0
        self.targetnum = max_target
        self.encoding=encoding
        self.rad_detect = Radar_detect(k=self.k,max_target=max_target, encoding=encoding).to(device)
        self.rad_angle_est = Angle_est(k=self.k,max_target=max_target).to(device)
        t = max_target
        self.d = torch.prod(self.k)
        #self.rad_angle_est = Radar_tracking(k=self.k).to(device)
        #self.rad_angle_uncertain = Angle_est_uncertainty(k=self.k).to(device)
        self.fcA1a = nn.Linear(self.d**2,self.d**2, device=device)
        self.fcA1b = nn.Linear(self.d**2,self.d**2, device=device)
        self.fcA1c = nn.Linear(1,self.d**2, device=device)
        self.fcA2x = nn.Linear(self.d**2,self.d**2, device=device)
        self.fcA2 = nn.Linear(self.d**2,self.d*4*t, device=device)
        self.fcA3 = nn.Linear(self.d*4*t,self.d*t, device=device)
        self.fcA4 = nn.Linear(self.d*t,2*max_target, device=device) # linear output layer angle
        self.fcB4 = nn.Linear(self.d*t,max_target, device=device) # linear output layer detect 
        self.activation_function = nn.ELU()
    
    def forward(self, c_k, targets=None, cpr=torch.tensor([1], device=device, dtype=torch.float32)):
        cpr = torch.unsqueeze(torch.zeros(c_k.size()[0],device=device) + cpr,1)

        out = self.activation_function(self.fcA1a(torch.real(c_k).type(torch.float32))).to(device)+self.activation_function(self.fcA1b(torch.imag(c_k).type(torch.float32))).to(device)+ self.activation_function(self.fcA1c(cpr)).to(device)
        #out = self.activation_function(self.fcA1a(c_k.type(torch.float32))).to(device)
        out = self.activation_function(self.fcA2x(out))
        out = self.activation_function(self.fcA2(out))
        out_2 = self.activation_function(self.fcA3(out))
        outx = (self.activation_function(self.fcA4(out_2)))
        outx = np.pi/2*torch.tanh(outx) # now two angles, elevation and azimuth
        angle_est = torch.reshape(outx,(-1,2,self.targetnum))
        detect = (self.activation_function(self.fcB4(out_2)))
        Pf =0.01 # maximum error probability
        if targets!=None:
            x = torch.nonzero((1-targets))
            t = detect[x[:,0],x[:,1]]
            sorted_nod, idx = torch.sort(torch.squeeze(t))
            if idx.numel():
                self.detect_offset = torch.mean(sorted_nod[int((1-Pf)*len(sorted_nod))])
            detect = detect - self.detect_offset
            #angle_est = self.rad_angle_est(c_k,targets,cpr=cpr)
        else:
            detect = detect - self.detect_offset
        
        return(detect, angle_est)#, #angle_uncertain)


def comm_loss_ra(rx,tx_bits,alpha):
    """
    weighted sum rate optimization reformulated as loss function
    max U = sum log(R_k), if apha=1,
            sum (R_k)^(1-alpha)/(1-alpha)

    input: float, shape: batch_size x M x num_ue
    R_k = (M-sum(BCE))
    """
    loss = nn.BCEWithLogitsLoss(reduction='none')
    logM = (rx[0].shape[1])

    if alpha==1:
        l=0
        for i in range(len(rx)):
            #l+= torch.mean(torch.sum(loss(rx[i],tx_bits[i]),dim=1),dim=0)
            l += -torch.log2(logM-torch.mean(torch.sum(loss(rx[i],tx_bits[i]),dim=1),dim=0))
            #l+= torch.sum(-torch.log2(1-torch.mean(loss(rx[i],tx_bits[i]),dim=0)+1e-5),dim=0)
    else:
        l=0
        for i in range(len(rx)):
            l += -torch.pow(torch.sum(logM-torch.mean(loss(rx[i],tx_bits[i]),dim=0),dim=0),1-alpha)/(1-alpha)
    return l
