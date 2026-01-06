#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import math
import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, update_model_inplace, test_inference
from utils import get_model, get_dataset, average_weights, exp_details, average_parameter_delta
import utils
from sklearn.cluster import kmeans_plusplus
import faiss
import scipy.stats as st
import math
def AMP_DA(y, X, H):
    H = H.cuda()
    y = y.cuda()
    X = X.cuda()

    N_RAs = H.shape[0]
    N_UEs = H.shape[1]
    N_dim = y.shape[1]
    N_M = y.shape[2]
    tol = 1e-5
    exx = 1e-10
    damp = 0.3
    alphabet = torch.arange(0.0, args.num_users * args.frac + 1, 1)
    M = len(alphabet) - 1

    lam = N_RAs / N_UEs
    c = torch.arange(0.01, 10, 10 / 1024)
    rho = (1 - 2 * N_UEs * ((1 + c ** 2) * st.norm.cdf(-c) - c * st.norm.pdf(c)) / N_RAs) / (
            1 + c ** 2 - 2 * ((1 + c ** 2) * st.norm.cdf(-c) - c * st.norm.pdf(c)))
    alpha = lam * torch.max(rho) * torch.ones((N_UEs, N_dim))
    x_hat0 = (alpha * torch.sum(alphabet) / M * torch.ones((N_UEs, N_dim)))[:, :, None].repeat(1, 1, N_M)
    x_hat = (x_hat0 + 1j * x_hat0).cuda()
    var_hat = torch.ones((N_UEs, N_dim, N_M)).cuda()

    V = torch.ones((N_RAs, N_dim, N_M)).cuda()
    V_new = torch.ones((N_RAs, N_dim, N_M)).cuda()
    Z_new = y.clone()
    sigma2 = 100
    t = 1
    Z = y.clone()
    maxIte = 50
    MSE = torch.zeros(maxIte)
    MSE[0] = 100
    hvar = (torch.norm(y) ** 2 - N_RAs * sigma2) / (N_dim * lam * torch.max(rho) * torch.norm(H) ** 2)
    hmean = 0
    alpha_new = torch.ones((N_UEs, N_dim, N_M))
    x_hat_new = torch.ones((N_UEs, N_dim, N_M)) + 1j * torch.ones((N_UEs, N_dim, N_M))
    var_hat_new = torch.ones((N_UEs, N_dim, N_M))

    hvarnew = torch.zeros(N_M)
    hmeannew = torch.zeros(N_M) + 1j * torch.zeros(N_M)
    sigma2new = torch.zeros(N_M)

    alphabet = alphabet.cuda()
    alpha = alpha.cuda()
    while t < maxIte:
        x_hat_pre = x_hat.clone()
        for i in range(N_M):
            V_new[:, :, i] = torch.abs(H) ** 2 @ var_hat[:, :, i]
            Z_new[:, :, i] = H @ x_hat[:, :, i] - ((y[:, :, i] - Z[:, :, i]) / (sigma2 + V[:, :, i])) * V_new[:, :, i]  # + 1e-8

            Z_new[:, :, i] = damp * Z[:, :, i] + (1 - damp) * Z_new[:, :, i]
            V_new[:, :, i] = damp * V[:, :, i] + (1 - damp) * V_new[:, :, i]

            var1 = (torch.abs(H) ** 2).T @ (1 / (sigma2 + V_new[:, :, i]))
            var2 = H.conj().T @ ((y[:, :, i] - Z_new[:, :, i]) / (sigma2 + V_new[:, :, i]))

            Ri = var2 / (var1) + x_hat[:, :, i]
            Vi = 1 / (var1)

            sigma2new[i] = ((torch.abs(y[:, :, i] - Z_new[:, :, i]) ** 2) / (
                        torch.abs(1 + V_new[:, :, i] / sigma2) ** 2) + sigma2 * V_new[:, :, i] / (
                                        V_new[:, :, i] + sigma2)).mean()

            if i == 0:
                r_s = Ri[None, :, :].repeat(M + 1, 1, 1) - alphabet[:, None, None].repeat(1, N_UEs, N_dim)
                pf8 = torch.exp(-(torch.abs(r_s) ** 2 / Vi)) / Vi / math.pi
                pf7 = torch.zeros((M + 1, N_UEs, N_dim)).cuda()
                pf7[0, :, :] = pf8[0, :, :] * (torch.ones((N_UEs, N_dim)).cuda() - alpha)
                pf7[1:, :, :] = pf8[1:, :, :] * (alpha / M)
                del pf8
                PF7 = torch.sum(pf7, axis=0)
                pf6 = pf7 / PF7
                del pf7, PF7
                AAA = alphabet[None, :, None].repeat(N_dim, 1, 1)
                BBB = torch.permute(pf6,(2,1,0))
                x_hat_new[:, :, i] = (torch.einsum("ijk,ikn->ijn", BBB, AAA).squeeze(-1)).T
                del AAA
                alphabet2 = alphabet ** 2
                AAA2 = alphabet2[None, :, None].repeat(N_dim, 1, 1)
                var_hat_new[:, :, i] = (torch.einsum("ijk,ikn->ijn", BBB, AAA2).squeeze(-1)).T.cpu() - torch.abs(
                    x_hat_new[:, :, i]) ** 2
                del AAA2
                alpha_new[:, :, i] = torch.clamp(torch.sum(pf6[1:, :, :], axis=0), exx, 1 - exx)
                del pf6
            else:
                A = (hvar * Vi) / (Vi + hvar)
                B = (hvar * Ri + Vi * hmean) / (Vi + hvar)
                lll = torch.log(Vi / (Vi + hvar)) / 2 + torch.abs(Ri) ** 2 / 2 / Vi - torch.abs(Ri - hmean) ** 2 / 2 / (
                            Vi + hvar)
                pai = torch.clamp(alpha / (alpha + (1 - alpha) * torch.exp(-lll)), exx, 1 - exx, out=None)
                x_hat_new[:, :, i] = pai * B
                var_hat_new[:, :, i] = (pai * (torch.abs(B) ** 2 + A)).cpu() - torch.abs(x_hat_new[:, :, i]) ** 2
                # mean update
                hmeannew[i] = (torch.sum(pai * B, axis=0) / torch.sum(pai, axis=0)).mean()
                # variance update
                hvarnew[i] = (torch.sum(pai * (torch.abs(hmean - B) ** 2 + Vi), axis=0) / torch.sum(pai, axis=0)).mean()
                # activity indicator update
                alpha_new[:, :, i] = torch.clamp(pai, exx, 1 - exx)
        if N_M > 1:
            hvar = hvarnew[1:].mean()
            hmean = hmeannew[1:].mean()
        sigma2 = sigma2new.mean()
        alpha = (torch.sum(alpha_new, axis=2) / N_M).cuda()
        # alpha = alpha_new
        III = x_hat_pre.cpu() - x_hat_new
        NMSE_iter = torch.sum(torch.abs(III) ** 2) / torch.sum(torch.abs(x_hat_new) ** 2)
        # del III
        MSE[t] = torch.sum(torch.abs(y - torch.permute(
            torch.einsum("ijk,ikn->ijn", torch.permute(x_hat, (2, 1, 0)), H.T[None, :, :].repeat(N_M, 1, 1)),
            (2, 1, 0))) ** 2) / N_RAs / N_dim / N_M

        x_hat = x_hat_new.cuda().clone()
        if t > 15 and MSE[t] >= MSE[t - 1]:
            x_hat = x_hat_pre.clone()
            break

        NMSE = 10 * math.log10(torch.sum(torch.abs(x_hat[:, :, 0] - X[:, :, 0]) ** 2) / torch.sum(torch.abs(X[:, :, 0]) ** 2))

        var_hat = var_hat_new.cuda().clone()
        # alpha = alpha_new
        V = V_new.clone()
        Z = Z_new.clone()
        t = t + 1
    return x_hat, var_hat, alpha, t, NMSE


def UMA_MIMO(MatInd, args):
    tau = 1   # factor of channel imperfection
    Na = args.antenna   # Number of antennas at BS
    Ka = MatInd.shape[0]
    h_a = (np.random.randn(Ka, Na).astype(np.float32) + 1j * np.random.randn(Ka, Na).astype(np.float32)) / np.sqrt(2)
    e = (np.random.randn(Ka).astype(np.float32) + 1j * np.random.randn(Ka).astype(np.float32)) / np.sqrt(2)
    h_e = tau*h_a[:, 0] + np.sqrt(1-tau)*e
    h_d = (1 / h_e)*h_a.T
    if tau == 1:
        h_d[0, :] = np.ones([1, Ka]).astype(np.float32)
    Np = MatInd.shape[1]  # Number of SMV problems

    Ph = abs(h_e)
    IdPh = np.where(Ph < 0.14)
    h_d[:, IdPh] = 0

    X_eq = np.zeros((args.M, Np, Na)).astype(np.float32) + 1j * np.zeros((args.M, Np, Na)).astype(np.float32)
    I1tmp = np.arange(Np)
    for i in range(Ka):
        I0tmp = MatInd[i, :]
        X_eq[I0tmp, I1tmp, :] = X_eq[I0tmp, I1tmp, :] + h_d[:, i]


    Y0 = np.einsum("ijk,ikn->ijn", X_eq.T, args.UM.T[np.newaxis, :, :].repeat(Na, axis=0)).T

    Ps = np.linalg.norm(Y0.reshape(1, -1), ord='fro') ** 2 / Np / args.L/ Na
    snr = 10 ** (args.SNR / 10)
    Pn = Ps / snr
    Y = Y0 + (math.sqrt(Pn / 2) * np.random.randn(args.L, Np, Na).astype(np.float32) +  1j * np.random.randn(args.L, Np, Na).astype(np.float32))

    # if args.Algo == 'AMP':
    H = torch.tensor(args.UM)
    Y = torch.tensor(Y)
    X_eq = torch.tensor(X_eq)
    x_hat, var_hat, alpha, t, NMSE = AMP_DA(Y, X_eq, H)

    return x_hat, alpha, NMSE, X_eq[:, :, 0]

if __name__ == '__main__':
    start_time = time.time()

    # parse args
    args = args_parser()
    args.seed = 42
    args.M = 2**6  # quantization levels
    args.Vb = 20   # diemnsion of each vector quantization
    args.iid = 0 # non i.i.d. data distribution
    args.L = 20 # length of each transmit codeword
    UMmat = args.UM[:args.L, :] # transmit codebook matrix
    args.UM = UMmat
    args.antenna = 2 # number of antennas at BS
    args.Nummm = 10000 # number of data samples for data split
    args.epochs = 1000 # number of global rounds
    args.model = 'resnet-s' # model name
    exp_details(args)
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # define paths
    file_name = '/Results_MIMOchannel_seed{}_Na{}_L{}_{}_{}_llr[{}]_glr[{}]_Vb[{}]_le[{}]_bs[{}]_iid[{}]_Ql[{}]_frac[{}]_{}.pkl'.\
                format(args.seed, args.antenna, args.L, args.model, args.optimizer,
                    args.local_lr, args.lr, args.Vb,
                    args.local_ep, args.local_bs, args.iid, args.M, args.frac, args.compressor)
    logger = SummaryWriter('./logs/'+file_name)
    
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1) # limit cpu use
    print ('-- pytorch version: ', torch.__version__)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device != 'cpu':
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.outfolder):
        os.mkdir(args.outfolder)

    # load dataset and user groups
    train_dataset, test_dataset, num_classes, user_groups = get_dataset(args)

    # Set the model to train and send it to device.
    global_model = get_model(args.model, args.dataset, train_dataset[0][0].shape, num_classes)
    global_model.to(device)
    global_model.train()
    
    momentum_buffer_list = []
    exp_avgs = []
    exp_avg_sqs = []
    max_exp_avg_sqs = [] 
    for i, p in enumerate(global_model.parameters()):         
        momentum_buffer_list.append(torch.zeros_like(p.data.detach().clone(), dtype=torch.float, requires_grad=False))
        exp_avgs.append(torch.zeros_like(p.data.detach().clone(), dtype=torch.float, requires_grad=False))
        exp_avg_sqs.append(torch.zeros_like(p.data.detach().clone(), dtype=torch.float, requires_grad=False))
        max_exp_avg_sqs.append(torch.zeros_like(p.data.detach().clone(), dtype=torch.float, requires_grad=False)+args.max_init) # 1e-2

    ### init error -------
    e = []
    for id in range(args.num_users):
        ei = []
        for i, p in enumerate(global_model.parameters()):         
            ei.append(torch.zeros_like(p.data.detach().clone(), dtype=torch.float, requires_grad=False))
        e.append(ei)
    D = sum(p.numel() for p in global_model.parameters())
    print('total dimension:', D)
    print('compressor:', args.compressor)

    # Check if divisibility condition is satisfied
    s_nExtraZero = D % args.Vb
    if s_nExtraZero != 0:
        s_nExtraZero = args.Vb - s_nExtraZero

    # initialize error feedback
    Qerr = torch.ones((int(args.frac * args.num_users), args.epochs))

    # Training
    train_loss_sampled, train_loss, train_accuracy = [], [], []
    test_loss, test_accuracy = [], []
    start_time = time.time()
    NMSEtot = np.zeros(args.epochs)
    KEST0 = []
    KEST =[]
    KACT = []
    for epoch in tqdm(range(args.epochs)):
        ep_time = time.time()
        
        local_weights, local_params, local_losses = [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        
        
        par_before = []
        for p in global_model.parameters():  # get trainable parameters
            par_before.append(p.data.detach().clone())
        # this is to store parameters before update
        w0 = global_model.state_dict()  # get all parameters, including batch normalization related ones
        
        
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # user for loop
        for ooo in range(len(idxs_users)):
            idx = idxs_users[ooo]
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            
            w, p, loss = local_model.update_weights_local(
                model=copy.deepcopy(global_model), global_round=epoch)
            

            ####### add error feedback #######
            delta = utils.sub_params(p, par_before)
            tmp = utils.add_params(e[idx], delta)

            for i in range(len(tmp)):
                if i == 0:
                    VecTmp = tmp[i].cpu().reshape(1, -1)
                else:
                    VecTmp = torch.cat((VecTmp, tmp[i].cpu().reshape(1, -1)), 1)
            VecTmp = torch.cat((VecTmp, torch.zeros((1, s_nExtraZero))), 1)

            # Vector Quantization
            if ooo == 0:
                data2 = VecTmp.numpy().reshape(-1, args.Vb)
                centers, _ = kmeans_plusplus(data2, n_clusters=args.M, random_state=0)
                index = faiss.IndexFlatL2(args.Vb)
                index.add(centers)
                _, intIndex = index.search(VecTmp.numpy().reshape(-1, args.Vb), 1)
                SigVQ0 = torch.tensor(centers[intIndex].reshape(1, -1))
            else:
                _, intIndex = index.search(VecTmp.numpy().reshape(-1, args.Vb), 1)
                SigVQ0 = torch.tensor(centers[intIndex].reshape(1, -1))
            if ooo == 0:
                MatInd = intIndex.T
            else:
                MatInd = np.vstack([MatInd, intIndex.T])
            del intIndex
            # Quantization error
            Qerr[ooo, epoch] = 10 * torch.log10(torch.sum((torch.abs(SigVQ0 - VecTmp) ** 2)) / torch.sum((torch.abs(VecTmp) ** 2)))
            if s_nExtraZero == 0:
                SigVQ = SigVQ0
            else:
                SigVQ = SigVQ0[:, :-s_nExtraZero]
            # Delta update
            delta_out = []
            L = 0
            for i in range(len(tmp)):
                Ll = L + tmp[i].numel()
                delta_out.append(SigVQ[:, L:Ll].reshape(tmp[i].shape).cuda())
                L = Ll
            del SigVQ
            # Error update
            e[idx] = utils.sub_params(tmp, delta_out)
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # Transmission through MIMO channel
        x_hat, alpha, NMSE, X_eq = UMA_MIMO(MatInd, args)
        x_hat = x_hat.cpu().numpy()
        X_eq = X_eq.cpu().numpy() # actual transmitted signal

        Kact = int(np.real(np.sum(X_eq, 0)[0])) # actual number of active users
        KACT.append(Kact)

        NMSEtot[epoch] = NMSE
        temp = pd.Series(np.sum(abs(np.around(x_hat[:, :, 0])), axis=0))
        Cont = temp.value_counts()
        Kest = int(Cont.keys()[0]) # estimated number of active users (proposed)
        KEST.append(Kest)
        Kest0 = int(np.mean(temp))  # estimated number of active users (benchmark)
        KEST0.append(Kest0)
        Est_delta = (np.abs(np.round(x_hat[:, :, 0].T)) @ centers).reshape(-1, 1).T / Kest

        # sparsity level for the AMP-DA algorithm
        Sptot = np.count_nonzero(X_eq, axis=0)
        if epoch == 0:
            SpLev = Sptot
        else:
            SpLev = np.vstack((SpLev, Sptot))

        # Check if divisibility condition is satisfied
        if s_nExtraZero == 0:
            Est_delta = Est_delta
        else:
            Est_delta = Est_delta[:, :-s_nExtraZero]


        global_delta = []
        L = 0
        for i in range(len(tmp)):
            Ll = L + tmp[i].numel()
            global_delta.append(torch.tensor(Est_delta[:, L:Ll]).reshape(tmp[i].shape).cuda())
            L = Ll

        bn_weights = average_weights(local_weights)
        global_model.load_state_dict(bn_weights)

        
        update_model_inplace(
            global_model, par_before, global_delta, args, epoch, 
            momentum_buffer_list, exp_avgs, exp_avg_sqs, max_exp_avg_sqs)
        
        # report and store loss and accuracy
        # this is local training loss on sampled users
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        global_model.eval()
        
        # Test inference after completion of training
        test_acc, test_ls = test_inference(args, global_model, test_dataset)
        test_accuracy.append(test_acc)
        test_loss.append(test_ls)

        # print global training loss after every rounds
        print('Epoch Run Time: {0:0.4f} of {1} global rounds'.format(time.time()-ep_time, epoch+1))
        print(f'Training Loss : {train_loss[-1]}')
        print(f'Test Loss : {test_loss[-1]}')
        print(f'Test Accuracy : {test_accuracy[-1]} \n')
        print(f'NMSE UMA : {NMSE} \n')
        logger.add_scalar('train loss', train_loss[-1], epoch)
        logger.add_scalar('test loss', test_loss[-1], epoch)
        logger.add_scalar('test acc', test_accuracy[-1], epoch)
        
        if args.save:
            # Saving the objects train_loss and train_accuracy:
            

            with open(args.outfolder + file_name, 'wb') as f:
                pickle.dump([train_loss, test_loss, test_accuracy, Qerr, SpLev, NMSEtot, KEST, KEST0, KACT], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    
