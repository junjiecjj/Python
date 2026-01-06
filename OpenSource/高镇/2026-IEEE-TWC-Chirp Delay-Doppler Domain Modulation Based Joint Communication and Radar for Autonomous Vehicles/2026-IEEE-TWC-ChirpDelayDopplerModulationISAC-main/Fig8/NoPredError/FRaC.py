
import os, json, numpy as np, matplotlib.pyplot as plt, torch
from tqdm.auto import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ───────────────────────── 1) 基础函数 ─────────────────────────
def GeneChannel(batch_size, device):
    '''
    流程和功能：
    1. 直接生成中频信号(0~20MHz均匀分布的单音信号，1024点)
    2. 生成信道的发射角和接收角(水平发射角和水平接收角在±60°，垂直发射角和垂直接收角在±15°)

    输出：
    1. IFsignal: torch.Size([batch_size, 1024])
    2. TxAngle: torch.Size([batch_size, 2]) --- 水平发射角和垂直发射角
    '''    
    f_if, fs = 5_000_000.0, 20_000_000
    t = torch.arange(1024, device=device) / fs
    base = torch.exp(1j * 2 * torch.pi * f_if * t).to(torch.complex64)
    IFsignal = base.unsqueeze(0).repeat(batch_size, 1)
    TxAngle = torch.empty(batch_size, 2, device=device)
    TxAngle[:, 0].uniform_(-60.0, 60.0)   # θ_h
    TxAngle[:, 1].uniform_(-15.0, 15.0)   # θ_v
    return IFsignal, TxAngle

def QPSK_modulate(bits):
    """
    对比特流进行 QPSK 调制。
    输入：bits - torch.Size([batch_size, 512, 2])  QPSK 每个中频子段需要两位比特进行调制
    输出：调制后的 QPSK 信号
    """
    b = bits.to(torch.float32)
    I = 1.0 - 2.0 * b[..., 0]
    Q = 1.0 - 2.0 * b[..., 1]
    return torch.complex(I, Q) / np.sqrt(2.0)

def add_awgn(tx, snr_db):
    Ps = tx.abs().pow(2).mean(1, keepdim=True)
    sigma = torch.sqrt(Ps / (10.0 ** (snr_db / 10.0) * 2.0))
    noise = torch.complex(torch.randn_like(tx.real), torch.randn_like(tx.real)) * sigma
    return tx + noise

# ─────────────────── 2) 全局常量（先在 CPU，使用时转设备） ───────────────────
_pairs6 = torch.tensor([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]], dtype=torch.long)  # 6 个天线对
# 选用 6 种中的 4 种来承载 2bit：这里选择索引 [1,2,4,5] → 对应对 {0,2},{0,3},{1,3},{2,3}
_pairs4_idx = torch.tensor([0,1,4,5], dtype=torch.long)
_pairs4 = _pairs6[_pairs4_idx]  # shape (4,2)

# ULA：4 阵元水平排列
_mh = torch.tensor([0,1,0,1], dtype=torch.float32)  # 水平索引
_mv = torch.tensor([0,0,1,1], dtype=torch.float32)  # 垂直索引（ULA 为 0）
# _mh = torch.tensor([0,1,2,3], dtype=torch.float32)  # 水平索引
# _mv = torch.tensor([0,0,0,0], dtype=torch.float32)  # 垂直索引（ULA 为 0）

_bpairs = torch.tensor([[0,0],[0,1],[1,1],[1,0]], dtype=torch.float32)  # Gray 顺序
_qconst_cpu = QPSK_modulate(_bpairs)            # shape(4,), complex64
_qconj_cpu  = _qconst_cpu.conj()

def _array_factor(TxAngle, device):
    """返回阵列方向因子 (B,4)，并确保在 device 上"""
    θh, θv = TxAngle.t() * (torch.pi/180.0)
    kh, kv = torch.sin(θh)*torch.cos(θv), torch.sin(θv)
    mh = _mh.to(device)
    mv = _mv.to(device)
    phase = 4*torch.pi*(kh.unsqueeze(1)*mh + kv.unsqueeze(1)*mv)  # d=2λ
    return torch.polar(torch.ones_like(phase, device=device), phase).to(torch.complex64)  # (B,4)

# ─────────────────── 3) 发送 / 接收 ───────────────────
def GeneRxSig_modular(B, TxBits, IFs, TxAng, snr_db, mode='all'):
    '''
    位分配（本版按“2bit 承载 4 种天线对”）：
      bit0~1 : 2bit → 4 种天线对（从 6 种中挑选的 4 种）
      bit2   : 保留位（恒置 0，保持与旧代码 8bit 结构兼容）
      bit3   : swap（两段中频与两根天线是否交换）
      bit4~5 : 第一个中频子段 QPSK（Gray）
      bit6~7 : 第二个中频子段 QPSK（Gray）

    输出：
      RxSig:       torch.Size([batch_size, 1024])
      TxBits_eff:  torch.Size([batch_size, 8])  —— 写回后的“有效发送比特”（bit2 恒为 0）
    '''    
    dev = IFs.device

    # —— 天线对选择：仅 2bit（4 态） ——
    if mode in ['spatial','all']:
        idx2 = (TxBits[:,0].to(torch.int64)<<1) | TxBits[:,1].to(torch.int64)   # 0..3
        ants = _pairs4.to(dev)[idx2]                                           # (B,2)
        swap = TxBits[:,3].bool()

        # 写回 2bit 索引（并把 bit2 置 0），形成“有效发送比特”
        TxBits_eff = TxBits.clone()
        TxBits_eff[:,0:2] = (((idx2.unsqueeze(1)) >> torch.tensor([1,0], device=dev)) & 1).to(torch.uint8)
        TxBits_eff[:,2]   = 0  # 保留位
        # 交换映射
        ant1 = torch.where(swap, ants[:,1], ants[:,0])
        ant2 = torch.where(swap, ants[:,0], ants[:,1])
    else:
        # 非空间模式：保持原比特
        TxBits_eff = TxBits
        ant1 = torch.zeros(B, device=dev, dtype=torch.long)
        ant2 = torch.ones (B, device=dev, dtype=torch.long)

    ph = _array_factor(TxAng, dev)
    ph1 = ph.gather(1, ant1.view(-1,1)).squeeze(1)
    ph2 = ph.gather(1, ant2.view(-1,1)).squeeze(1)

    IF1, IF2 = IFs[:,:512], IFs[:,512:]
    if mode in ['qpsk','all']:
        s1 = QPSK_modulate(torch.stack([TxBits[:,4],TxBits[:,5]],1)).unsqueeze(1).repeat(1,512)
        s2 = QPSK_modulate(torch.stack([TxBits[:,6],TxBits[:,7]],1)).unsqueeze(1).repeat(1,512)
    else:  # pilot
        pilot = torch.ones(B,512,device=dev,dtype=torch.complex64)/np.sqrt(2.0)
        s1 = s2 = pilot

    tx = torch.cat([IF1*s1*ph1.unsqueeze(1), IF2*s2*ph2.unsqueeze(1)],1)
    return add_awgn(tx, snr_db), TxBits_eff

def Demod_modular(B, RxSig, IFs, TxAng, mode='all'):
    '''
    MLE 解调（与旧版保持基本结构，仅把“6 对 → 4 对”）：
      - spatial: 只在 4 对 × swap(2) 里打分
      - all    : 在 4 对 × swap(2) × q1(4) × q2(4) 里打分
      - qpsk   : 同旧版，只解调两个子段的 QPSK
    返回：DemodBits uint8, shape = [B, 8]
    '''
    dev = RxSig.device
    IF1,Rx1,IF2,Rx2 = IFs[:,:512],RxSig[:,:512], IFs[:,512:],RxSig[:,512:]
    C1 = (Rx1*IF1.conj()).sum(1)   # (B,)
    C2 = (Rx2*IF2.conj()).sum(1)   # (B,)
    phc = _array_factor(TxAng, dev).conj()  # (B,4)

    qconj = _qconj_cpu.to(dev)     # (4,)
    bpairs = _bpairs.to(dev)       # (4,2)

    if mode in ['all']:
        pv = phc[:, _pairs4.to(dev).t()].permute(0,2,1)   # (B,4,2)
        # 评分：四种天线对 × swap(2) × q1(4) × q2(4)
        # 不交换
        t1 = C1[:,None,None,None] * pv[:,:,0][:,:,None,None] * qconj[None,None,:,None]
        t2 = C2[:,None,None,None] * pv[:,:,1][:,:,None,None] * qconj[None,None,None,:]
        scr0 = (t1 + t2).real  # (B,4,4,4)
        # 交换
        t1s = C1[:,None,None,None] * pv[:,:,1][:,:,None,None] * qconj[None,None,:,None]
        t2s = C2[:,None,None,None] * pv[:,:,0][:,:,None,None] * qconj[None,None,None,:]
        scr1 = (t1s + t2s).real  # (B,4,4,4)

        scr = torch.stack([scr0, scr1], dim=2).reshape(B, -1)  # (B, 4*2*4*4 = 128)
        best = scr.argmax(1)

        q2 = best % 4
        t  = best // 4
        q1 = t % 4
        s  = (t // 4) % 2
        cd = t // 8                       # 0..3（对应 _pairs4 的索引）
        out = torch.zeros(B,8,device=dev,dtype=torch.uint8)
        # 写 2bit（cd）到 bit0~1，bit2 置 0
        out[:,0:2] = (cd[:,None] >> torch.tensor([1,0], device=dev)) & 1
        out[:,2]   = 0
        out[:,3]   = s.to(torch.uint8)
        out[:,4:6] = bpairs[q1.long()].to(torch.uint8)
        out[:,6:8] = bpairs[q2.long()].to(torch.uint8)

    elif mode=='spatial':
        pv = phc[:, _pairs4.to(dev).t()].permute(0,2,1)   # (B,4,2)
        scr0 = (C1[:,None]*pv[:,:,0] + C2[:,None]*pv[:,:,1]).real  # (B,4)
        scr1 = (C1[:,None]*pv[:,:,1] + C2[:,None]*pv[:,:,0]).real  # (B,4)
        scr  = torch.stack([scr0, scr1], dim=2).reshape(B, -1)      # (B,8) = 4×swap(2)

        best = scr.argmax(1)
        s  = best % 2
        cd = best // 2                     # 0..3
        out  = torch.zeros(B,8,device=dev,dtype=torch.uint8)
        out[:,0:2] = (cd[:,None] >> torch.tensor([1,0], device=dev)) & 1
        out[:,2]   = 0
        out[:,3]   = s.to(torch.uint8)

    elif mode=='qpsk':
        # 先做每路的已知相位补偿
        z1 = C1 * phc[:, 0]
        z2 = C2 * phc[:, 1]
        scores1 = (z1[:, None] * qconj).real   # (B,4)
        scores2 = (z2[:, None] * qconj).real   # (B,4)
        best1 = scores1.argmax(1)
        best2 = scores2.argmax(1)

        out  = torch.zeros(B, 8, device=dev, dtype=torch.uint8)
        out[:, 4:6] = bpairs[best1]
        out[:, 6:8] = bpairs[best2]

    else:   # 'none'
        out = torch.zeros(B,8,device=dev,dtype=torch.uint8)
    return out

def CalErr(B, TxBits, Demod, acc):
    diff = (TxBits != Demod)
    acc['BER'] += diff.sum().item()             # 全 8 位（其中 bit2 恒为 0，与 BER 分母配合使用）
    acc['SER'] += diff.any(1).sum().item()
    acc['BER_QPSK'] += diff[:,4:8].sum().item()
    acc['SER_QPSK'] += diff[:,4:8].any(1).sum().item()
    acc['BER_AntSel'] += diff[:,:2].sum().item()             # 2 bit（原来是 :3）
    acc['SER_AntSel'] += diff[:,:2].any(1).sum().item()
    acc['BER_perm'] += diff[:,3].sum().item()
    acc['SER_perm'] += diff[:,3].sum().item()
    return acc

# ───────────────────────── 4) 主流程 ─────────────────────────
if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(2025)
    SNRdBs = torch.arange(-30, 0, 5, device=dev)
    IterAll, B = 10_000, 1_000
    Iters = IterAll//B
    modes = ['spatial','qpsk','all']
    res = {m:{int(s):{'BER':0,'SER':0,'BER_AntSel':0,'SER_AntSel':0,
                      'BER_QPSK':0,'SER_QPSK':0,'BER_perm':0,'SER_perm':0}
              for s in SNRdBs.tolist()} for m in modes}
    with torch.no_grad():
        for snr in tqdm(SNRdBs, desc='SNR'):
            s = int(snr)
            for mode in modes:
                for _ in range(Iters):
                    # 仍然生成 8 位，比特位布局见 GeneRxSig_modular 注释
                    TxBits = torch.randint(0,2,(B,8),device=dev,dtype=torch.uint8)
                    # 为避免“保留位误差”，把 bit2 强制置 0（便于与解调输出对齐）
                    TxBits[:,2] = 0

                    IFs, Ang = GeneChannel(B,dev)
                    Rx, TxBits_eff  = GeneRxSig_modular(B,TxBits,IFs,Ang,snr,mode)
                    Dem = Demod_modular(B,Rx,IFs,Ang,mode)
                    res[mode][s] = CalErr(B,TxBits_eff,Dem,res[mode][s])
    with open('FRaC.json','w',encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print('Simulation finished → FRaC.json')

    # 可视化（可选）
    plt.style.use('default')
    snrs = [x for x in SNRdBs.tolist()]
    mk = {'none':'x','spatial':'s','qpsk':'^','all':'o'}

    # —— BER ——
    plt.figure(figsize=(10,7))
    for m in modes:
        if m == 'qpsk':
            ber = np.array([res[m][s]['BER_QPSK']   for s in snrs]) / (IterAll * 4)  # 4bit
        elif m == 'spatial':
            ber = np.array([res[m][s]['BER_AntSel'] for s in snrs]) / (IterAll * 2)  # 2bit
        elif m == 'all':
            ber = np.array([res[m][s]['BER']        for s in snrs]) / (IterAll * 7)  # 有效 7bit（bit2 为保留位）
        else:
            continue
        ber[ber == 0] = np.nan
        plt.semilogy(snrs, ber, marker=mk[m], label=m)

    plt.xlabel('SNR (dB)'); plt.ylabel('BER'); plt.grid(True,which='both'); plt.ylim(1e-5,1)
    plt.legend(); plt.tight_layout(); 

    # —— SER ——
    plt.figure(figsize=(10,7))
    for m in modes:
        if m == 'qpsk':
            ser = np.array([res[m][s]['SER_QPSK']   for s in snrs]) / (IterAll)
        elif m == 'spatial':
            ser = np.array([res[m][s]['SER_AntSel'] for s in snrs]) / (IterAll)
        elif m == 'all':
            ser = np.array([res[m][s]['SER']        for s in snrs]) / (IterAll)
        else:
            continue
        ser[ser == 0] = np.nan
        plt.semilogy(snrs, ser, marker=mk[m], label=m)

    plt.xlabel('SNR (dB)')
    plt.ylabel('SER')
    plt.grid(True, which='both')
    plt.ylim(1e-5, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()
