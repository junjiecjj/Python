#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Low‑memory mutual‑information Monte‑Carlo for delay‑Doppler Index‑QAM
"""
import os, time, json
import numpy as np
import torch
from typing import Tuple, List, Any, Dict

# ---------- utilities --------------------------------------------------------
def complex_randn(*size, device):
    """i.i.d. ℂ𝓝(0,1) / √2"""
    return (torch.randn(*size, device=device) +
            1j * torch.randn(*size, device=device)) / np.sqrt(2)

def qam_constellation_torch(Mc: int, device) -> torch.Tensor:
    """Square QAM (Mc=4/64/256…) normalized to average power 1"""
    m = int(np.sqrt(Mc))
    pts = torch.arange(-(m-1), m, 2, device=device)
    const = torch.cartesian_prod(pts, pts.flip(0)).float()
    const = const[:, 0] + 1j * const[:, 1]
    const /= torch.sqrt((const.abs()**2).mean())
    return const.to(torch.cfloat)                                  # (Mc,)

def _make_params_dict(**kwargs) -> Dict[str, Any]:
    """把所有运行参数整理成可 JSON 化的字典，用来校验 checkpoint 是否可继续用。"""
    out = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, torch.Tensor):
            out[k] = v.cpu().tolist()
        else:
            out[k] = v
    return out

# ---------- implicit codebook -----------------------------------------------
class IndexQAM:
    """Codebook C = { e_i·s | i∈[0,N), s∈𝒮_QAM } stored implicitly."""
    def __init__(self, N: int, constellation: torch.Tensor):
        self.N  = N
        self.C  = constellation
        self.Mc = constellation.numel()
        self.M  = N * self.Mc                                       # total codewords

    def sample(self, num: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uniformly sample `num` codewords. Return (index, symbol)."""
        idx  = torch.randint(self.N, (num,), device=device)         # (num,)
        symI = torch.randint(self.Mc, (num,), device=device)
        return idx, self.C[symI]                                    # (num,), (num,)
class IndexOnly:
    """仅索引调制：符号恒为 1。M = N."""
    def __init__(self, N: int, device):
        self.N  = N
        self.C  = torch.tensor([1+0j], dtype=torch.cfloat, device=device)  # 单点
        self.Mc = 1
        self.M  = N

    def sample(self, num: int, device):
        idx = torch.randint(self.N, (num,), device=device)
        sym = self.C.expand(num)    # (num,)
        return idx, sym

class QAMOnly:
    """仅 QAM 调制：索引恒为 0。M = Mc."""
    def __init__(self, constellation: torch.Tensor):
        self.N  = 1                                     # 只用第 0 列
        self.C  = constellation                         # (Mc,)
        self.Mc = constellation.numel()
        self.M  = self.Mc

    def sample(self, num: int, device):
        idx  = torch.zeros(num, device=device, dtype=torch.long)  # 全是 0
        symI = torch.randint(self.Mc, (num,), device=device)
        return idx, self.C[symI]
    
# ---------- column‑on‑demand delay‑Doppler channel ---------------------------
class LowMemDDChannel:
    r"""Delay‑Doppler H_DD = Σ_l kron(H_t(l)^T , H_f(l)).

    Memory: O(L·(N_f²+N_c²)).  Access: column(j) in O(L·N_f·N_c).
    """
    def __init__(self, L: int, N_f: int, N_c: int, device):
        self.Nf, self.Nc, self.L = N_f, N_c, L
        self.device = device
        self.scale  = 1 / np.sqrt(L)

        # pre‑compute per‑path small matrices
        self.H_f: List[torch.Tensor] = []
        self.H_t: List[torch.Tensor] = []
        for _ in range(L):
            k_f = torch.randint(0, N_f, ()).item()
            k_t = torch.randint(0, N_c, ()).item()

            lam_f = torch.exp(2j * torch.pi * torch.arange(N_f, device=device) * k_f / N_f)
            lam_t = torch.exp(2j * torch.pi * torch.arange(N_c, device=device) * k_t / N_c)

            F_f = torch.fft.fft(torch.eye(N_f, device=device)) / np.sqrt(N_f)
            F_t = torch.fft.fft(torch.eye(N_c, device=device)) / np.sqrt(N_c)

            self.H_f.append(F_f.conj().T @ torch.diag(lam_f) @ F_f)    # (N_f,N_f)
            self.H_t.append(F_t @ torch.diag(lam_t) @ F_t.conj().T)    # (N_c,N_c)

    @property
    def N(self):
        return self.Nf * self.Nc

    def column(self, j: int) -> torch.Tensor:
        """Return j‑th column as shape (N,) complex tensor."""
        j_f = j % self.Nf
        j_t = j // self.Nf
        col = torch.zeros(self.N, dtype=torch.cfloat, device=self.device)
        for Hf, Ht in zip(self.H_f, self.H_t):
            a = Hf[:, j_f]            # (N_f,)
            b = Ht.T[:, j_t]          # (N_c,)
            col += (a[:, None] * b[None, :]).reshape(-1)
        return col * self.scale 

# ---------- MI estimator -----------------------------------------------------
def mi_mc_gpu_lowmem(H: LowMemDDChannel,
                     code: IndexQAM,
                     snr_lin: float,
                     n_samples: int = 2_000,
                     batch: int = 2_000,
                     col_batch: int = 4_096) -> float:
    r"""Low‑memory Monte‑Carlo estimator of I(X;Y)."""
    device = H.device
    N, Mc, M = H.N, code.Mc, code.M
    sigma2   = 1.0 / snr_lin
    log_M    = np.log(M)

    sum_est = 0.0
    for off in range(0, n_samples, batch):
        bsz          = min(batch, n_samples - off)
        idx, sym     = code.sample(bsz, device)              # (bsz,), (bsz,)
        # gather channel columns for chosen indices
        h_cols       = torch.stack([H.column(int(i)) for i in idx])   # (bsz,N)
        noise        = complex_randn(bsz, N, device=device) * np.sqrt(sigma2)
        y            = sym[:, None] * h_cols + noise                 # (bsz,N)
        y_norm2      = (y.abs()**2).sum(-1)                         # (bsz,)

        # accumulate log‑sum‑exp in a numerically stable way
        lse = torch.full((bsz,), -torch.inf, device=device)

        for s in code.C:                        # iterate constellation points
            for j0 in range(0, N, col_batch):   # iterate over code indices in chunks
                j1   = min(j0 + col_batch, N)
                cols = torch.stack([H.column(j) for j in range(j0, j1)])   # (m,N)
                hnorm2 = (cols.abs()**2).sum(-1)                           # (m,)
                # dot products: (bsz,m)
                dot     = (cols.conj() @ y.T).T
                d2      = y_norm2[:, None] + (abs(s)**2) * hnorm2[None] \
                          - 2 * torch.real(s.conj() * dot)
                ll      = -d2 / sigma2                                     # (bsz,m)
                lse     = torch.logaddexp(lse, torch.logsumexp(ll, dim=1))

        ll_true = -((y - sym[:, None] * h_cols).abs()**2).sum(-1) / sigma2
        sum_est += (ll_true - lse).sum().item()

    return (sum_est / n_samples + log_M) / np.log(2)               # bits/use

# ---------- demo / main ------------------------------------------------------
def demo_gpu_lowmem(case: str = 'IMQAM',
                    N_f: int = 256,
                    N_c: int = 128,
                    Mc: int = 4,
                    L: int = 2,
                    snr_db: np.ndarray = np.arange(-10, 31, 4),
                    n_realizations: int = 5,
                    n_samples: int = 2_000,
                    batch: int = 2_000,
                    checkpoint_path: str = "mi_IMQAM_4.pt",
                    use_progress_bar: bool = True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on {device.upper()}, N_f={N_f}, N_c={N_c}, Mc={Mc}, L={L}')

    const = qam_constellation_torch(Mc, device)
    if case == 'IM':
        code = IndexOnly(N_f * N_c, device)
    elif case == 'QAM':
        code = QAMOnly(const)
    else:
        code  = IndexQAM(N_f * N_c, const)
    
    

    R, K = n_realizations, len(snr_db)
    params_now = _make_params_dict(
        N_f=N_f, N_c=N_c, Mc=Mc, L=L,
        snr_db=snr_db, n_realizations=n_realizations,
        n_samples=n_samples, batch=batch
    )

    # ---------- checkpoint ----------
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        if ckpt.get("params", {}) == params_now:
            mi_all = ckpt["mi_all"]
            print("✅  checkpoint loaded, will resume unfinished jobs.")
        else:
            print("⚠️  parameter mismatch, start fresh run.")
            mi_all = torch.full((R, K), float('nan'))
    else:
        mi_all = torch.full((R, K), float('nan'))

    # ---------- progress ----------
    total_steps = R * K
    done_steps  = (~torch.isnan(mi_all)).sum().item()   # 已完成数量
    try:
        from tqdm import tqdm
        pbar = tqdm(total=total_steps, initial=done_steps,
                    disable=not use_progress_bar)
    except ModuleNotFoundError:
        pbar = None
        if use_progress_bar:
            print("(install tqdm for nicer progress bar)")

    # ---------- main loop ----------
    for r in range(R):
        H = LowMemDDChannel(L, N_f, N_c, device)
        for k, snr_d in enumerate(snr_db):
            if not torch.isnan(mi_all[r, k]):           # already computed
                if pbar: pbar.update(1)
                continue

            t0 = time.perf_counter()
            mi = mi_mc_gpu_lowmem(
                    H, code, 10**(snr_d / 10),
                    n_samples=n_samples, batch=batch)
            t1 = time.perf_counter()
            mi_all[r, k] = mi

            # save checkpoint
            torch.save({"mi_all": mi_all, "params": params_now}, checkpoint_path)

            # logging
            msg = (f"[r{r+1}/{R}  SNR={snr_d:>3} dB]  "
                   f"I={mi:.4f} bit/use  "
                   f"({t1 - t0:.2f}s)")
            if pbar:
                pbar.set_description(msg)
                pbar.update(1)
            else:
                print(msg)

    if pbar: pbar.close()

    # ---------- plot if finished ----------
    if torch.isnan(mi_all).any():
        print("⚠️  Incomplete run detected — data saved. Run again to finish.")
        return

    mi_mean = mi_all.mean(0).cpu().numpy()

    # try:
    #     import matplotlib.pyplot as plt
    #     plt.plot(snr_db, mi_mean, 'o-', label=f'avg of {n_realizations} real.')
    #     plt.xlabel('SNR (dB)')
    #     plt.ylabel('I (bit / use)')
    #     plt.title(f'Index-QAM (N={N_f*N_c}, {Mc}-QAM, low-mem)')
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
    #     print("📈  Plot finished.")
    # except ModuleNotFoundError:
    #     print('matplotlib not installed; skipping plot.')
    #     print('MI (bit/use) vs SNR (dB):')
    #     for s, mi in zip(snr_db, mi_mean):
    #         print(f'  {s:>3} : {mi:.4f}')

def demo_gpu_lowmem_IM_only(N_f=512, N_c=128, L=2,
                            snr_db=np.arange(-10, 31, 4),
                            n_realizations=5, n_samples=2_000,
                            batch=2_000, ckpt="mi_IM_4.pt"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N = N_f * N_c
    

    # 其余完全沿用原版 demo，唯一把 `Mc`/星座相关的输出改下
    demo_gpu_lowmem(             # 复用原函数，但换掉参数
        case='IM',  # 只用 Index-only 的代码
        N_f=N_f, N_c=N_c, Mc=1,  # dummy Mc=1 方便写标题
        L=L, snr_db=snr_db,
        n_realizations=n_realizations,
        n_samples=n_samples,
        batch=batch,
        checkpoint_path=ckpt,
        use_progress_bar=True,
        # 传入自定义 codebook 供下层使用
    )

def demo_gpu_lowmem_QAM_only(N_f=512, N_c=128, Mc=4, L=2,
                             snr_db=np.arange(-10, 31, 4),
                             n_realizations=5, n_samples=2_000,
                             batch=2_000, ckpt="mi_QAM_4.pt"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    const  = qam_constellation_torch(Mc, device)

    demo_gpu_lowmem(
        case='QAM',  # 只用 QAM-only 的代码
        N_f=1,          # 只用一列即可，但保留 channel 维度无妨
        N_c=1,
        Mc=Mc,
        L=L,
        snr_db=snr_db,
        n_realizations=n_realizations,
        n_samples=n_samples,
        batch=batch,
        checkpoint_path=ckpt,
        use_progress_bar=True,
        # 同样传入自定义 codebook
    )

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    # 只跑 Index-only
    demo_gpu_lowmem_IM_only(
        N_f=512, N_c=1, L=2,
        snr_db=np.arange(-10, 31, 4),
        n_realizations=5, n_samples=2_000,
        batch=2_000, ckpt="mi_IM_4.pt"
    )

    # 只跑 QAM-only
    demo_gpu_lowmem_QAM_only(
        N_f=512, N_c=1, Mc=4, L=2,
        snr_db=np.arange(-10, 31, 4),
        n_realizations=5, n_samples=2_000,
        batch=2_000, ckpt="mi_QAM_4.pt"
    )

    demo_gpu_lowmem(
        case='IMQAM',        # 复用原版 demo   
        N_f=512,            # fast‑time (delay) dimension
        N_c=1,            # slow‑time (Doppler) dimension
        Mc=4,              # 16‑QAM
        L=2,                # #paths
        snr_db=np.arange(-10, 31, 4),
        n_realizations=5,
        n_samples=2_000,
        checkpoint_path="mi_IMQAM_4.pt"
    )
