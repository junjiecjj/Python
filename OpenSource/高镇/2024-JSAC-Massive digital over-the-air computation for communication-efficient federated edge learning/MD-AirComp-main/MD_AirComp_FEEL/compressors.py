#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import random, math


class CompressorType:
  IDENTICAL                = 1 # Identical compressor
  LAZY_COMPRESSOR          = 2 # Lazy or Bernulli compressor
  RANDK_COMPRESSOR         = 3 # Rank-K compressor
  NATURAL_COMPRESSOR_FP64  = 4 # Natural compressor with FP64
  NATURAL_COMPRESSOR_FP32  = 5 # Natural compressor with FP32
  STANDARD_DITHERING_FP64  = 6 # Standard dithering with FP64
  STANDARD_DITHERING_FP32  = 7 # Standard dithering with FP32
  NATURAL_DITHERING_FP32   = 8 # Natural Dithering applied for FP32 components vectors
  NATURAL_DITHERING_FP64   = 9 # Natural Dithering applied for FP64 components vectors
  TOPK_COMPRESSOR          = 10 # Top-K compressor
  SIGN_COMPRESSOR          = 11 # Sign compressor
  ONEBIT_SIGN_COMPRESSOR   = 12 # One bit sign compressor

class Compressor:
    def __init__(self, compressorName = ""):
        self.compressorName = compressorName
        self.compressorType = CompressorType.IDENTICAL
        self.w = 0.0
        self.last_need_to_send_advance = 0
        self.component_bits_size = 32

    def name(self):
        omega = r'$\omega$'
        if self.compressorType == CompressorType.IDENTICAL: return f"Identical"
        if self.compressorType == CompressorType.LAZY_COMPRESSOR: return f"Bernoulli(Lazy) [p={self.P:g},{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.RANDK_COMPRESSOR: return f" (K={self.K})"
        if self.compressorType == CompressorType.NATURAL_COMPRESSOR_FP64: return f"Natural for fp64 [{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.NATURAL_COMPRESSOR_FP32: return f"Natural for fp32 [{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.STANDARD_DITHERING_FP64: return f"Standard Dithering for fp64[s={self.s}]"
        if self.compressorType == CompressorType.STANDARD_DITHERING_FP64: return f"Standard Dithering for fp32[s={self.s}]"
        if self.compressorType == CompressorType.NATURAL_DITHERING_FP32:  return f"Natural Dithering for fp32[s={self.s},{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.NATURAL_DITHERING_FP64:  return f"Natural Dithering for fp64[s={self.s},{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.TOPK_COMPRESSOR: return f" Top (K={self.K})"
        if self.compressorType == CompressorType.SIGN_COMPRESSOR: return f"Sign"
        if self.compressorType == CompressorType.ONEBIT_SIGN_COMPRESSOR: return f"One Bit Sign"
        return "?"

    def fullName(self):
        omega = r'$\omega$'
        if self.compressorType == CompressorType.IDENTICAL: return f"Identical"
        if self.compressorType == CompressorType.LAZY_COMPRESSOR: return f"Bernoulli(Lazy) [p={self.P:g},{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.RANDK_COMPRESSOR: return f"Rand [K={self.K}]"
        if self.compressorType == CompressorType.NATURAL_COMPRESSOR_FP64: return f"Natural for fp64 [{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.NATURAL_COMPRESSOR_FP32: return f"Natural for fp32 [{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.STANDARD_DITHERING_FP64: return f"Standard Dithering for fp64[s={self.s}]"
        if self.compressorType == CompressorType.STANDARD_DITHERING_FP64: return f"Standard Dithering for fp32[s={self.s}]"
        if self.compressorType == CompressorType.NATURAL_DITHERING_FP32:  return f"Natural Dithering for fp32[s={self.s},{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.NATURAL_DITHERING_FP64:  return f"Natural Dithering for fp64[s={self.s},{omega}={self.getW():.1f}]"
        if self.compressorType == CompressorType.TOPK_COMPRESSOR: return f"Top [K={self.K}]"
        if self.compressorType == CompressorType.SIGN_COMPRESSOR: return f"Sign"
        if self.compressorType == CompressorType.ONEBIT_SIGN_COMPRESSOR: return f"One Bit Sign"
        return "?"

    def resetStats(self):
        self.last_need_to_send_advance = 0

    def makeIdenticalCompressor(self):
        self.compressorType = CompressorType.IDENTICAL
        self.resetStats()

    def makeLazyCompressor(self, P):
        self.compressorType = CompressorType.LAZY_COMPRESSOR
        self.P = P
        self.w = 1.0 / P - 1.0
        self.resetStats()

    def makeStandardDitheringFP64(self, levels, vectorNormCompressor, p = float("inf")):
        self.compressorType = CompressorType.STANDARD_DITHERING_FP64
        self.levelsValues = np.arange(0.0, 1.1, 1.0/levels)     # levels + 1 values in range [0.0, 1.0] which uniformly split this segment
        self.s = len(self.levelsValues) - 1                     # # should be equal to level
        assert self.s == levels

        self.p = p
        self.vectorNormCompressor = vectorNormCompressor
        self.w = 0.0 # TODO

        self.resetStats()

    def makeStandardDitheringFP32(self, levels, vectorNormCompressor, p = float("inf")):
        self.compressorType = CompressorType.STANDARD_DITHERING_FP32
        self.levelsValues = torch.arange(0.0, 1.1, 1.0/levels)     # levels + 1 values in range [0.0, 1.0] which uniformly split this segment
        self.s = len(self.levelsValues) - 1                        # should be equal to level
        assert self.s == levels

        self.p = p
        self.vectorNormCompressor = vectorNormCompressor
        self.w = 0.0 # TODO

        self.resetStats()

    def makeQSGD_FP64(self, levels, dInput):
        norm_compressor = Compressor("norm_compressor")
        norm_compressor.makeIdenticalCompressor()
        self.makeStandardDitheringFP64(levels, norm_compressor, p = 2)
        # Lemma 3.1. from https://arxiv.org/pdf/1610.02132.pdf, page 5
        self.w = min(dInput/(levels*levels), dInput**0.5/levels)

    def makeNaturalDitheringFP64(self, levels, dInput, p = float("inf")):
        self.compressorType = CompressorType.NATURAL_DITHERING_FP64
        self.levelsValues = torch.zeros(levels + 1)
        for i in range(levels):
            self.levelsValues[i] = (1.0/2.0)**i
        self.levelsValues = torch.flip(self.levelsValues, dims = [0])
        self.s = len(self.levelsValues) - 1
        assert self.s == levels

        self.p = p

        r = min(p, 2)
        self.w = 1.0/8.0 + (dInput** (1.0/r)) / (2**(self.s - 1)) * min(1, (dInput**(1.0/r)) / (2**(self.s-1)))
        self.resetStats()

    def makeNaturalDitheringFP32(self, levels, dInput, p = float("inf")):
        self.compressorType = CompressorType.NATURAL_DITHERING_FP32
        self.levelsValues = torch.zeros(levels + 1)
        for i in range(levels):
            self.levelsValues[i] = (1.0/2.0)**i
        self.levelsValues = torch.flip(self.levelsValues, dims=[0])
        self.s = len(self.levelsValues) - 1
        assert self.s == levels

        self.p = p

        r = min(p, 2)
        self.w = 1.0/8.0 + (dInput** (1.0/r)) / (2**(self.s - 1)) * min(1, (dInput**(1.0/r)) / (2**(self.s-1)))
        self.resetStats()

    # K - how much component we leave from input vector
    def makeRandKCompressor(self, K):
        self.compressorType = CompressorType.RANDK_COMPRESSOR
        self.K = K
        self.resetStats()

    def makeTopKCompressor(self, K):
        self.compressorType = CompressorType.TOPK_COMPRESSOR
        self.K = K
        self.resetStats()

    def makeNaturalCompressorFP64(self):
        self.compressorType = CompressorType.NATURAL_COMPRESSOR_FP64
        self.w = 1.0/8.0
        self.resetStats()

    def makeNaturalCompressorFP32(self):
        self.compressorType = CompressorType.NATURAL_COMPRESSOR_FP32
        self.w = 1.0/8.0
        self.resetStats()

    def makeSignCompressor(self, freeze_iteration=0):
        self.compressorType = CompressorType.SIGN_COMPRESSOR
        self.freeze_iteration = freeze_iteration
        self.resetStats()

    def makeOneBitSignCompressor(self, freeze_iteration=0):
        self.compressorType = CompressorType.ONEBIT_SIGN_COMPRESSOR
        self.freeze_iteration = freeze_iteration
        self.resetStats()

    def getW(self):
        return self.w

    def compressVector(self, x, iteration=0):
        d = max(x.shape)

        if self.compressorType == CompressorType.IDENTICAL:
            out = x.clone()
            self.last_need_to_send_advance = d * self.component_bits_size


        elif self.compressorType == CompressorType.TOPK_COMPRESSOR:
            #S = torch.arange(d)
            # np.random.shuffle(S)
            top_size = max(int(self.K*d), 1)
            _, S = torch.topk(torch.abs(x), top_size)
            out = torch.zeros_like(x)
            out[S] = x[S]
            # !!! in real case, one needs to send the out vector and a support set to indicate the indices of top K 
            self.last_need_to_send_advance = 2 * top_size * self.component_bits_size

        elif self.compressorType == CompressorType.SIGN_COMPRESSOR:
            if iteration < self.freeze_iteration:
                out = x.clone()
                self.last_need_to_send_advance = d * self.component_bits_size
            else:
                
                out = torch.sign(x)
                
                scale = torch.norm(x, p=1) / torch.numel(x)
                
                out.mul_(scale) # <-- we use this just for similation
                
                
                # !!! in real case, one needs to send D bits for {0, 1} and 32 bits for the scale constant
                self.last_need_to_send_advance = d + self.component_bits_size

        elif self.compressorType == CompressorType.ONEBIT_SIGN_COMPRESSOR:
            # according to one bit adam paper,
            # during warmup, the signal is not compressed
            if iteration < self.freeze_iteration:
                out = x.clone()
                self.last_need_to_send_advance = d * self.component_bits_size
            else:
                out = torch.sign(x)
                # out.add_(1).bool().float().add_(-0.5).mul_(2.0)
                scale = torch.norm(x) / np.sqrt(torch.numel(x))
                # out = torch.cat((scale, out), 0) <-- in real case, only send a scale, and a {0,1}^D output
                # this is just for similate
                out.mul_(scale) # <-- we use this just for similation
                # !!! in real case, one needs to send D bits for {0, 1} and 32 bits for the scale constant
                self.last_need_to_send_advance = d + self.component_bits_size

        return out
