import numpy as np
import os
import argparse
import zipfile as zp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from datetime import datetime
import json
import scipy
from scipy.io import savemat
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import cv2
import time

np.seterr(divide='ignore', invalid='ignore')

class ImageGrid:
  def __init__(self, cellrows, cellcols, cellsize, isuint8, bitUp, bitDown, backgroundIsRed=True):
    self.cellrows   = cellrows
    self.cellcols   = cellcols
    self.cellsize   = cellsize
    self.isuint8    = isuint8
    self.bitUp      = bitUp
    self.bitDown    = bitDown
    self.topval     = 255 if self.isuint8 else 1
    dtype           = np.uint8 if isuint8 else np.float32
    self.img        = np.zeros((cellrows*cellsize[0], cellcols*cellsize[1], 3), dtype=dtype)
    if backgroundIsRed:
      self.img[:,:,2] = self.topval

  def putCell(self, celli, cellj, values):
    celli = int(celli*self.cellsize[0])
    cellj = int(cellj*self.cellsize[1])
    self.img[celli:celli+values.shape[0],cellj:cellj+values.shape[1],:] = values
  
  def putBinaryCell(self, celli, cellj, cellOnes):
    celli = int(celli*self.cellsize[0])
    cellj = int(cellj*self.cellsize[1])
    areOneIdxs      = (cellOnes==self.bitUp  ).nonzero()
    areZeroIdxs     = (cellOnes==self.bitDown).nonzero()
    #import code; code.interact(local=vars())
    if len(areOneIdxs[0])>0:
      areOneIdxs    = [ areOneIdxs[0]+celli,  areOneIdxs[1]+cellj]
    if len(areZeroIdxs[0])>0:
      areZeroIdxs   = [areZeroIdxs[0]+celli, areZeroIdxs[1]+cellj]
    self.img[areZeroIdxs[0],areZeroIdxs[1],2] = 0
    self.img[ areOneIdxs[0], areOneIdxs[1],0] = self.topval
    self.img[ areOneIdxs[0], areOneIdxs[1],1] = self.topval

  def dealBinaryBitsToAllCells(self, biti, bitj, bitmask):
    for a0 in range(bitmask.shape[0]):
      for a1 in range(bitmask.shape[1]):
        i = a0*self.cellsize[0]+biti
        j = a1*self.cellsize[1]+bitj
        if not (len(i)==1 and len(j)==1):
          raise Exception(f'fucked this up!!!')
        i = int(i)
        j = int(j)
        if bitmask[a0,a1]==self.bitDown:
          self.img[i,j,2] = 0
        else:
          self.img[i,j,0] = self.topval
          self.img[i,j,1] = self.topval

  def dealPositiveAndNegativeValuesToAllCells(self, matrix_p, matrix_n, i_incell, j_incell):
    for a0 in range(self.cellrows):
      for a1 in range(self.cellcols):
        bit0 = a0*self.cellsize[0]+i_incell
        bit1 = a1*self.cellsize[1]+j_incell
        self.img[bit0,bit1,0]=matrix_p[a0,a1]
        self.img[bit0,bit1,1]=matrix_n[a0,a1]
        self.img[bit0,bit1,2]=0

  def dealPositiveAndNegativeValuesToSingleCell(self, matrix_p, matrix_n, celli, cellj):
    celli = int(celli*self.cellsize[0])
    cellj = int(cellj*self.cellsize[1])
    si = slice(celli, celli+matrix_p.shape[0])
    sj = slice(cellj, cellj+matrix_p.shape[1])
    self.img[si, sj, 0] = matrix_p
    self.img[si, sj, 1] = matrix_n
    self.img[si, sj, 2] = 0

  def normalizePerChannelAndCell(self, touint8=True):
    for a0 in range(self.cellrows):
      for a1 in range(self.cellcols):
        s1 = slice(a0*self.cellsize[0], (a0+1)*self.cellsize[0])
        s2 = slice(a0*self.cellsize[1], (a0+1)*self.cellsize[1])
        self.img[s1,s2,0] /= self.img[s1,s2,0].max()
        self.img[s1,s2,1] /= self.img[s1,s2,1].max()
    self.img[:,:,2]       /= self.img[:,:,2].max()
    if touint8:
      self.img             = (self.img*255).astype(np.uint8)


class OneBeamGridDataset(data.Dataset):
  def __init__(self, filename, sz=10, device=None, in_degrees=True, to_XY=True, also_dct=True, normalize_by_dct_mean=True, filter_angles=None, classicRepresentation1bit=True, additionalOuput=None):
    self.filename  = filename
    self.names    = []
    self.rows     = sz
    self.cols     = sz
    self.also_dct = also_dct
    self.bitUp    = 1
    self.bitDown  = 0#-1 if also_dct else 0
    numelems      = self.rows * self.cols
    use_filter_angles = filter_angles is not None
    if use_filter_angles:
      #this is only for the old codepath
      phi_min   = filter_angles[0][0]
      phi_max   = filter_angles[0][1]
      theta_min = filter_angles[1][0]
      theta_max = filter_angles[1][1]
    self.classicRepresentation1bit = classicRepresentation1bit
    self.normalize_by_dct_mean = normalize_by_dct_mean
    self.additionalOuputMode = 0
    if filename.lower().endswith('.mat'):
      mat = scipy.io.loadmat(filename)
      self.nbeams = mat['DATASET']['beams'][0,0][0,0]
      self.nbits  = mat['DATASET']['bits'] [0,0][0,0]
      self.grids            = mat['DATASET']['labels'][0,0].transpose(2,0,1)
      if self.nbits==1 and self.classicRepresentation1bit:
        self.grids          = self.grids.astype(np.int8)
        self.gridAsAngles   = False
        self.angleValues    = np.array([0,  1], dtype=np.float32)
        self.balancedValues = np.array([-1, 1], dtype=np.float32)
      else:
        self.grids          = self.grids.astype(np.float32)#*np.pi
        self.gridAsAngles   = True
        numstates           = 2**self.nbits
        self.angleValues    = np.linspace(0,  2, numstates, endpoint=False, dtype=np.float32) # values as they should appear in the tensor imported from matlab
        self.balancedValues = np.linspace(-1, 1, numstates,                 dtype=np.float32) # values for dct balacing
      self.beam_params      = mat['DATASET']['input'][0,0].astype(np.int32)#.transpose(1,0)
      self.realidxs         = np.arange(self.beam_params.shape[0], dtype=np.int32)
      if self.grids.shape[-1]!=sz or self.grids.shape[-2]!=sz:
        raise Exception(f'grids have shape {self.grids.shape}, but pattern size is {sz}!!!!!')
      if self.beam_params.shape[1]!=2*self.nbeams:
        raise Exception(f'Number of beams is {self.nbeams}, this should mean that there are {2*self.nbeams} parameters per tablero, but parameter shape is {self.beam_params.shape}')
      # filter out dataset elements
      if use_filter_angles:
        retain            = np.ones(self.beam_params.shape[0], dtype=bool)
        for k in range(self.beam_params.shape[0]):
          params = self.beam_params[k]
          for nang in range(len(params)):
            if params[nang]<filter_angles[nang][0] or params[nang]>filter_angles[nang][1]:
              retain[k] = False 
              break
        if not np.all(retain):
          self.beam_params = self.beam_params[retain,:]
          self.grids       = self.grids[retain,:,:]
          self.realidxs    = self.realidxs[retain]
    else:
      raise Exception(f'Filename with unknown extensions: {filename}')
    if self.gridAsAngles:
      #angles in the range 0-2*pi are codified in radians/pi, so they are in the range 0-2
      self.scalarMapCol  = ScalarMappable(norm=Normalize(vmin=0.0, vmax=2.0), cmap='hsv') # use cyclic colormap!
      self.scalarMapGray = ScalarMappable(norm=Normalize(vmin=0.0, vmax=self.angleValues.max()), cmap='gray')
      #convert grids to symbolic values in range(numstates) to avoid annoying floating point errors from biting us in the ass
      grids_symbolic     = np.full(self.grids.shape, -1, np.int8)
      cutoff = 1e-4
      for i,v in enumerate(self.angleValues):
        mask = np.abs(self.grids-v)<cutoff
        if (grids_symbolic[mask]!=-1).any():
          raise Exception('there is something wrong with the grids!!!!')
        grids_symbolic[mask] = i
      if (grids_symbolic<0).any() or (grids_symbolic>=numstates).any():
        raise Exception(f'there are non-valid values in the grid!!! Valid values are {values}')
      #compute all equivalent variants
      all_grids          = []
      all_symbolic_grids = []
      all_balanced_grids = []
      for k in range(numstates):
        #compute current sym grids
        current_sym = (grids_symbolic+k)%numstates
        #compute current/balanced grids
        current_grids = np.zeros(self.grids.shape, dtype=np.float32)
        current_balanced_grids = np.zeros(self.grids.shape, dtype=np.float32) if also_dct else None
        for i,(v,b) in enumerate(zip(self.angleValues, self.balancedValues)):
          mask = current_sym==i
          current_grids[mask] = v
          if also_dct:
            current_balanced_grids[mask] = b
        all_grids.append(current_grids)
        all_symbolic_grids.append(current_sym)
        if also_dct:
          all_balanced_grids.append(current_balanced_grids)
      #self.original_grids = self.grids
      self.grids          = np.stack(all_grids,          axis=1)
      self.symbolic_grids = np.stack(all_symbolic_grids, axis=1).astype(np.int64)
      if also_dct:
        balanced_grids = np.stack(all_balanced_grids, axis=1)
     #create DCTs, and normalize if required
    if also_dct:
      self.dcts    = np.zeros(self.grids.shape, dtype=np.float32)
      import scipy.fftpack as fftpack
      if self.gridAsAngles:
        for k in range(self.grids.shape[0]):
          for i in range(self.grids.shape[1]):
            self.dcts[k,i,:,:]    = fftpack.dctn(balanced_grids[k,i,:,:], norm='ortho')
        if normalize_by_dct_mean:
          means = self.dcts.reshape(*self.dcts.shape[:2], -1)
          means = means.mean(axis=-1)
          means = np.argmin(means, axis=-1)
          if len(means.shape)!=1 or means.shape[0]!=self.grids.shape[0]:
            raise Exception('strange error computing canonical indexes!!!!')
          for i,s in enumerate(means):
            #if not already there, put the canonical instance at position 0 in all tensors replicated for numstates
            if s!=0:
              canonical_grid           = self.grids[i,s].copy()
              self.grids[i,s]          = self.grids[i,0]
              self.grids[i,0]          = canonical_grid
              canonical_dct            = self.dcts[i,s].copy()
              self.dcts [i,s]          = self.dcts[i,0]
              self.dcts [i,0]          = canonical_dct
              canonical_sym            = self.symbolic_grids[i,s].copy()
              self.symbolic_grids[i,s] = self.symbolic_grids[i,0]
              self.symbolic_grids[i,0] = canonical_sym
      else:
        balanced_grids = self.grids.copy()
        balanced_grids[balanced_grids==0] = -1
        balanced_grids = balanced_grids.astype(np.float32)
        for k in range(self.beam_params.shape[0]):
          self.dcts[k,:,:]    = fftpack.dctn(balanced_grids[k,:,:], norm='ortho')
          if normalize_by_dct_mean and np.mean(self.dcts[k,:,:])<0:
            self. dcts[k,:,:] = -self. dcts[k,:,:]
            self.grids[k,:,:] = 1-self.grids[k,:,:]
      del balanced_grids
    self.grids         = self.grids.astype(np.float32)
    self.phi_theta     = self.beam_params
    self.beam_params   = np.array(self.beam_params, dtype=np.float32)
    if in_degrees:
      self.beam_params*= np.pi/180
      self.phi_theta_radians = self.beam_params.astype(np.float32)
    if to_XY:
      beam_params = np.zeros((self.beam_params.shape[0],self.beam_params.shape[1]*2), dtype=np.float32)
      for nang in range(self.beam_params.shape[1]):
        beam_params[:,nang*2  ] = np.sin(self.beam_params[:,nang])
        beam_params[:,nang*2+1] = np.cos(self.beam_params[:,nang])
      self.beam_params = beam_params
    self.names = None
    self.device = device
    if device is not None:
      self.phi_theta_radians= torch.from_numpy(self.phi_theta_radians).to(device)
      self.beam_params      = torch.from_numpy(self.beam_params)   .to(device)
      self.grids            = torch.from_numpy(self.grids)         .to(device)
      if also_dct:
        self.dcts           = torch.from_numpy(self.dcts)          .to(device)
      if self.gridAsAngles:
        self.symbolic_grids = torch.from_numpy(self.symbolic_grids).to(device)
    self.orig_phi_theta     = self.phi_theta
    self.orig_phi_theta_rad = self.phi_theta_radians
    self.orig_beam_params   = self.beam_params

  def colorizeAngles(self, angles, color=True):
    sm = self.scalarMapCol if color else self.scalarMapGray
    return sm.to_rgba(angles,bytes=True)[...,:3]

  def setAdditionalOutput(self, additionalOuput):
    if     additionalOuput is None:
      self.additionalOuputMode = 0
    elif   additionalOuput=='dct':
      if not self.also_dct:
        raise Exception('If also_dct is False, you cannot put DCT as additional output!!!')
      self.additionalOuputMode = 1
    elif   additionalOuput=='symbolic':
      if not self.gridAsAngles:
        raise Exception('If grids are not angles, you cannot put SYMBOLIC as additional output!!!')
      self.additionalOuputMode = 2
    else:
      raise Exception(f'Unexpected value for additionalOuput: {additionalOuput}')

  def __getitem__(self, index):
    phi_theta = self.phi_theta[index,:]
    beam_params  = self.beam_params[index,:]
    idx = index#self.realidxs[index]
    if   self.additionalOuputMode==0:
      additional = np.nan
    elif self.additionalOuputMode==1:
      additional = self.dcts[index]
    elif self.additionalOuputMode==2:
      additional = self.symbolic_grids[index]
    return phi_theta, beam_params, self.grids[index], idx, additional
      
  def __len__(self):
    return self.beam_params.shape[0]

def load_output_dct_transform(filename):
  with np.load(filename) as data:
    mins  = data['mins']
    maxs  = data['maxs']
    means = data['means']
    stds  = data['mins']
  #import code; code.interact(local=vars())
  min_to_mean        = np.abs(means-mins)
  max_to_mean        = np.abs(means-maxs)
  max_extent_to_mean = np.maximum(min_to_mean, max_to_mean)*1.05
  A                  = max_extent_to_mean
  A                  = torch.from_numpy(A)
  return RangedRegressionUnbiasedTransform(A)

class RangedRegressionUnbiasedTransform(nn.Module):
  def __init__(self, A):
    super(RangedRegressionUnbiasedTransform, self).__init__()
    A = torch.unsqueeze(A, 0)
    self.tanh1 = (np.exp(2)-1)/(np.exp(2)+1)
    A = A/self.tanh1
    self.register_buffer('A', A)
  def forward(self, x):
    x = torch.tanh(x)
    x = x*self.A
    return x

def compute_2d_basis_functions(sz, func):
  basis_functions = np.zeros( (sz,sz,sz,sz), dtype=np.float32)
  for i in range(sz):
    for j in range(sz):
      transformed              = np.zeros((sz,sz), dtype=np.float32)
      transformed[i,j]         = 1.0
      basis_functions[i,j,:,:] = func(transformed)
  return basis_functions

def apply_basis_functions(original, basis_functions, zerofun=np.zeros):
  transformed = zerofun( original.shape, dtype=original.dtype)
  #singletons_1 = [1 for d in original.shape[:-2]]
  #singletons_2 = [1 for d in original.shape[-2:]]
  #single_basis_function_shape = (*singletons_1, *basis_functions.shape[-2:])
  #single_coefficient_shape    = (*original.shape[:-2], *singletons_2)
  numdims = len(original.shape)
  if numdims==3:
    single_basis_function_shape = (1, *basis_functions.shape[-2:])
    single_coefficient_shape    = (original.shape[0], 1, 1)
    for i in range(original.shape[-2]):
      for j in range(original.shape[-1]):
        # the indexing is not straightforward in order to make this code batch-ready
        transformed += original[..., i,j].reshape(single_coefficient_shape) * basis_functions[i,j,:,:].reshape(single_basis_function_shape)
  elif numdims==2:
    for i in range(original.shape[-2]):
      for j in range(original.shape[-1]):
        # the indexing is not straightforward in order to make this code batch-ready
        transformed += original[i,j] * basis_functions[i,j,:,:]
  return transformed

# This goes as the second argument for Artisanal2DTransform
def original_idct(x):
  import scipy.fftpack as fftpack
  return fftpack.idctn(x, norm='ortho')

class Artisanal2DTransform(nn.Module):

  def __init__(self, sz, func, device=None, also_torch=True):
    super().__init__()
    self.sz                  = sz
    self.device              = device
    self.raw_basis_functions = compute_2d_basis_functions(sz, func)
    self.also_torch          = also_torch
    if also_torch:
      basis_functions        = torch.from_numpy(self.raw_basis_functions).to(device)
      self.register_buffer('basis_functions', basis_functions)

  def compute_np(self, x):
    return apply_basis_functions(x, self.raw_basis_functions, np.zeros)

  def forward(self, x):
    return apply_basis_functions(x, self.basis_functions, lambda *args, **kwargs: torch.zeros(*args, device=self.device, **kwargs))

def compute_multibit_from_raw_angles(raw_angles, comparable_platonic_values, output_platonic_values):
  # compute differences to canonical angle values
  ones = [1]*len(raw_angles.shape)
  out  = torch.abs(raw_angles.reshape(1, *raw_angles.shape)-comparable_platonic_values.reshape(-1,*ones))
  # compute closest canonical angle in each case
  out  = out.argmin(axis=0)
  # fill results with canonical angles (instead of the corresponding raw angles)
  out  = torch.take(output_platonic_values, out)
  return out


class BeamGenerator(nn.Module):
    def __init__(self, sz=10, input_dims=4, latent_space_dims=30, channel_multipler=100, num_output_channels=1, regressionTransformStatsFile=None, also_idct=False, device=None, **kwargs):
        super(BeamGenerator, self).__init__()
        self.selector            = lambda phi_thetas, beam_params: beam_params
        self.do_regression       = regressionTransformStatsFile is not None
        self.do_idct             = also_idct
        self.latent_space_dims   = latent_space_dims
        self.dens_0 = nn.Linear( input_dims, latent_space_dims, bias=False)
        self.norm_0 = nn.BatchNorm1d(latent_space_dims)
        self.relu_0 = nn.SiLU()
        self.conv_1 = nn.ConvTranspose2d( latent_space_dims, channel_multipler * 8, 4, 1, 0, 0, bias=False)
        self.norm_1 = nn.BatchNorm2d(channel_multipler * 8)
        self.relu_1 = nn.SiLU()
        self.conv_2 = nn.ConvTranspose2d(channel_multipler * 8, channel_multipler * 4, 4, 1, 0, 0, bias=False)
        self.norm_2 = nn.BatchNorm2d(channel_multipler * 4)
        self.relu_2 = nn.SiLU()
        self.conv_3 = nn.ConvTranspose2d( channel_multipler * 4, channel_multipler * 2, 4, 1, 0, 0, bias=False)
        self.norm_3 = nn.BatchNorm2d(channel_multipler * 2)
        self.relu_3 = nn.SiLU()
        if sz==10:
            self.conv_4 = nn.Conv2d( channel_multipler * 2, channel_multipler, 3, 1, 1, bias=False) 
            self.norm_4 = nn.BatchNorm2d(channel_multipler)
            self.relu_4 = nn.SiLU()
            self.conv_5 = nn.Conv2d( channel_multipler, num_output_channels, 3, 1, 1, bias=False) 
        elif sz==15:
            self.conv_4 = nn.ConvTranspose2d( channel_multipler * 2, channel_multipler, 3, 1, 0, 0, bias=False)
            self.norm_4 = nn.BatchNorm2d(channel_multipler)
            self.relu_4 = nn.SiLU()
            self.conv_5 = nn.ConvTranspose2d( channel_multipler, num_output_channels, 4, 1, 0, 0, bias=False)
        elif sz==20:
            self.conv_4 = nn.ConvTranspose2d( channel_multipler * 2, channel_multipler, 4, 2, 0, 0, bias=False)
            self.norm_4 = nn.BatchNorm2d(channel_multipler)
            self.relu_4 = nn.SiLU()
            self.conv_5 = nn.Conv2d( channel_multipler, num_output_channels, 3, 1, 0, bias=False) 
        else:
            raise Exception(f'Size not expected: {sz}')
        if self.do_regression:
          self.rangedRegression = load_output_dct_transform(regressionTransformStatsFile)
          if self.do_idct:
            self.device = device
            self.idct = Artisanal2DTransform(sz, original_idct, device=device, also_torch=True)

    def forward(self, input):
        out = input
        out = self.dens_0(out)
        out = self.norm_0(out)
        out = self.relu_0(out)
        out = torch.reshape(out, (*out.shape, 1, 1))
        out = self.conv_1(out)
        out = self.norm_1(out)
        out = self.relu_1(out)
        out = self.conv_2(out)
        out = self.norm_2(out)
        out = self.relu_2(out)
        out = self.conv_3(out)
        out = self.norm_3(out)
        out = self.relu_3(out)
        out = self.conv_4(out)
        out = self.norm_4(out)
        out = self.relu_4(out)
        out = self.conv_5(out)
        out = torch.squeeze(out)
        if self.do_regression:
          out = self.rangedRegression(out)
          if self.do_idct:
            out = self.idct(out)
            if self.dataset.nbit==1:
              out = out>0
            else:
              out = compute_multibit_from_raw_angles(out, self.balancedValues, self.angleValues)
        return out

class MultipleGenerators(nn.Module): 
  def __init__(self, input_list, rangeSelector=None):
    super(MultipleGenerators, self).__init__()
    self.selector      = lambda phi_thetas, beam_params: (phi_thetas, beam_params)
    self.generators    = nn.ModuleList(input_list)
    self.setRangeSelector(rangeSelector)

  def setRangeSelector(self, rangeSelector):
    if rangeSelector is not None:
      if rangeSelector.getNumRanges()!=len(self.generators):
        raise Exception(f'The number of generators is {len(self.generators)}, but it is not the same as the rangeSelector number of ranges {rangeSelector.getNumRanges()}')
      self.rangeSelector = rangeSelector

  def forward(self, args):
    phi_thetas, beam_params = args
    selected = self.rangeSelector.selectSingleRangeForAllAngleTuples(phi_thetas)
    return self.generators[selected](beam_params)

class AnglesRangesSelector:
  """ class accepting a range definition (num_rangesxtuple_sizex2) and providing methods to map angle tuples to ranges """
  def __init__(self, ranges):
    """ ranges has shape num_tuplesxtuple_sizex2 """
    if len(ranges.shape)!=3:
      raise Exception(f'ranges must have shape num_tuplesxtuple_sizex2, but it has shape {ranges.shape}!!')
    if ranges.shape[0]==0:
      raise Exception(f'first dimension of ranges (num_tuples) must not be 0!')
    if ranges.shape[1]==0:
      raise Exception(f'second dimension of ranges (tuple_size) must not be 0!')
    if ranges.shape[2]!=2:
      raise Exception(f'third dimension of ranges must be 2 (min, max), but it is {ranges.shape[2]}!')
    self.ranges = ranges

  def getNumRanges(self):
    return self.ranges.shape[0]

  def getTupleSize(self):
    return self.ranges.shape[1]

  def selectSingleRangeForAllAngleTuples(self, angles, stopOnError=True):
    """ angles has shape num_batchesxtuple_size.
    Returns None if there was no range selected for all tuples """
    if len(angles.shape)!=2:
      handleErrorMessage(f'angles shape must have 2 dimensions, but it has {len(angles.shape)}', stopOnError)
    if angles.shape[-1]!=self.ranges.shape[1]:
      handleErrorMessage(f'last dimension for angles must be tuple size {self.ranges.shape[1]}, but it is {angles.shape[-1]}', stopOnError)
    if isinstance(angles, torch.Tensor):
      angles = angles.detach().cpu().numpy()
    selected = None
    for idx, (spans_for_this_range) in enumerate(self.ranges):
      selected_for_this = None
      for angle_idx, (angle_min, angle_max) in enumerate(spans_for_this_range):
        if angle_min==-np.inf and angle_max==np.inf:
          continue
        this_angle = angles[:,angle_idx]
        this_selected = np.logical_and(this_angle>=angle_min, this_angle<=angle_max)
        selected_for_this = this_selected if selected_for_this is None else np.logical_and(selected_for_this, this_selected)
      count = selected_for_this.sum()
      if count==selected_for_this.size:
        if selected is None:
          selected = idx
        else:
          handleErrorMessage(f'There can be only one selected generator, but more than one was selected! First {selected}, now also {idx}', stopOnError)
      elif count>0:
        #import code; code.interact(local=vars())
        handleErrorMessage(f'All angle {angles.shape[-1]}-tuples must be assigned to a specific range, but for range {idx} only some are: {angles}', stopOnError)
    if selected is None:
      handleErrorMessage(f'No range was selected for this input {angles}', stopOnError)
    return selected


class ActivationHandler:
  """ This class stores the setup to digest network activations into grids (validation) and loss values (training). The setup is different for each outputType. """
  def __init__(self, outputType, patternSize, dataset, device, useoldmax_straightAngles):
    self.gridAsAngles                 = dataset.gridAsAngles
    if dataset.angleValues is not None:
      angleValues                     = torch.from_numpy(dataset.angleValues)
      self.angleValues                = angleValues.to(device)
      self.wrappedAngleValues         = torch.cat((angleValues, torch.zeros((1,), dtype=torch.float32)+2.0)).to(device)
      self.wrappedPlatonicAngleValues = torch.cat((angleValues, torch.zeros((1,), dtype=torch.float32))).to(device)
      self.max_straightAngles         = 2 if useoldmax_straightAngles else angleValues.max()
    else:
      self.angleValues                = None
      self.wrappedAngleValues         = None
      self.wrappedPlatonicAngleValues = None
    self.balancedValues               = torch.from_numpy(dataset.balancedValues).to(device) if dataset.balancedValues is not None else None
    #self.dataset = dataset

    allgrids                 = dataset.grids.detach().cpu().numpy()
    if len(allgrids.shape)==4:
      allgrids               = allgrids[:,0,:,:]

    if self.gridAsAngles:
      if   outputType=='straightAngles':
        self.idct                = None
        self.input_ok            = lambda x: len(x.shape)==3
        self.computefun          = self._compute_multibit_angles
        dataset.setAdditionalOutput(None)
      elif outputType=='straightDCT':
        self.idct                = get_custom_idct(patternSize, device)
        self.input_ok            = lambda x: len(x.shape)==3
        self.computefun          = self._compute_multibit_dct
        dataset.setAdditionalOutput('dct')
      else:
        raise Exception(f'outputType not understood for multibit: {outputType}')
    else:
      if   outputType=='straightBinary':
        self.idct                = None
        self.negate              = lambda x: 1-x
        self.computefun          = self._compute_classic_with_sigmoid
        dataset.setAdditionalOutput(None)
      elif   outputType=='straightBinaryMSE':
        self.idct                = None
        self.negate              = lambda x: 1-x
        self.computefun          = self._compute_classic_with_sigmoid
        dataset.setAdditionalOutput(None)
      elif outputType=='straightDCT':
        self.idct                = get_custom_idct(patternSize, device)
        self.negate              = lambda x:  -x
        self.computefun          = self._compute_classic_with_dct
        dataset.setAdditionalOutput('dct')
      else:
        raise Exception(f'outputType not understood: {outputType}')

  def _compute_multibit_angles(self, activations):
    # compute raw angles
    activations  = activations.sigmoid()
    out = activations*self.max_straightAngles
    return compute_multibit_from_raw_angles(out, self.angleValues, self.angleValues)

  def _compute_multibit_dct(self, activations):
    out = self.idct(activations)
    return compute_multibit_from_raw_angles(out, self.balancedValues, self.angleValues)

  def _compute_classic_with_sigmoid(self, activations):
    return activations.sigmoid().round().type(torch.int8)

  def _compute_classic_with_dct(self, activations):
    return self.idct(activations)>0

  def compute_grid(self, activations):
    grids = self.computefun(activations)
    grids = grids.detach().cpu().numpy()
    return grids


def save_result_as_mat(filename, phi_theta, sines_cosines, grids, preds):
  savedict = {'phi_theta': phi_theta, 'sines_cosines': sines_cosines, 'grids': grids, 'predictions': preds}
  savemat(filename, savedict, appendmat=False)

def do_validate(device, dataset, model, filename, imgbase, matfilename, outputType='straightBinary', alsoNegated=True, patternSize=10, incidence=0, cellstep=None, useoldmax_straightAngles=False):
  act_handler  = ActivationHandler(outputType, patternSize, dataset, device, useoldmax_straightAngles)
  model.train(False)
  #import code; code.interact(local=vars())
  if len(dataset.grids.shape)==4:
    lens = (dataset.grids.shape[0], dataset.grids.shape[2], dataset.grids.shape[3])
  else:
    lens = dataset.grids.shape
  preds = np.zeros(lens, dtype=np.float32)
  with open(filename, 'w') as FILE:
    validation = Validation(True, FILE, dataset.bitDown, act_handler, dataset, alsoNegated=alsoNegated, num_instances=dataset.phi_theta.shape[0], device=device, compute_CM=True, incidence=incidence, cellstep=cellstep)
    for k in range(len(dataset)):
      if k % 1000 == 0:
        print(f'Validating pattern {k}/{len(dataset)}')
      beam_params   = torch.unsqueeze(dataset.beam_params[k], 0)
      grid          = torch.unsqueeze(dataset.grids[k], 0)
      phi_theta     = np.expand_dims(dataset.phi_theta[k], axis=0)
      realidx       = (dataset.realidxs[k],)
      input_args    = model.selector(phi_theta, beam_params)
      prediction    = model(input_args)
      if len(prediction.shape)<3:
        prediction  = torch.unsqueeze(prediction, 0)
      #import code; code.interact(local=vars())
      pred          = validation.collect_statistics(prediction, grid, phi_theta, beam_params, realidx)
      preds[k]      = pred#[0]
    summary = validation.summary_str()
    FILE.write(f'\n\n{summary}')
    #print(f'DO_VALIDATE SUMMARY: \n{summary}')
  grids = dataset.grids.detach().cpu().numpy()#[:,0,:,:]
  if imgbase is not None:
    print('Writing images')
  makeValidationImages(imgbase, dataset.phi_theta, dataset, grids, preds)
  save_result_as_mat(matfilename, dataset.phi_theta, dataset.beam_params.detach().cpu().numpy(), grids, preds)
  makeGroundTruthImages(f'{imgbase}gt_', dataset.phi_theta, dataset, dataset.grids.detach().cpu().numpy(), dcts=dataset.dcts.detach().cpu().numpy() if dataset.also_dct else None)#; sys.exit()    

class Validation:

  def __init__(self, saveAllResults, FILE, bitDown, act_handler, dataset, alsoNegated=False, num_instances=None, device=None, compute_CM=False, incidence=0, cellstep=None):
    self.alsoRadError   = True
    if self.alsoRadError:
      self.radErrorComp = ZeroIncidenceRadiationError(sz=dataset.rows, incidence=incidence, cellstep=cellstep)
    self.saveAllResults = saveAllResults
    self.FILE           = FILE
    self.bitDown        = bitDown
    self.act_handler    = act_handler
    self.gridAsAngles   = dataset.gridAsAngles
    self.angleValues    = dataset.angleValues
    # This name is unfortunate (negation is a gimmick specific to 1bit boards), but we keep it for multibit, where it means to take into account all equivalent boards
    self.alsoNegated    = alsoNegated
    self.num_instances  = num_instances
    self.by_instance    = num_instances is not None
    self.cutoff         = 1e-2
    self.compute_CM     = compute_CM
    if compute_CM:
      self.cm_template    = np.zeros((self.angleValues.size, self.angleValues.size), dtype=np.int32)
      self.confmatrixgen  = np.zeros((self.angleValues.size, self.angleValues.size), dtype=np.int32)
    self.reset()

  def reset(self):
    self.count_bad     = 0
    self.count_nums    = 0
    self.count_grids   = 0
    self.count         = 0
    self.count_n       = 0
    self.count_errordB = 0
    if self.by_instance:
      self.idx         = 0
      self.bads        = np.zeros((self.num_instances,), dtype=np.float64)
      if self.compute_CM:
        self.cms       = np.zeros((self.num_instances, self.angleValues.size, self.angleValues.size), dtype=np.int32)
  
  def collect_statistics(self, predictions, grids, phi_thetas, beam_params, realidxs):
    if self.gridAsAngles:
      assert len(grids.shape)==4
    else:
      assert len(grids.shape)==3
    #import code; code.interact(local=vars())
    #print(f'Mira predictions 0: {predictions.shape}')
    predictions = self.act_handler.compute_grid(predictions)
    #print(f'Mira predictions 1: {predictions.shape}')
    if len(predictions.shape)==3:
        assert(predictions.shape[0]==1)
        predictions = predictions[0]
        #print(f'Mira predictions 2: {predictions.shape}')
    grids   = grids.detach()
    if not self.gridAsAngles:
      grids = grids.type(torch.int8)
    grids   = grids.cpu().numpy()
    if True:#for k in range(predictions.shape[0]):
      k=0
      if self.gridAsAngles:
        grid     = grids[k,0]
      else:
        grid     = grids[k]
      #prediction = predictions[k]#[k,:,:]
      prediction = predictions
      if self.alsoRadError:
        count_n, raderror = self.radErrorComp.compute_raderror(prediction, grid)
        self.count         += 1
        self.count_n       += float(count_n)
        self.count_errordB += float(raderror)
      realidx    = realidxs[k]
      #print(f'Mira prediction/grid 3: {prediction.shape} {grid.shape}')
      #import code; code.interact(local=vars())
      if prediction.dtype in (np.int8, bool):
        bads     = grid!=prediction
      else:
        bads     = np.abs(grid-prediction)>self.cutoff
      num_bad    = int(bads.sum())
      num_vals   = int(prediction.size)
      best_grid  = grid
      if self.alsoNegated:
        if self.gridAsAngles:
          for i in range(1, grids.shape[1]):
            grid      = grids[k,i]
            #print(f'Mira prediction/grid 3: {prediction.shape} {grid.shape}')
            badsb     = np.abs(grid-prediction)>self.cutoff
            num_badb  = int(badsb.sum())
            if num_badb < num_bad:
              bads    = badsb
              num_bad = num_badb
              best_grid = grid
        else:
          badsb       = np.logical_not(bads)
          num_badb    = int(badsb.sum())
          if num_badb < num_bad:
            bads      = badsb
            num_bad   = num_badb
            best_grid = np.logical_not(grid)
      if self.compute_CM:
        self.cm_template[:,:] = 0
        get_confusion_matrix(self.angleValues, self.cutoff, self.cm_template, best_grid, prediction)
        self.confmatrixgen += self.cm_template
        if self.saveAllResults:
          cmstr = f', {print_confusion_matrix("CM", self.cm_template)}'
        if self.by_instance:
          self.cms[self.idx,:,:] = self.cm_template
      else:
        cmstr = ', CM=[]'
      if self.by_instance:
        self.bads[self.idx]        = num_bad/num_vals
        self.idx += 1
      self.count_bad     += num_bad
      self.count_nums    += num_vals
      self.count_grids   += 1
      if self.saveAllResults:
        if phi_thetas is None:
          idstr = ''
        else:
          idstr = f' angles {phi_thetas[k,:]} / trigonometric functions {beam_params[k,:]}, '
        if self.alsoRadError:
          phystr = f',  RADIATION ERROR: n={count_n}, rad_error={raderror}'
        else:
          phystr = ''
        self.FILE.write(f"\n\nrealidx {realidx:5d}{idstr}, ratio bad values {(num_bad):>6d}/{num_vals:<6d}={(num_bad)/num_vals:0.3f}{cmstr}{phystr}\n")
    return prediction

  def summary_str(self):
    num_all = self.count_nums
    if self.compute_CM:
      cmstr = f', {print_confusion_matrix("ALL_CM", self.confmatrixgen)}'
    else:
      cmstr = ', ALL_CM=[]'
    if self.alsoRadError:
      phystr = f',  RADIATION ERROR: n={self.count_n/self.count}, rad_error={self.count_errordB/self.count}'
    else:
      phystr=''
    return f"for {int(self.count_grids)} grids: global bad values {self.count_bad:>6d}/{num_all:<6d} = {self.count_bad/num_all}{cmstr}{phystr}"

def get_confusion_matrix(values, cutoff, confusion_matrix, ground_truth, prediction):
  for i_gt, v_gt in enumerate(values):
    for i_pred, v_pred in enumerate(values):
      confusion_matrix[i_gt, i_pred] += np.logical_and(np.abs(ground_truth-v_gt)<cutoff, np.abs(prediction-v_pred)<cutoff).sum()

def print_confusion_matrix(name, confusion_matrix):
  return f'{name}=[{",".join(str(a) for a in confusion_matrix.ravel())}]'

def cart2sph_just_elev(x,y,z):
  hypotxy = np.hypot(x,y)
  elev    = np.arctan2(z,hypotxy)
  return elev

class ZeroIncidenceRadiationError:

  def __init__(self, sz=10, limit_mask=-9, incidence=0, cellstep=None):
    
    self.limit_mask = limit_mask
    
    c     = 3e8       # Velocidad de la luz
    freq  = 25e9      # Frecuencia de trabajo
    lambd = c/freq    # Longitud de onda
    k     = (2*np.pi)/lambd

    # Feed
    print(f'Mira incidencia {incidence}')
    theta_feed = incidence*np.pi/180    # Angulo en theta del feed
    phi_feed   = 0*np.pi/180    # Angulo rn phi del feed
    d_foco     = 15000e-3       # Campo lejano 
    x_feed     = np.sin(theta_feed)*np.cos(phi_feed)*d_foco # Posición-X del feed (metros)
    y_feed     = np.sin(theta_feed)*np.sin(phi_feed)*d_foco # Posición-Y del feed (metros)
    z_feed     = np.cos(theta_feed)*d_foco                  # Posición-Z del feed (metros)
    qfeed      = 27                         # Factor del feed para la amplitud del campo
    qelem      = 1                          # Factor del elemento para la amplitud del campo
    Gamma      = 1                          # Coeficiente de reflexión del RA
    self.mX    = sz     # No. de elementos en la dirección-X
    self.mY    = sz     # No. de elementos en la dirección-Y
    perX       = cellstep #8.5e-3 # Distancia entre elementos en la dirección-X (PERIODICIDAD)
    perY       = cellstep #8.5e-3 # Distancia entre elementos en la dirección-Y (PERIODICIDAD)

    R        = np.zeros((self.mX,self.mY,3))
    PosEleX  = np.linspace(-(self.mX-1)/2,  (self.mX-1)/2, num=self.mX, endpoint=True) * perX # Posición de los elementos (en el eje X)
    PosEleY  = np.linspace( (self.mY-1)/2, -(self.mY-1)/2, num=self.mY, endpoint=True) * perY # Posición de los elementos (en el eje Y)
    R[:,:,0] = PosEleX.reshape(( 1,-1)) # Matriz de posiciones de X
    R[:,:,1] = PosEleY.reshape((-1, 1)) # Matriz de posiciones de Y
    R_i      = np.sqrt((R[:,:,0]-x_feed)**2 + (R[:,:,1]-y_feed)**2 + (R[:,:,2]-z_feed)**2)

    self.step_phi     = 181
    self.step_theta   = 361
    v_phi             = np.linspace(0,np.pi,self.step_phi)
    v_theta           = np.linspace(-np.pi/2,np.pi/2,self.step_theta)
    r_mn              = R
    matrix_phi        = np.empty((self.step_phi, self.step_theta), dtype=v_phi.dtype)
    matrix_phi[:,:]   = v_phi.reshape((-1,1))
    matrix_theta      = np.empty((self.step_phi, self.step_theta), dtype=v_phi.dtype)
    matrix_theta[:,:] = v_theta.reshape((1,-1))
    
    sin_matrix_theta   = np.sin(matrix_theta)
    cos_matrix_theta   = np.cos(matrix_theta)
    cos_matrix_phi     = np.cos(matrix_phi)
    sin_matrix_phi     = np.sin(matrix_phi)
    sin_theta_cos_phi  = sin_matrix_theta * cos_matrix_phi
    sin_theta_sin_phi  = sin_matrix_theta * sin_matrix_phi
    r_e_f              = [x_feed-r_mn[:,:,0], y_feed-r_mn[:,:,1], z_feed-r_mn[:,:,2]]
    theta_e_mn         = cart2sph_just_elev(*r_e_f)
    tau_mn             = np.power(np.cos(np.pi/2 - theta_e_mn), qelem) #Modeled by a cosine model whose pointing angle is pi/2 (z-direction)
    theta_f_center     = cart2sph_just_elev(x_feed, y_feed, z_feed)
    theta_f_mn         = cart2sph_just_elev(*r_e_f) #*r_f_e
    angle              = -k*R_i
    I_mn_constant      = np.power(np.cos(theta_f_center - theta_f_mn), qfeed) * (np.cos(angle)+np.sin(angle)*1j) * tau_mn
    self.aux_mn        = np.zeros((self.mX, self.mY, self.step_phi, self.step_theta), dtype=np.complex64)
    for ii in range(self.mX):
      for jj in range(self.mY):
        dot_rmn_uo               = r_mn[ii,jj,0]*sin_theta_cos_phi + r_mn[ii,jj,1]*sin_theta_sin_phi + r_mn[ii,jj,2]*cos_matrix_theta
        angle                    = k*dot_rmn_uo
        A_mn                     = np.power(cos_matrix_theta, qelem) * (np.cos(angle)+np.sin(angle)*1j)
        self.aux_mn[ii,jj,:,:] = A_mn * I_mn_constant[ii,jj]

  def compute_raderror(self, prediction, real_grid):
    E_pattern_real   = 0
    E_pattern_salida = 0
    
    angle_real      = real_grid  * np.pi
    angle_salida    = prediction * np.pi
    complejo_real   = np.cos(angle_real  )+np.sin(angle_real  )*1j
    complejo_salida = np.cos(angle_salida)+np.sin(angle_salida)*1j
    
    for ii in range(self.mX):
      for jj in range(self.mY):
        aux_real         = self.aux_mn[ii,jj] * complejo_real  [ii,jj]
        aux_salida       = self.aux_mn[ii,jj] * complejo_salida[ii,jj]
        E_pattern_real   = E_pattern_real     + aux_real
        E_pattern_salida = E_pattern_salida   + aux_salida

    FA_power_dB_real   = 20*np.log10(np.abs(E_pattern_real))
    FA_power_dB_salida = 20*np.log10(np.abs(E_pattern_salida))
    FA_power_dB_real   = np.maximum(FA_power_dB_real,   0)
    FA_power_dB_salida = np.maximum(FA_power_dB_salida, 0)

    max_EdB = FA_power_dB_real.max()
    EdB     = FA_power_dB_real - max_EdB
    mask    = np.maximum(EdB, self.limit_mask)
    
    EdB_modificado = FA_power_dB_salida - max_EdB
    field          = np.maximum(EdB_modificado, self.limit_mask)

    bool_mask      = field!=mask

    errors         = np.where(bool_mask, np.power(np.abs(mask-field), 2), 0)
    
    radiation_error= errors.sum()/(181*361)
    
    count_n = bool_mask.sum()

    return count_n, radiation_error

def makeGroundTruthImages(imgbase, phi_thetas, dataset, grids, dcts=None):
  if dataset.nbeams>1:
    raise Exception('dataset images are implemented only for datasets with one beam!!!')
  if not dataset.gridAsAngles:
    grids = grids.astype(np.int8)
  phis                    = np.unique(phi_thetas[:,0])
  thetas                  = np.unique(phi_thetas[:,1])
  cellsize1               = grids.shape[-1]+2#12
  img_all_patterns        = ImageGrid(phis.size, thetas.size, cellsize=[cellsize1, cellsize1], isuint8=True, bitUp=dataset.bitUp, bitDown=dataset.bitDown, backgroundIsRed=not dataset.gridAsAngles)
  if dcts is not None:
    img_all_dcts          = ImageGrid(phis.size, thetas.size, cellsize=[cellsize1, cellsize1], isuint8=True, bitUp=dataset.bitUp, bitDown=dataset.bitDown)
  for idx, (phi, theta) in enumerate(phi_thetas):
    if False and idx % 1000 == 0:
      print(f'Processing grid with idx {idx}/{phi_thetas.shape[0]}: phi {phi}, theta {theta}')
    #first, draw in the image with each pattern as a sub-image
    i0   = (  phis==phi  ).nonzero()[0]
    i1   = (thetas==theta).nonzero()[0]
    if not (len(i0)==1 and len(i1)==1):
      raise Exception(f'For phi {phi} and theta {theta}, there are more than one ondex: {i}')

    if dataset.gridAsAngles:
      g = dataset.colorizeAngles(grids[idx,0,:,:], color=True)
      img_all_patterns.putCell(i0, i1, g)
    else:
      g  = grids[idx,:,:]
      img_all_patterns.putBinaryCell(i0, i1, g)
    if dcts is not None:
      if dataset.gridAsAngles:
        dcto = dcts[idx,0]
      else:
        dcto = dcts[idx]
      dct  = (dcto/np.abs(dcto).max()*255).astype(np.int16)
      dct_p,  dct_n  = separate_pos_neg(dct)
      #dcto_p, dcto_n = separate_pos_neg(dcto)
      img_all_dcts.dealPositiveAndNegativeValuesToSingleCell(dct_p, dct_n, i0, i1)

  cv2.imwrite(f'{imgbase}all_patterns.png',       img_all_patterns.img)
  if dcts is not None:
    cv2.imwrite(f'{imgbase}all_dcts.png',       img_all_dcts.img)
  #import code; code.interact(local=vars())


def makeValidationImages(imgbase, phi_thetas, dataset, grids, preds):
  if dataset.nbeams>1:
    raise Exception('dataset images are implemented only for datasets with one beam!!!')
  phis                    = np.unique(phi_thetas[:,0])
  thetas                  = np.unique(phi_thetas[:,1])
  cellsize1               = grids.shape[-1]+2#12
  img_all_preds           = ImageGrid(phis.size, thetas.size, cellsize=[cellsize1, cellsize1], isuint8=True, bitUp=dataset.bitUp, bitDown=dataset.bitDown, backgroundIsRed=not dataset.gridAsAngles)
  img_all_diffs           = ImageGrid(phis.size, thetas.size, cellsize=[cellsize1, cellsize1], isuint8=True, bitUp=dataset.bitUp, bitDown=dataset.bitDown)
  for idx, (phi, theta) in enumerate(phi_thetas):
    if False and idx % 1000 == 0:
      print(f'Processing grid with idx {idx}/{phi_thetas.shape[0]}: phi {phi}, theta {theta}')
    #first, draw in the image with each pattern as a sub-image
    i0   = (  phis==phi  ).nonzero()[0]
    i1   = (thetas==theta).nonzero()[0]
    if not (len(i0)==1 and len(i1)==1):
      raise Exception(f'For phi {phi} and theta {theta}, there are more than one ondex: {i}')

    if dataset.gridAsAngles:
      p  = preds[idx,:,:]
      gs = grids[idx,:,:,:]
      bestdiff    = None
      bestnumdiff = np.inf
      for g in gs:
        diff    = np.abs(p-g)
        numdiff = diff.sum()
        if numdiff<bestnumdiff:
          bestnumdiff = numdiff
          bestdiff    = diff
      img_all_preds.putCell(i0, i1, dataset.colorizeAngles(p,        color=True))
      img_all_diffs.putCell(i0, i1, dataset.colorizeAngles(bestdiff, color=False))
    else:
      p  = preds[idx,:,:]
      g  = grids[idx,:,:]
      d1 = np.logical_xor(g, p)
      d2 = np.logical_xor(g, np.logical_not(p))
      d  = d1 if d1.sum()<=d2.sum() else d2
      img_all_preds   .putBinaryCell(i0, i1, p)
      img_all_diffs   .putBinaryCell(i0, i1, d)

  cv2.imwrite(f'{imgbase}all_predictions.png',    img_all_preds.img)
  cv2.imwrite(f'{imgbase}all_differences.png',    img_all_diffs.img)
  #import code; code.interact(local=vars())

# helper to separate positive and negative components into two matrices
def separate_pos_neg(matrix):
  matrix_p = matrix.copy()
  matrix_p[matrix_p<0] = 0
  matrix_n = matrix.copy()
  matrix_n[matrix_n>0] = 0
  matrix_n=-matrix_n
  return matrix_p, matrix_n


def get_custom_idct(patternSize, device):
  return Artisanal2DTransform(patternSize, original_idct, device=device, also_torch=True)
  #return Artisanal2DTransform(patternSize, original_idct, also_torch=False).compute_np

def get_args():
  parser = argparse.ArgumentParser(description='fitting activation grids to beam shapes')
  parser.add_argument('--device', type=int, default=0)
  parser.add_argument('--input', type=str)
  parser.add_argument('--exclude', action='store_true')
  parser.add_argument('--no-exclude', dest='exclude', action='store_false')
  parser.set_defaults(exclude=False)
  parser.add_argument('--exclude_threshold', type=int, default=0)
  parser.add_argument('--beam_decoder_initialization', type=str, default=None)
  parser.add_argument('--just_one_pixel', type=str, default='None')
  parser.add_argument('--traindir', type=str, default=None)
  parser.add_argument('--config_file', type=str, default=None)
  parser.add_argument('--checkpoint_file', type=str, default=None)
  parser.add_argument('--validate_log', type=str, default=None)
  parser.add_argument('--img_basename', type=str, default=None)
  parser.add_argument('--matfilename', type=str, default=None)
  parser.add_argument('--partition_ranges', type=str, default=None)
  parser.add_argument('--filter_angles', type=str, default=None)
  parser.add_argument('--useoldmax_straightAngles', action='store_true')
  parser.add_argument('--no-useoldmax_straightAngles', dest='useoldmax_straightAngles', action='store_false')
  parser.set_defaults(useoldmax_straightAngles=True)
  parser.add_argument('--outputType', type=str, default='straightBinary')
  parser.add_argument('--patternSize', type=int, default=10)
  parser.add_argument('--incidence', type=int, default=0)
  parser.add_argument('--alsoNegated', action='store_true')
  parser.add_argument('--no-alsoNegated', dest='alsoNegated', action='store_false')
  parser.add_argument('--classicRepresentation1bit', action='store_true')
  parser.add_argument('--no-classicRepresentation1bit', dest='classicRepresentation1bit', action='store_false')
  parser.add_argument('--cellstep', type=float, default=9e-3)
  parser.set_defaults(classicRepresentation1bit=True)
  parser.set_defaults(alsoNegated=False)
  args = parser.parse_args()
  non_overrideable_params = {'just_one_pixel', 'traindir', 'config_file', 'checkpoint_file', 'resume_epoch', 'validate', 'validate_log', 'img_basename', 'matfilename', 'export_onnx', 'export_onnx_dct', 'onnx_name', 'weights'}
  return args, non_overrideable_params

if __name__ == "__main__":
  args, non_overrideable_params = get_args()
  device = torch.device(f'cuda:{args.device}')
  traindir = args.traindir
  if not os.path.isdir(traindir):
    raise Exception(f'trying to use traindir <{traindir}>, but it does not exist!')
  if args.config_file is None:
    raise Exception(f'argument --config_file must be provided!!!')
  if args.checkpoint_file is None:
    raise Exception(f'argument --checkpoint_file must be provided!!!')
  conf_filename       = os.path.join(traindir, args.config_file)
  checkpoint_filename = os.path.join(traindir, args.checkpoint_file)
  if not os.path.isfile(conf_filename):
    raise Exception(f'config file {conf_filename} does not exist!!!')
  if not os.path.isfile(checkpoint_filename):
    raise Exception(f'checkpoint file {checkpoint_filename} does not exist!!!')
  with open(conf_filename, 'r') as f:
    conf = json.load(f)
  for k,v in conf.items():
    if k not in non_overrideable_params:
      setattr(args, k, v)
  if args.filter_angles is not None:
    args.filter_angles = eval(args.filter_angles)
  decoder = eval(args.beam_decoder_initialization).to(device)
  checkpoint = torch.load(checkpoint_filename, map_location=device)
  decoder.load_state_dict(checkpoint)
  if args.partition_ranges is not None:
    if not hasattr(decoder, 'setRangeSelector'):
      raise Exception('If you set the argument --partition_ranges, the decoder must support the setRangeSelector method!!!')
    partition_ranges = eval(args.partition_ranges)
    partition_ranges = np.array(partition_ranges)
    partition_ranges = AnglesRangesSelector(partition_ranges)
    decoder.setRangeSelector(partition_ranges)
  dataset = OneBeamGridDataset(args.input, sz=args.patternSize, device=device, filter_angles=args.filter_angles, classicRepresentation1bit=args.classicRepresentation1bit)
  #import code; code.interact(local=vars())
  validate_log = os.path.join(traindir, args.validate_log)
  img_basename = args.img_basename
  if img_basename is not None:
    img_basename = os.path.join(traindir, img_basename)
  matfilename = args.matfilename
  if matfilename is not None:
    matfilename = os.path.join(traindir, matfilename)
  do_validate(device, dataset, decoder, validate_log, img_basename, matfilename, outputType=args.outputType, alsoNegated=args.alsoNegated, patternSize=args.patternSize, incidence=args.incidence, cellstep=args.cellstep, useoldmax_straightAngles=args.useoldmax_straightAngles)




