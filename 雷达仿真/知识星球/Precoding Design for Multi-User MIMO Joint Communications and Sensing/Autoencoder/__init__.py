import sys, os
# insert at position 1 in the path, as 0 is the path of this file.
sys.path.insert(1, os.path.dirname(__file__))


from imports import * 
from NN_classes import Beamformer,Encoder,Decoder,Radar_receiver
from functions import * 
from training_routine import train_network