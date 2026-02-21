from models.bases import pdnet_y, pdnet_ys, dncnn_ys, dncnn_y, afm_model
from models.model import DenoiseModel

batch_size = 1
num_workers = 1
pin_memory = False
device = "mps"

epochs = 5
lr = 0.001
betas = (0.5, 0.999)
train_test_split = 0.8

# Denoise model hyperparameters
input_shape = (2, 255, 256)
hidden_channel = 16
level = 3
kls_thesh = 0.0001

depth = 10
img_channels = 2
n_filters = 64
kernel_size = 3

domain = "time"
current_model = pdnet_ys.PDNet(
    input_shape=input_shape, hidden_channel=hidden_channel, level=level
)
# current_model = dncnn_y.DnCNN(
#     depth=depth,
#     img_channels=img_channels,
#     n_filters=n_filters,
#     kernel_size=kernel_size,
# ).to(device)
model = DenoiseModel(model=current_model, domain=domain).to(device)

# training model
afm_model = afm_model.AFM_PRBS(domain=domain, device=device)
