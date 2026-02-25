from utils.scales import scale_complex_matrices, resize_matrix
from utils.resize import ComplexMatrixResizer

# Load dataset
root_dir = "/data/training/mats/freq/"
output_dir = "exps/time/PDNet_YS/afm"
eval_dir = "evals/time/PDNet_YS/afm"


categories = [
    "P_prbs",
    "Z_PRBS_waveform_no_noise",
    "Z_PRBS_waveform",
    "Y_PRBS_waveform_no_noise",
    "Y_PRBS_waveform",
]

snrs = [
    "-20",
    "-10",
    "-5",
    "0",
    "5",
    "10",
    "20",
    "30",
]

transform_funcs = [
    # MinMaxScaler(feature_range=None, device="mps"),
    # ComplexMatrixResizer(method="lanczos"),
]
