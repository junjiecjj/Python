function [RD_map_denoised_inference, M_afm_inference, metadata_inference] = ...
    inference_run_pdnet_mask(RD_map_input_inference, RD_map_prbs_inference, ...
                             checkpoint_inference, options_inference)
%INFERENCE_RUN_RDPDNET_MASK Run trained RDPDNet AND rebuild the AFM mask.
%
% Companion to inference_run_pdnet: calls scripts/inference_pdnet_mask_mat.py,
% which additionally reconstructs the training-time adversarial mask
%     M_afm = MaskNet(Z_rd_hat) .* (1 - M_tars),  M_tars = create_mask(Z_rd_prbs)
% Pass the no-data RD map (RD_map_noise_no_data_shifted) as
% RD_map_prbs_inference; pass [] to skip M_tars (then M_afm = M_mask).
%
% Returns:
%   RD_map_denoised_inference : complex, same shape as input
%   M_afm_inference           : single, (B x) 2 x H x W in [0,1]
%   metadata_inference        : remaining exchange variables (incl. M_mask)

arguments
    RD_map_input_inference {mustBeNumeric, mustBeNonempty}
    RD_map_prbs_inference
    checkpoint_inference (1,1) string
    options_inference.PythonExecutable (1,1) string = ""
    options_inference.RepoRoot (1,1) string = ""
    options_inference.Device (1,1) string {mustBeMember(options_inference.Device, ...
        ["auto","cpu","cuda","mps"])} = "auto"
    options_inference.BatchSize (1,1) double {mustBeInteger,mustBePositive} = 8
    options_inference.TrialID = int64(1)
    options_inference.KeepExchangeFiles (1,1) logical = false
    options_inference.ExchangeDirectory (1,1) string = string(tempdir)
end

this_file = string(mfilename("fullpath"));
default_root = string(fileparts(fileparts(this_file)));
repo_root = options_inference.RepoRoot;
if strlength(repo_root) == 0
    repo_root = default_root;
end

main1_matlab_config;  % defines PDISAC_cfg (inference paths/device)
if strlength(options_inference.Device) == 0
    options_inference.Device = string(PDISAC_cfg.inference.device);
end

python_executable = options_inference.PythonExecutable;
if strlength(python_executable) == 0
    python_executable = string(PDISAC_cfg.inference.python_executable);
end
if ~(startsWith(python_executable, filesep) || ...
     ~isempty(regexp(python_executable, '^[A-Za-z]:[\\/]', 'once')) || ...
     isfile(python_executable))
    python_executable = fullfile(repo_root, python_executable);
end
inference_script = fullfile(repo_root, "alg_pdisac", "scripts", ...
    "inference_pdnet_mask_mat.py");

must_exist(python_executable, "Python executable");
must_exist(inference_script, "RDPDNet mask inference script");
must_exist(checkpoint_inference, "RDPDNet checkpoint");

exchange_dir = options_inference.ExchangeDirectory;
if ~isfolder(exchange_dir)
    mkdir(exchange_dir);
end
token = string(char(java.util.UUID.randomUUID));
input_file_inference = fullfile(exchange_dir, ...
    "dataset_inference_mask_input_" + token + ".mat");
output_file_inference = fullfile(exchange_dir, ...
    "dataset_inference_mask_output_" + token + ".mat");

trial_id_inference = options_inference.TrialID; %#ok<NASGU>
RD_map_input_inference = single(RD_map_input_inference); %#ok<NASGU>
if isempty(RD_map_prbs_inference)
    save(input_file_inference, "RD_map_input_inference", "trial_id_inference", "-v7");
else
    RD_map_prbs_inference = single(RD_map_prbs_inference); %#ok<NASGU>
    save(input_file_inference, "RD_map_input_inference", ...
        "RD_map_prbs_inference", "trial_id_inference", "-v7");
end

cleanup_object = onCleanup(@() cleanup_files(input_file_inference, ...
    output_file_inference, options_inference.KeepExchangeFiles)); %#ok<NASGU>

alg_dir = fullfile(repo_root, "alg_pdisac");
old_pythonpath = getenv("PYTHONPATH");
pythonpath_guard = onCleanup(@() setenv("PYTHONPATH", old_pythonpath)); %#ok<NASGU>
if strlength(string(old_pythonpath)) == 0
    setenv("PYTHONPATH", alg_dir);
else
    setenv("PYTHONPATH", alg_dir + pathsep + string(old_pythonpath));
end

command = sprintf('"%s" "%s" --checkpoint "%s" --input "%s" --output "%s" --device %s --batch-size %d', ...
    python_executable, inference_script, checkpoint_inference, ...
    input_file_inference, output_file_inference, options_inference.Device, ...
    options_inference.BatchSize);
[status_inference, message_inference] = system(command);
if status_inference ~= 0
    error("PDISAC:RDPDNetMaskInferenceFailed", ...
        "RDPDNet mask inference failed with status %d:\n%s", ...
        status_inference, message_inference);
end
if ~isfile(output_file_inference)
    error("PDISAC:MissingInferenceOutput", ...
        "Python completed without producing %s.", output_file_inference);
end

result_inference = load(output_file_inference);
if ~isfield(result_inference, "RD_map_denoised_inference") || ...
   ~isfield(result_inference, "M_afm_inference")
    error("PDISAC:InvalidInferenceOutput", ...
        "Output MAT file lacks RD_map_denoised_inference and/or M_afm_inference.");
end
RD_map_denoised_inference = result_inference.RD_map_denoised_inference;
M_afm_inference = result_inference.M_afm_inference;
metadata_inference = rmfield(result_inference, ...
    ["RD_map_denoised_inference", "M_afm_inference"]);

if any(~isfinite(RD_map_denoised_inference(:))) || any(~isfinite(M_afm_inference(:)))
    error("PDISAC:NonfiniteInferenceOutput", ...
        "RDPDNet mask inference output contains NaN or Inf values.");
end
end


function must_exist(path_value, description)
if ~isfile(path_value)
    error("PDISAC:MissingInferenceDependency", "%s not found: %s", ...
        description, path_value);
end
end


function cleanup_files(input_file, output_file, keep_files)
if keep_files
    return;
end
if isfile(input_file), delete(input_file); end
if isfile(output_file), delete(output_file); end
end
