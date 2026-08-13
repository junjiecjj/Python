function [RD_map_denoised_inference, metadata_inference] = inference_run_pdnet( ...
    RD_map_input_inference, checkpoint_inference, options_inference)
%INFERENCE_RUN_RDPDNET Run trained Python RDPDNet and return a complex RD map.
%
% The RDPDNet inference dataset may be H-by-W or B-by-H-by-W. Exchange variable
% names use the suffix "_inference". Python is launched once per call, so
% pass a batch for Monte-Carlo evaluation when possible.
%
% Example:
%       "full_training", "fold_1", "checkpoints", "best_checkpoint.pth");

arguments
    RD_map_input_inference {mustBeNumeric, mustBeNonempty}
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
% Absolute paths (e.g. a conda env interpreter outside the repo) are used
% as-is; repo-relative paths are resolved against the repository root.
if ~(startsWith(python_executable, filesep) || ...
     ~isempty(regexp(python_executable, '^[A-Za-z]:[\\/]', 'once')) || ...
     isfile(python_executable))
    python_executable = fullfile(repo_root, python_executable);
end
inference_script = fullfile(repo_root, "alg_pdisac", "scripts", ...
    "inference_pdnet_mat.py");

must_exist(python_executable, "Python executable");
must_exist(inference_script, "RDPDNet inference script");
must_exist(checkpoint_inference, "RDPDNet checkpoint");

exchange_dir = options_inference.ExchangeDirectory;
if ~isfolder(exchange_dir)
    mkdir(exchange_dir);
end
token = string(char(java.util.UUID.randomUUID));
input_file_inference = fullfile(exchange_dir, ...
    "dataset_inference_input_" + token + ".mat");
output_file_inference = fullfile(exchange_dir, ...
    "dataset_inference_output_" + token + ".mat");

trial_id_inference = options_inference.TrialID; %#ok<NASGU>
RD_map_input_inference = single(RD_map_input_inference); %#ok<NASGU>
save(input_file_inference, "RD_map_input_inference", "trial_id_inference", "-v7");

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
    error("PDISAC:RDPDNetInferenceFailed", ...
        "RDPDNet inference failed with status %d:\n%s", ...
        status_inference, message_inference);
end
if ~isfile(output_file_inference)
    error("PDISAC:MissingInferenceOutput", ...
        "Python completed without producing %s.", output_file_inference);
end

result_inference = load(output_file_inference);
if ~isfield(result_inference, "RD_map_denoised_inference")
    error("PDISAC:InvalidInferenceOutput", ...
        "Output MAT file lacks RD_map_denoised_inference.");
end
RD_map_denoised_inference = result_inference.RD_map_denoised_inference;
metadata_inference = rmfield(result_inference, "RD_map_denoised_inference");

if ~isequal(size(RD_map_denoised_inference), size(RD_map_input_inference))
    error("PDISAC:InferenceShapeMismatch", ...
        "RDPDNet output size %s differs from input size %s.", ...
        mat2str(size(RD_map_denoised_inference)), ...
        mat2str(size(RD_map_input_inference)));
end
if any(~isfinite(RD_map_denoised_inference(:)))
    error("PDISAC:NonfiniteInferenceOutput", ...
        "RDPDNet output contains NaN or Inf values.");
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
