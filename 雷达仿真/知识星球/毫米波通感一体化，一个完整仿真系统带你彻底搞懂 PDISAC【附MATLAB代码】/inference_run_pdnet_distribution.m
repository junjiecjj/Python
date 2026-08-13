function [dist_maps, dist_labels, metadata_inference] = inference_run_pdnet_distribution( ...
    RD_map_input_inference, RD_map_prbs_inference, checkpoint_inference, options_inference)
%INFERENCE_RUN_RDPDNET_DISTRIBUTION Export RDPDNet distributions for 3-D plotting.
%
% Calls alg_pdisac/scripts/distribution_pdnet_mat.py, which runs
% model.distribution(Z_rd, Z_rd_prbs) and returns every distribution of the
% hierarchy (observed RD map, clean RD map, bottom-up and top-down latents,
% and the reconstruction), each averaged over the batch (Monte-Carlo trials)
% and over the channel dimension.
%
% Inputs
%   RD_map_input_inference : complex H x W or B x H x W (noisy, data-embedded)
%   RD_map_prbs_inference  : complex, same shape (clean / data-free reference)
%   checkpoint_inference   : path to the trained RDPDNet checkpoint
%
% Outputs
%   dist_maps   : K x H x W double, one aggregated map per distribution
%   dist_labels : 1 x K cellstr of LaTeX labels (e.g. '$p(\mathbf{Z}_{\rm rd})$')
%   metadata_inference : remaining fields returned by the Python side

arguments
    RD_map_input_inference {mustBeNumeric, mustBeNonempty}
    RD_map_prbs_inference  {mustBeNumeric, mustBeNonempty}
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
distribution_script = fullfile(repo_root, "alg_pdisac", "scripts", ...
    "distribution_pdnet_mat.py");

must_exist(python_executable, "Python executable");
must_exist(distribution_script, "RDPDNet distribution script");
must_exist(checkpoint_inference, "RDPDNet checkpoint");

if ~isequal(size(RD_map_input_inference), size(RD_map_prbs_inference))
    error("PDISAC:DistributionShapeMismatch", ...
        "Noisy input size %s differs from clean reference size %s.", ...
        mat2str(size(RD_map_input_inference)), mat2str(size(RD_map_prbs_inference)));
end

exchange_dir = options_inference.ExchangeDirectory;
if ~isfolder(exchange_dir)
    mkdir(exchange_dir);
end
token = string(char(java.util.UUID.randomUUID));
input_file_inference = fullfile(exchange_dir, ...
    "dataset_distribution_input_" + token + ".mat");
output_file_inference = fullfile(exchange_dir, ...
    "dataset_distribution_output_" + token + ".mat");

trial_id_inference = options_inference.TrialID; %#ok<NASGU>
RD_map_input_inference = single(RD_map_input_inference); %#ok<NASGU>
RD_map_prbs_inference  = single(RD_map_prbs_inference);  %#ok<NASGU>
save(input_file_inference, "RD_map_input_inference", "RD_map_prbs_inference", ...
    "trial_id_inference", "-v7");

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
    python_executable, distribution_script, checkpoint_inference, ...
    input_file_inference, output_file_inference, options_inference.Device, ...
    options_inference.BatchSize);
[status_inference, message_inference] = system(command);
if status_inference ~= 0
    error("PDISAC:RDPDNetDistributionFailed", ...
        "RDPDNet distribution export failed with status %d:\n%s", ...
        status_inference, message_inference);
end
if ~isfile(output_file_inference)
    error("PDISAC:MissingDistributionOutput", ...
        "Python completed without producing %s.", output_file_inference);
end

result_inference = load(output_file_inference);
if ~isfield(result_inference, "dist_maps_inference")
    error("PDISAC:InvalidDistributionOutput", ...
        "Output MAT file lacks dist_maps_inference.");
end
dist_maps = double(result_inference.dist_maps_inference);
dist_labels = cellstr(string(result_inference.dist_labels_inference));
metadata_inference = rmfield(result_inference, "dist_maps_inference");
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
