import pickle
import torch
import logging
import argparse
from global_config import global_config
from data_processing import load_data, create_data_loader
from train import initialize_model, train, validate
from utils import set_seeds_and_reproducibility, save_rmse_mae_stats, save_model_predictions
from custom_logging import setup_logging

# Setup custom logging
setup_logging()

# Clear CUDA memory cache
torch.cuda.empty_cache()

def parse_args():
    """
    Parses command-line arguments related to the configuration of the model, data preprocessing,
    and experiment settings using argparse.

    Returns:
        argparse.Namespace: An object containing attributes for each command line argument collected.
        This object allows easy access to the configuration parameters throughout the code.
    """
    parser = argparse.ArgumentParser(description="Training and evaluation.")

    # Model parameters
    parser.add_argument('--model', type=str, default='GAT', help='Model type')
    parser.add_argument('--learning_rate', type=float, default=0.0006703373871168542, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--test_batch_size', type=int, default=8, help='Test batch size')
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--out_channels', type=int, default=128, help='Output channels')
    parser.add_argument('--in_channels', type=int, default=22, help='Input channels')
    parser.add_argument('--out_features', type=int, default=5, help='Output features')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')

    # Data Preprocessing
    parser.add_argument('--three_dim', type=bool, default=True, help='3D data flag')
    parser.add_argument('--required_features', nargs='+', default=['node_positions', 'node_noise'], help='Required features')
    parser.add_argument('--additional_features', nargs='+', default=['weighted_centroid_radius', 'weighted_centroid_sin_theta', 'weighted_centroid_cos_theta', 'weighted_centroid_sin_az', 'weighted_centroid_cos_az', 'dist_to_wcl', 'median_noise', 'max_noise', 'noise_differential', 'vector_x', 'vector_y', 'vector_z', 'rate_of_change_signal_strength'], help='Engineered node features')
    parser.add_argument('--num_neighbors', type=int, default=3, help='Number of graph KNN')
    parser.add_argument('--downsample', type=bool, default=True, help='Downsample flag')
    parser.add_argument('--max_nodes', type=int, default=1000, help='Maximum number of nodes to downsample to.')
    parser.add_argument('--val_discrite_coeff', type=float, default=0.1, help='Validation discretization coefficient')
    parser.add_argument('--test_discrite_coeff', type=float, default=0.1, help='Test discretization coefficient')
    parser.add_argument('--aug', nargs='+', default=['drop_node'], help='Graph data augmentation methods')

    # Experiments
    parser.add_argument('--experiments_folder', type=str, default='experiments/', help='Experiments folder')
    parser.add_argument('--dataset_path', type=str, default='data/dynamic_data.pkl', help='Dataset path')
    parser.add_argument('--dynamic', type=bool, default=True, help='Dynamic flag')
    parser.add_argument('--inference', type=bool, default=False, help='Inference flag')
    parser.add_argument('--reproduce', type=bool, default=True, help='Reproduce flag')
    parser.add_argument('--plot_network', type=bool, default=False, help='Plot network flag')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers')

    return parser.parse_args()


def main():
    """
    Executes the training, evaluation, and inference pipeline for the GNN model.

    This function orchestrates the process based on the configuration settings defined in `global_config.args`.
    It supports both dynamic and static data class scenarios.

    Workflow:
    - Sets seeds for reproducibility.
    - Loads or initializes datasets and corresponding DataLoader objects.
    - Initializes model.
    - Executes training and validation cycles, saving the best model based on validation loss.
    - In inference mode, loads a trained model and performs evaluations directly.
    - Aggregates and logs RMSE and MAE statistics for model performance analysis.
    """
    seeds = [1]

    args = parse_args()

    # Set the parsed arguments to the global configuration
    global_config.args = args

    if global_config.args.dynamic:
        dataset_classes = ['dynamic']
    else:
        dataset_classes = ['circle', 'triangle', 'rectangle', 'random',
                           'circle_jammer_outside', 'triangle_jammer_outside',
                           'rectangle_jammer_outside', 'random_jammer_outside']

    for data_class in dataset_classes:
        rmse_vals = []
        mae_vals = []

        logging.info("Executing GNN jamming source localization script")
        for trial_num, seed in enumerate(seeds):
            # Set seed for data saving
            set_seeds_and_reproducibility(100)

            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logging.debug("Device: ", device)

            # Set path to save or load model
            model_path = f"{global_config.args.experiments_folder}trained_model_seed{seed}_{global_config.args.model}_{data_class}.pth"

            # Inference
            if global_config.args.inference:
                model_path = 'experiments/trained_model_seed1_CAGE_polar_knn3_unit_sphere_1000noise_dynamic.pth'
                train_dataset, val_dataset, test_dataset = load_data(data_class, global_config.args.experiments_folder)
                _, _, test_loader, deg_histogram = create_data_loader(train_dataset, val_dataset, test_dataset, global_config.args.experiments_folder)
                model, optimizer, scheduler, criterion = initialize_model(device, len(test_loader), deg_histogram)

                # Load trained model
                model.load_state_dict(torch.load(model_path))

                # Predict jammer position
                predictions, actuals, err_metrics, perc_completion_list, pl_exp_list, sigma_list, jtx_list, num_samples_list, weight_list = validate(model, test_loader, criterion, device, test_loader=True)
                rmse_vals.append(err_metrics['rmse'])
                mae_vals.append(err_metrics['mae'])  # MAE from err_metrics
                logging.info(f"Seed {seed}, MAE: {err_metrics['mae']}, RMSE: {err_metrics['rmse']}")

                # Save model predictions
                save_model_predictions(seed, data_class, predictions, actuals, perc_completion_list, pl_exp_list, sigma_list, jtx_list, num_samples_list, weight_list)

                return predictions
            else:
                # Load the datasets
                train_dataset, val_dataset, test_dataset = load_data(data_class, global_config.args.experiments_folder)
                set_seeds_and_reproducibility(seed)

                # Create data loaders
                train_loader, val_loader, test_loader, deg_histogram = create_data_loader(train_dataset, val_dataset, test_dataset, global_config.args.experiments_folder)

                # Initialize model
                model, optimizer, scheduler, criterion = initialize_model(device, len(train_loader), deg_histogram)

                # Training
                logging.info("Training and validation loop")
                train_details ={}
                val_details ={}
                best_val_loss = float('inf')
                for epoch in range(global_config.args.num_epochs):
                    train_loss, train_detailed_metrics = train(model, train_loader, optimizer, criterion, device, len(train_loader), scheduler)
                    val_loss, val_detailed_metrics = validate(model, val_loader, criterion, device)
                    train_details[epoch] = train_detailed_metrics
                    val_details[epoch] = val_detailed_metrics
                    logging.info(f'Epoch: {epoch}, Train Loss: {train_loss:.15f}, Val Loss: {val_loss:.15f}')

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = model.state_dict()

                # Save trained model
                torch.save(best_model_state, model_path)

                # Save the train and validation epoch results
                validation_details_path = global_config.args.experiments_folder + f'validation_details_{global_config.args.model}_{data_class}_seed{seed}.pkl'
                train_details_path = global_config.args.experiments_folder + f'train_details_{global_config.args.model}_{data_class}_seed{seed}.pkl'

                with open(validation_details_path, 'wb') as f:
                    pickle.dump(val_details, f)

                with open(train_details_path, 'wb') as f:
                    pickle.dump(train_details, f)

                # Evaluate the model on the test set
                model.load_state_dict(best_model_state)
                predictions, actuals, err_metrics, perc_completion_list, pl_exp_list, sigma_list, jtx_list, num_samples_list, weight_list = validate(model, test_loader, criterion, device, test_loader=True)
                rmse_vals.append(err_metrics['rmse'])
                mae_vals.append(err_metrics['mae'])  # MAE from err_metrics
                logging.info(f"Seed {seed}, MAE: {err_metrics['mae']}, RMSE: {err_metrics['rmse']}")

                # Save model predictions
                save_model_predictions(seed, data_class, predictions, actuals, perc_completion_list, pl_exp_list, sigma_list, jtx_list, num_samples_list, weight_list)

        # Save RMSE and MAE results over trials
        save_rmse_mae_stats(rmse_vals, mae_vals, data_class)

if __name__ == "__main__":
    main()

