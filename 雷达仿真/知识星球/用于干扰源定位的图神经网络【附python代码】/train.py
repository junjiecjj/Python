import logging
import math
from typing import Tuple, List, Dict


import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from custom_logging import setup_logging
from data_processing import convert_output_eval
from model import GNN
from utils import AverageMeter
from global_config import global_config

setup_logging()


def initialize_model(device: torch.device, steps_per_epoch=None, deg_histogram=None) -> Tuple[GNN, optim.Optimizer, OneCycleLR, torch.nn.Module]:
    """
    Initialize the model, optimizer, scheduler, and loss criterion.

    Args:
        device (torch.device): The device (CPU or GPU) on which to run the model.
        steps_per_epoch (int, optional): Number of steps per epoch, needed for the OneCycleLR scheduler.
        deg_histogram (np.ndarray, optional): Degree histogram for PNA model.

    Returns:
        Tuple[GNN, optim.Optimizer, OneCycleLR, torch.nn.Module]: A tuple containing the initialized model, optimizer,
        scheduler, and loss criterion, respectively.
    """
    logging.info("Initializing model...")
    model = GNN(in_channels=global_config.args.in_channels, dropout_rate=global_config.args.dropout_rate, num_heads=global_config.args.num_heads, model_type=global_config.args.model, hidden_channels=global_config.args.hidden_channels,
                out_channels=global_config.args.out_channels, num_layers=global_config.args.num_layers, deg=deg_histogram).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=global_config.args.learning_rate, weight_decay=global_config.args.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=global_config.args.learning_rate, epochs=global_config.args.num_epochs, steps_per_epoch=steps_per_epoch, pct_start=0.2, anneal_strategy='linear')
    criterion = torch.nn.MSELoss()
    return model, optimizer, scheduler, criterion


def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: torch.device, steps_per_epoch: int,
          scheduler) -> Tuple[float, List[Dict]]:
    """
    Trains model tracking loss and epoch training metrics.

    This function performs forward and backward passes on batches from a DataLoader,
    computes combined losses, and updates the model using optimizer.
    Loss for each batch is tracked using an AverageMeter, and detailed per-graph RMSE and percentage completion
    are stored in a dictionary for further analysis.

    Args:
        model (torch.nn.Module): The GNN model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader providing the training batches.
        optimizer (torch.optim.Optimizer): Optimizer to update model weights.
        criterion (torch.nn.Module): Loss criterion to evaluate prediction accuracy.
        device (torch.device): Device on which to perform the training (e.g., 'cuda' or 'cpu').
        steps_per_epoch (int): Maximum number of batches to process per epoch.
        scheduler: Scheduler to adjust the learning rate across training steps.

    Returns:
        Tuple[float, List[Dict]]: Average loss for the epoch and a list of dictionaries containing train metrics per graph.
    """
    loss_meter = AverageMeter()
    model.train()
    progress_bar = tqdm(train_loader, total=steps_per_epoch, desc="Training", leave=True)

    # Dictionary to store the results by graph index and epoch
    detailed_metrics = []

    for num_batches, data in enumerate(progress_bar):
        if steps_per_epoch is not None and num_batches >= steps_per_epoch:
            break

        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass
        gnn_prediction, final_prediction, weight = model(data)

        # Compute loss for each output
        loss_gnn = criterion(gnn_prediction, data.y)
        loss_final = criterion(final_prediction, data.y)

        # Regularization term
        lambda_reg = 0
        regularization = lambda_reg * torch.sum((1 - weight) ** 2)

        # Combine losses with regularization included in the average
        loss = ((loss_gnn + loss_final) / 2) + regularization

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update AverageMeter with the current batch loss
        loss_meter.update(loss.item(), data.num_graphs)

        # Get the current learning rate from the optimizer
        current_lr = optimizer.param_groups[0]['lr']

        # Log the average loss so far and the current learning rate in the progress bar
        progress_bar.set_postfix({
            "Train Loss (MSE)": loss_meter.avg,
            "Learning Rate": current_lr
        })

        # Dictionary to store individual graph details
        graph_details = {}

        # Calculate RMSE for each graph in the batch
        for idx in range(data.num_graphs):
            prediction = convert_output_eval(final_prediction[idx], data[idx], 'prediction', device)
            actual = convert_output_eval(data.y[idx], data[idx], 'target', device)

            mse = mean_squared_error(actual.cpu().numpy(), prediction.cpu().numpy())
            rmse = math.sqrt(mse)
            perc_completion = data.perc_completion[idx].item()

            # Storing the metrics in the dictionary with graph id as key
            graph_details[idx] = {'rmse': rmse, 'perc_completion': perc_completion}

        # Append to the detailed metrics dict
        detailed_metrics.append(graph_details)

    # Return the average loss tracked by AverageMeter
    return loss_meter.avg, detailed_metrics


def validate(model: torch.nn.Module, validate_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module, device: torch.device, test_loader=False):
    """
    Validates model and returns performance metrics.

    The function computes validation loss and test set performance metrics
    In test set mode, it returns predictions, actuals, graph/instance information,
    and metrics (MAE, MSE, RMSE). For validation set, returns the average MAE/RMSE loss
    and a dictionary containing validation set details during over epochs.

    Args:
        model (torch.nn.Module): The model to validate.
        validate_loader (torch.utils.data.DataLoader): DataLoader providing the validation batches.
        criterion (torch.nn.Module): Loss criterion for evaluating model performance.
        device (torch.device): Device on which to perform the validation (e.g., 'cuda' or 'cpu').
        test_loader (bool): Flag indicating whether to run validation in test set or validation set mode.

    Returns:
        Tuple[float, List[Dict]]: If not in test mode, returns average loss for the validation
                                  and a list of dictionaries containing detailed metrics per graph.
                                  In test mode, returns performance metrics including
                                  predictions, actuals, error metrics, etc.
    """
    model.eval()
    predictions, actuals, perc_completion_list, pl_exp_list, sigma_list, jtx_list, num_samples_list, weight_list = [], [], [], [], [], [], [], []
    loss_meter = AverageMeter()
    progress_bar = tqdm(validate_loader, desc="Validating", leave=True)

    # Dictionary to store the results by graph index and epoch
    detailed_metrics = []

    with torch.no_grad():
        for data in progress_bar:
            data = data.to(device)
            gnn_prediction, final_prediction, weight = model(data)

            if test_loader:
                predicted_coords = convert_output_eval(final_prediction, data, 'prediction', device)
                predictions.append(predicted_coords.cpu().numpy())

                actual_coords = convert_output_eval(data.y, data, 'target', device)
                actuals.append(actual_coords.cpu().numpy())

                # print("predictions: ", predictions)
                # print("actuals: ", actuals)

                loss = criterion(predicted_coords, actual_coords)

                perc_completion_list.append(data.perc_completion.cpu().numpy())
                pl_exp_list.append(data.pl_exp.cpu().numpy())
                sigma_list.append(data.sigma.cpu().numpy())
                jtx_list.append(data.jtx.cpu().numpy())
                num_samples_list.append(data.num_samples.cpu().numpy())
                weight_list.append(weight.cpu().numpy())
            else:
                loss = criterion(final_prediction, data.y)

                # Dictionary to store individual graph details
                graph_details = {}

                # Calculate RMSE for each graph in the batch
                for idx in range(data.num_graphs):
                    prediction = convert_output_eval(final_prediction[idx], data[idx], 'prediction', device)
                    actual = convert_output_eval(data.y[idx], data[idx], 'target', device)

                    mse = mean_squared_error(actual.cpu().numpy(), prediction.cpu().numpy())
                    rmse = math.sqrt(mse)
                    perc_completion = data.perc_completion[idx].item()
                    pl_exp = data.pl_exp[idx].item()
                    sigma = data.sigma[idx].item()
                    jtx = data.jtx[idx].item()
                    num_samples = data.num_samples[idx].item()

                    # Storing the metrics in the dictionary with graph id as key
                    graph_details[idx] = {'rmse': rmse, 'perc_completion': perc_completion, 'pl_exp': pl_exp, 'sigma': sigma, 'jtx': jtx, 'num_samples': num_samples}

                # Append to the detailed metrics dict
                detailed_metrics.append(graph_details)

            if not global_config.args.inference:
                # Update AverageMeter with the current RMSE and number of graphs
                loss_meter.update(loss.item(), data.num_graphs)

                # Update the progress bar with the running average RMSE
                progress_bar.set_postfix({"Validation Loss (MSE)": loss_meter.avg})

    if test_loader:
        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)
        perc_completion_list = np.concatenate(perc_completion_list)
        pl_exp_list = np.concatenate(pl_exp_list)
        sigma_list = np.concatenate(sigma_list)
        jtx_list = np.concatenate(jtx_list)
        num_samples_list = np.concatenate(num_samples_list)
        weight_list = np.concatenate(weight_list)

        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        rmse = math.sqrt(mse)

        err_metrics = {
            'actuals': actuals,
            'predictions': predictions,
            'perc_completion': perc_completion_list,
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }
        return predictions, actuals, err_metrics, perc_completion_list, pl_exp_list, sigma_list, jtx_list, num_samples_list, weight_list

    return loss_meter.avg, detailed_metrics

