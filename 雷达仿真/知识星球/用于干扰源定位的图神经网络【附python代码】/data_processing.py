import os
import pickle
import hashlib
import json

import pandas as pd
import numpy as np
import random
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch.utils.data import Dataset
from typing import Tuple, List
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import logging
from utils import cartesian_to_polar
from custom_logging import setup_logging
from global_config import global_config

setup_logging()


class Instance:
    """
    A class representing a single instance of data with methods to manipulate jammed network features.

    Attributes:
        num_samples (int): The number of samples or nodes in the instance.
        node_positions (np.ndarray): Node positions in the specified format.
        node_positions_cart (np.ndarray): Cartesian coordinates of node positions.
        node_noise (np.ndarray): Noise levels associated with each node.
        pl_exp (float): Path loss exponent.
        sigma (float): Standard deviation of the noise.
        jammer_power (float): Power of the jamming device.
        jammer_position (np.ndarray): The position of the jammer.
        jammer_gain (float): Gain of the jamming device.
        dataset (str): Identifier for the dataset from which this instance is derived.
        jammed_at (int, optional): Index at which jamming occurs, applicable in dynamic scenarios.

    Methods:
        get_crop(start, end): Returns a cropped version of the instance from start index to end index.
        drop_node(drop_rate=0.2, min_nodes=3): Randomly drops nodes from the instance based on a specified drop rate.
        downsample(): Downsamples the node positions and node noise using a binning strategy.

    Args:
        row (dict): A dictionary containing all necessary data attributes to initialize an instance.
    """
    def __init__(self, row):
        # Initialize attributes from the pandas row and convert appropriate fields to numpy arrays only if not already arrays
        self.num_samples = row['num_samples']
        self.node_positions = row['node_positions'] if isinstance(row['node_positions'], np.ndarray) else np.array(row['node_positions'])
        self.node_positions_cart = row['node_positions_cart'] if isinstance(row['node_positions_cart'], np.ndarray) else np.array(row['node_positions_cart'])
        self.node_noise = row['node_noise'] if isinstance(row['node_noise'], np.ndarray) else np.array(row['node_noise'])
        self.pl_exp = row['pl_exp']
        self.sigma = row['sigma']
        self.jammer_power = row['jammer_power']
        self.jammer_position = row['jammer_position'] if isinstance(row['jammer_position'], np.ndarray) else np.array(row['jammer_position'])
        self.jammer_gain = row['jammer_gain']
        self.dataset = row['dataset']
        if global_config.args.dynamic:
            self.jammed_at = row['jammed_at']

    def get_crop(self, start, end):
        """
        Extracts a subsection of instance, returning a new instance object representing the specified range.

        This method crops the instance's data attributes such as node positions and noises from the 'start' index
        to the 'end' index, creating a new instance that reflects this subset. This is particularly useful in dynamic
        scenarios where the temporal aspect of data needs to be considered.

        Args:
            start (int): The starting index of the crop.
            end (int): The ending index of the crop, non-inclusive.

        Returns:
            Instance: A new instance object containing only the data within the specified range.
        """
        # Base dictionary for the cropped instance
        cropped_data = {
            'num_samples': end - start,
            'node_positions': self.node_positions[start:end],
            'node_positions_cart': self.node_positions_cart[start:end],
            'node_noise': self.node_noise[start:end],
            'pl_exp': self.pl_exp,
            'sigma': self.sigma,
            'jammer_power': self.jammer_power,
            'jammer_position': self.jammer_position,
            'jammer_gain': self.jammer_gain,
            'dataset': self.dataset
        }

        # Add 'jammed_at' only if dynamic mode is enabled
        if global_config.args.dynamic:
            cropped_data['jammed_at'] = self.jammed_at  # Jammed index remains the same

        # Create and return the new instance
        return Instance(cropped_data)

    def drop_node(self, drop_rate=0.2, min_nodes=3):
        """
        Apply NodeDrop augmentation: randomly drops nodes from the instance to simulate data.

        Args:
            drop_rate (float): The probability of dropping any single node, thus 1 - drop_rate is the
                               probability of keeping a node. Defaults to 0.2.
            min_nodes (int): The minimum number of nodes that must remain after dropping. Defaults to 3.
        """
        # Get the number of nodes
        num_nodes = len(self.node_positions)

        # Generate a binary mask for dropping nodes
        # 1 = keep the node, 0 = drop the node
        mask = np.random.binomial(1, 1 - drop_rate, size=num_nodes)

        # Ensure that at least `min_nodes` are not dropped
        while sum(mask) < min_nodes:
            mask[np.random.choice(num_nodes, min_nodes - sum(mask), replace=False)] = 1

        # Apply the mask to each feature array
        self.node_positions = self.node_positions[mask == 1]
        self.node_positions_cart = self.node_positions_cart[mask == 1]
        self.node_noise = self.node_noise[mask == 1]

        # Update the number of samples after dropping nodes
        self.num_samples = len(self.node_positions)


    def downsample(self):
        """
        Downsample the number of nodes in the instance by spatially binning node features.
        """
        node_df = pd.DataFrame({
            'r': self.node_positions[:, 0],
            'sin_theta': self.node_positions[:, 1],
            'cos_theta': self.node_positions[:, 2],
            'sin_phi': self.node_positions[:, 3],
            'cos_phi': self.node_positions[:, 4],
            'x': self.node_positions_cart[:, 0],
            'y': self.node_positions_cart[:, 1],
            'z': self.node_positions_cart[:, 2],
            'noise_level': self.node_noise
        })

        binned_nodes = bin_nodes(node_df, grid_meters=1)
        self.node_positions = binned_nodes[['r', 'sin_theta', 'cos_theta', 'sin_phi', 'cos_phi']].to_numpy()
        self.node_positions_cart = binned_nodes[['x', 'y', 'z']].to_numpy()
        self.node_noise = binned_nodes['noise_level'].to_numpy()


class TemporalGraphDataset(Dataset):
    """
    A dataset class for handling temporal graph data, specifically for dynamic scenarios.

    This class manages the expansion of samples into multiple temporal points if configured to do so,
    supporting both dynamic and static data.

    Attributes:
        data (pd.DataFrame): The raw data frame containing graph data.
        test (bool): Flag to indicate if the dataset is used for testing, which changes how samples are preprocessed.
        dynamic (bool): Flag to indicate if the dataset should handle dynamic sampling, where each sample
                        can represent a different time point in a sequence.
        discretization_coeff (float): Coefficient to determine the granularity of temporal expansion,
                                      where higher values lead to fewer, more significant steps.

    Methods:
        expand_samples(): Expands the raw data into multiple temporal samples based on the discretization coefficient.
        precompute_graph(instance): Processes an instance to create a precomputed graph representation.
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx, start_crop=0): Retrieves a graph at a specific index, handling dynamic cropping and feature engineering.

    Args:
        data (pd.DataFrame): DataFrame containing the initial data with necessary columns for processing.
        test (bool, optional): Specifies whether the dataset is being used for testing. Defaults to False.
        dynamic (bool, optional): Specifies whether the dataset should operate in a dynamic mode. Defaults to True.
        discretization_coeff (float, optional): Specifies the step size coefficient for temporal expansion. Defaults to 0.25.
    """
    def __init__(self, data, test=False, dynamic=True, discretization_coeff=0.25):
        self.data = data
        self.test = test  # for test set
        self.dynamic = dynamic
        self.discretization_coeff = discretization_coeff

        if self.test:
            # Precompute the graphs during dataset initialization for the test set
            self.samples = self.expand_samples()
            self.precomputed_graphs = [self.precompute_graph(instance) for instance in self.samples]
        else:
            self.samples = [Instance(row) for _, row in data.iterrows()]

    def expand_samples(self):
        """
        Expands each row in the dataset into multiple temporal samples based on the configuration of the dataset.
        This method is for dynamic scenarios where the temporal evolution of data points
        needs to be captured in different stages for analysis or training.

        For dynamic datasets, this method expands the samples from the point of an event (e.g., 'jammed_at') to
        the end of the available data, stepping through the data at intervals determined by the `discretization_coeff`.
        In static settings, each sample is represented fully by a single instance without expansion.

        Returns:
            List[Instance]: A list of expanded or transformed instances, each representing a different
                            temporal stage of the data based on the discretization coefficient or a full snapshot
                            of the data if not dynamic.
        """
        expanded_samples = []
        for _, row in self.data.iterrows():
            if global_config.args.inference:
                instance = Instance(row)
                instance.perc_completion = 1
                expanded_samples.append(instance)
            else:
                if global_config.args.dynamic:
                    lb_end = max(int(row['jammed_at']), min(global_config.args.max_nodes, len(row['node_positions'])))
                    ub_end = len(row['node_positions'])

                    # Define step size
                    if self.discretization_coeff == -1:
                        step_size = 1
                    elif isinstance(self.discretization_coeff, float):
                        step_size = max(1, int(self.discretization_coeff * (ub_end - lb_end)))
                    else:
                        raise ValueError("Invalid discretization coefficient type")

                    # Generate instances for various end points with the step size
                    for i in range(lb_end, ub_end + 1, step_size):
                        instance = Instance(row).get_crop(0, i)
                        instance.perc_completion = i/ub_end
                        expanded_samples.append(instance)
                else:
                    instance = Instance(row)
                    instance.perc_completion = 1
                    expanded_samples.append(instance)
        return expanded_samples


    def precompute_graph(self, instance):
        """
        Processes an individual data instance to create a graph representation.
        This method converts the instance data into a PyTorch Geometric graph object, optionally applying downsampling
        and feature engineering.

        Args:
            instance (Instance): An instance of the data which contains features and target values that need to be
                                 transformed into a graph data structure.

        Returns:
            torch_geometric.data.Data: A graph data object containing nodes with engineered features and constrcuted edges.
        """
        if global_config.args.downsample:
            instance.downsample()
        graph = create_torch_geo_data(instance)
        graph = engineer_node_features(graph)
        return graph

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The total number of samples available in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx, start_crop=0):
        """
        Retrieves a graph from the dataset at a specified index, processing it based on the dataset's configuration.

        For test datasets, this method returns precomputed graph objects.
        For non-test datasets, it dynamically selects and modifies a portion of the data instance based on the dataset's
        configuration settings, such as dynamic adjustment, downsampling, and node dropping, before converting it into a graph format.

        Args:
            idx (int): Index of the sample to retrieve.
            start_crop (int, optional): Starting index used to crop the data in a dynamic scenario. Defaults to 0.

        Returns:
            torch_geometric.data.Data: Graph data object.
        """
        if self.test:
            # Return the precomputed graph for test set
            return self.precomputed_graphs[idx]

        # For non-test set, perform random cropping
        instance = self.samples[idx]
        if global_config.args.dynamic:
            # Check if jammed_at is not NaN and set the lower bound for random selection
            if np.isnan(instance.jammed_at):
                raise ValueError("No jammed instance")
            lb_end = max(int(instance.jammed_at), min(global_config.args.max_nodes, len(instance.node_positions)))
            ub_end = len(instance.node_positions)
            end = random.randint(lb_end, ub_end)
            instance = instance.get_crop(start_crop, end)
            instance.perc_completion = end / ub_end
        else:
            instance.perc_completion = 1.0

        if global_config.args.downsample:
            instance.downsample()

        if 'drop_node' in global_config.args.aug:
            instance.drop_node()

        # Create and engineer the graph on the fly for training
        graph = create_torch_geo_data(instance)
        graph = engineer_node_features(graph)

        return graph

def angle_to_cyclical(positions):
    """
    Convert a list of positions from spherical to trigonometric spherical representation.

    Args:
        positions (list): List of polar coordinates [r, theta, phi] for each point.
                          r is the radial distance,
                          theta is the polar angle from the positive z-axis (colatitude),
                          phi is the azimuthal angle in the xy-plane from the positive x-axis.

    Returns:
        list: List of cyclical coordinates [r, sin(theta), cos(theta), sin(phi), cos(phi)] for each point.
    """
    transformed_positions = []
    if global_config.args.three_dim:
        for position in positions:
            r, theta, phi = position
            sin_theta = np.sin(theta)  # Sine of the polar angle
            cos_theta = np.cos(theta)  # Cosine of the polar angle
            sin_phi = np.sin(phi)  # Sine of the azimuthal angle
            cos_phi = np.cos(phi)  # Cosine of the azimuthal angle
            transformed_positions.append([r, sin_theta, cos_theta, sin_phi, cos_phi])
    else:
        for position in positions:
            r, theta = position
            sin_theta = np.sin(theta)  # Sine of the azimuthal angle
            cos_theta = np.cos(theta)  # Cosine of the azimuthal angle
            transformed_positions.append([r, sin_theta, cos_theta])

    return transformed_positions


def apply_min_max_normalization_instance(instance):
    """
    Applies min-max normalization to the noise and Cartesian node position attributes of a given instance.

    Args:
        instance (Instance): The instance object containing node noise and node positions data to be normalized.
    """

    # Normalize Noise values to range [0, 1]
    instance.node_noise_original = instance.node_noise.copy()
    min_noise = np.min(instance.node_noise)
    max_noise = np.max(instance.node_noise)
    range_noise = max_noise - min_noise if max_noise != min_noise else 1
    normalized_noise = (instance.node_noise - min_noise) / range_noise
    instance.node_noise = normalized_noise

    # Normalize node positions cartesian for weights to range [0, 1]
    instance.node_positions_cart_original = instance.node_positions_cart.copy()
    min_coords = np.min(instance.node_positions_cart, axis=0)
    max_coords = np.max(instance.node_positions_cart, axis=0)
    range_coords = np.where(max_coords - min_coords == 0, 1, max_coords - min_coords)
    normalized_positions = (instance.node_positions_cart - min_coords) / range_coords
    instance.node_positions_cart = normalized_positions

def apply_unit_sphere_normalization(instance):
    """
    Apply unit sphere normalization to position data radius.

    Parameters:
    instance: An object with attributes 'node_positions' and 'jammer_position',
              each an array of positions where the first index is the radius.

    Modifies:
    instance.node_positions: Normalized positions.
    instance.jammer_position: Normalized jammer position.
    instance.max_radius: Maximum radius used for normalization.
    """

    # Extract the radius component from each position
    # radius is at index 0 of each sub-array in node_positions
    radii = instance.node_positions[:, 0]  # Extracts the first column from the positions array

    # Calculate the maximum radius from the radii
    max_radius = np.max(radii)

    # Normalize only the radius component of the positions
    normalized_positions = instance.node_positions.copy()  # Create a copy to avoid modifying the original data
    normalized_positions[:, 0] /= max_radius  # Normalize only the radius component

    # normalize jammer pos
    normalized_jammer_position = instance.jammer_position.copy()
    normalized_jammer_position[0][0] /= max_radius  # Normalize only the radius component of the jammer position

    # Update the instance variables
    instance.jammer_position = normalized_jammer_position

    # Update the instance variables
    instance.node_positions = normalized_positions
    instance.max_radius = max_radius


def convert_data_type(data, load_saved_data):
    """
    Converts data types of specified features in a dataset.

    Args:
        data (pd.DataFrame): The dataset containing features that may need type conversion.
        load_saved_data (bool): Flag indicating whether to load pre-converted data or apply conversions
                                dynamically based on current settings.
    """
    if load_saved_data:
        if global_config.args.dynamic:
            dataset_features = global_config.args.required_features + ['jammer_position', 'jammed_at', 'jammer_power',  'num_samples',  'sigma', 'jammer_power']
        else:
            dataset_features = global_config.args.required_features + ['jammer_position', 'jammer_power', 'num_samples', 'sigma', 'jammer_power']
    else:
        # Convert from str to required data type for specified features
        dataset_features = global_config.args.required_features + ['jammer_position']
    # Apply conversion to each feature directly
    for feature in dataset_features:
        data[feature] = data[feature].apply(lambda x: safe_convert_list(x, feature))


def calculate_noise_statistics(graph, noise_stats_to_compute):
    """
    Calculates noise-related statistics and spatial features for a given graph.
    This function is tailored to extract noise statistics and other metrics such as maximum noise, median noise,
    noise differential, and weighted centroid localization.

    Args:
        graph (list): Graph object.
        noise_stats_to_compute (list): List of statistical measures to compute.

    Returns:
        dict: A dictionary containing calculated noise statistics and spatial features. The contents of the dictionary
              depend on whether the graph is being treated as 2D or 3D.
    """
    subgraph = graph[0]
    edge_index = subgraph.edge_index
    if global_config.args.three_dim:
        node_positions = subgraph.x[:, :5]  # radius, sin(theta), cos(theta), sin(az), cos(az)
        node_noises = subgraph.x[:, 5]
    else:
        node_positions = subgraph.x[:, :3]  # radius, sin(theta), cos(theta)
        node_noises = subgraph.x[:, 3]

    # Create an adjacency matrix from edge_index and include self-loops
    num_nodes = node_noises.size(0)
    adjacency = torch.zeros(num_nodes, num_nodes, device=node_noises.device)
    adjacency[edge_index[0], edge_index[1]] = 1
    torch.diagonal(adjacency).fill_(1)  # Add self-loops

    # Calculate the sum and count of neighbor noises
    neighbor_sum = torch.mm(adjacency, node_noises.unsqueeze(1)).squeeze()
    neighbor_count = adjacency.sum(1)

    # Avoid division by zero for mean calculation
    neighbor_count = torch.where(neighbor_count == 0, torch.ones_like(neighbor_count), neighbor_count)
    mean_neighbor_noise = neighbor_sum / neighbor_count

    # Range: max - min for each node's neighbors
    expanded_noises = node_noises.unsqueeze(0).repeat(num_nodes, 1)
    max_noise = torch.where(adjacency == 1, expanded_noises, torch.full_like(expanded_noises, float('-inf'))).max(1).values

    # Median noise calculation
    median_noise = torch.full_like(mean_neighbor_noise, float('nan'))  # Initial fill
    for i in range(num_nodes):
        neighbors = node_noises[adjacency[i] == 1]
        if neighbors.numel() > 0:
            median_noise[i] = neighbors.median()

    # Noise Differential
    noise_differential = node_noises - mean_neighbor_noise

    # Convert from polar to Cartesian coordinates
    r = node_positions[:, 0]  # Radius
    sin_theta = node_positions[:, 1]  # sin(theta)
    cos_theta = node_positions[:, 2]  # cos(theta)
    if global_config.args.three_dim:
        sin_az = node_positions[:, 3]  # sin(azimuth)
        cos_az = node_positions[:, 4]  # cos(azimuth)
        # Compute x, y, z in Cartesian coordinates for 3D
        x = r * cos_theta * cos_az
        y = r * sin_theta * cos_az
        z = r * sin_az
        node_positions_cart = torch.stack((x, y, z), dim=1)  # 3D Cartesian coordinates

        vector_from_previous_position = torch.diff(node_positions_cart, dim=0, prepend=torch.zeros(1, node_positions_cart.size(1)))
        vector_x = vector_from_previous_position[:, 0]
        vector_y = vector_from_previous_position[:, 1]
        vector_z = vector_from_previous_position[:, 2]

        # Calculate rate of change of signal strength (first derivative)
        rate_of_change_signal_strength = torch.diff(node_noises, prepend=torch.tensor([node_noises[0]]))

    else:
        # Compute x, y in Cartesian coordinates for 2D
        x = r * cos_theta
        y = r * sin_theta
        node_positions_cart = torch.stack((x, y), dim=1)  # 2D Cartesian coordinates

    # Weighted Centroid Localization (WCL) calculation in Cartesian space
    weighted_centroid_radius = torch.zeros(num_nodes, device=node_positions.device)
    weighted_centroid_sin_theta = torch.zeros(num_nodes, device=node_positions.device)
    weighted_centroid_cos_theta = torch.zeros(num_nodes, device=node_positions.device)
    weighted_centroid_positions = torch.zeros_like(node_positions_cart)

    if global_config.args.three_dim:
        weighted_centroid_sin_az = torch.zeros(num_nodes, device=node_positions.device)
        weighted_centroid_cos_az = torch.zeros(num_nodes, device=node_positions.device)

    for i in range(num_nodes):
        weights = torch.pow(10, node_noises[adjacency[i] == 1] / 10)
        valid_neighbor_positions = node_positions_cart[adjacency[i] == 1]

        if weights.sum() > 0:
            centroid_cartesian = (weights.unsqueeze(1) * valid_neighbor_positions).sum(0) / weights.sum()
            radius = torch.norm(centroid_cartesian, p=2)
            sin_theta = centroid_cartesian[1] / radius if radius != 0 else 0
            cos_theta = centroid_cartesian[0] / radius if radius != 0 else 0

            weighted_centroid_radius[i] = radius
            weighted_centroid_sin_theta[i] = sin_theta
            weighted_centroid_cos_theta[i] = cos_theta
            weighted_centroid_positions[i] = centroid_cartesian

            if global_config.args.three_dim:
                sin_az = centroid_cartesian[2] / radius if radius != 0 else 0
                cos_az = torch.sqrt(centroid_cartesian[0] ** 2 + centroid_cartesian[1] ** 2) / radius if radius != 0 else 1
                weighted_centroid_sin_az[i] = sin_az
                weighted_centroid_cos_az[i] = cos_az


    # Distance from Weighted Centroid
    distances_to_wcl = torch.norm(node_positions_cart - weighted_centroid_positions, dim=1)

    # Base dictionary with entries that are common to both 2D and 3D cases
    noise_stats = {
        'max_noise': max_noise,
        'median_noise': median_noise,
        'noise_differential': noise_differential,
        'weighted_centroid_radius': weighted_centroid_radius,
        'weighted_centroid_sin_theta': weighted_centroid_sin_theta,
        'weighted_centroid_cos_theta': weighted_centroid_cos_theta,
        'dist_to_wcl': distances_to_wcl
    }
    # Conditionally add 3D specific elements
    if global_config.args.three_dim:
        noise_stats.update({
            'weighted_centroid_sin_az': weighted_centroid_sin_az,
            'weighted_centroid_cos_az': weighted_centroid_cos_az,
            'vector_x': vector_x,
            'vector_y': vector_y,
            'vector_z': vector_z,
            'rate_of_change_signal_strength': rate_of_change_signal_strength
        })

    return noise_stats

def engineer_node_features(graph):
    """
    Calculate new node features based on raw data.

    Args:
        graph (torch_geometric.data.Data): A graph data object containing raw node features.

    Returns:
        torch_geometric.data.Data: The graph data object with augmented node features.
    """
    if graph.x.size(0) == 0:
        raise ValueError("Empty graph encountered")

    new_features = []

    # Extract components
    r = graph.x[:, 0]  # Radii
    sin_theta = graph.x[:, 1]  # Sin of angles
    cos_theta = graph.x[:, 2]  # Cos of angles
    if global_config.args.three_dim:
        sin_az = graph.x[:, 3]  # Sin of azimuth
        cos_az = graph.x[:, 4]  # Cos of azimuth
        # Convert to 3D Cartesian coordinates
        x = r * cos_theta * cos_az
        y = r * sin_theta * cos_az
        z = r * sin_az
        cartesian_coords = torch.stack((x, y, z), dim=1)
    else:
        # Convert to 2D Cartesian coordinates
        x = r * cos_theta
        y = r * sin_theta
        cartesian_coords = torch.stack((x, y), dim=1)

    # Calculate centroid
    centroid = torch.mean(cartesian_coords, dim=0)

    if 'dist_to_centroid' in global_config.args.additional_features:
        distances = torch.norm(cartesian_coords - centroid, dim=1, keepdim=True)
        new_features.append(distances)

    # Graph-based noise stats
    graph_stats = [
        'median_noise', 'max_noise', 'weighted_centroid_radius',
        'weighted_centroid_sin_theta', 'weighted_centroid_cos_theta',
        'noise_differential','dist_to_wcl', 'weighted_centroid_sin_az', 'weighted_centroid_cos_az',
        'vector_x', 'vector_y', 'vector_z', 'rate_of_change_signal_strength'
    ]

    noise_stats_to_compute = [stat for stat in graph_stats if stat in global_config.args.additional_features]
    if noise_stats_to_compute:
        noise_stats = calculate_noise_statistics([graph], noise_stats_to_compute)

        # Add calculated statistics directly to the features list
        for stat in noise_stats_to_compute:
            if stat in noise_stats:
                new_features.append(noise_stats[stat].unsqueeze(1))

    if new_features:
        try:
            new_features_tensor = torch.cat(new_features, dim=1)
            graph.x = torch.cat((graph.x, new_features_tensor), dim=1)
        except RuntimeError as e:
            raise e

    return graph


def convert_to_polar(data):
    """
    Converts Cartesian coordinates of node and jammer positions to polar/spherical coordinates.

    Args:
        data (pd.DataFrame): The dataset containing 'node_positions' and 'jammer_position' in Cartesian coordinates.
    """
    data['node_positions_cart'] = data['node_positions'].copy()
    data['node_positions'] = data['node_positions'].apply(lambda x: angle_to_cyclical(cartesian_to_polar(x)))
    if global_config.args.dynamic:
       data['jammer_position'] = data['jammer_position'].apply(lambda x: [x])
       data['jammer_position'] = data['jammer_position'].apply(lambda x: angle_to_cyclical(cartesian_to_polar(x)))
    else:
       data['jammer_position'] = data['jammer_position'].apply(lambda x: angle_to_cyclical(cartesian_to_polar(x)))


def convert_output_eval(output, data_batch, data_type, device):
    """
    Convert and evaluate the model output or target coordinates by reversing the preprocessing steps:
    normalization and converting polar/spherical coordinates to Cartesian coordinates.

    Args:
        output (torch.Tensor): The model output tensor or target tensor.
        data_batch (dict): Dictionary containing data batch with necessary metadata.
        data_type (str): The type of data, either 'prediction' or 'target'.
        device (torch.device): The device on which the computation is performed.

    Returns:
        torch.Tensor: The converted Cartesian coordinates after reversing preprocessing steps.
    """
    # This function is called on both prediction and actual jammer pos
    # output = output.to(device)  # Ensure the tensor is on the right device
    #
    # # Ensure output always has at least two dimensions [batch_size, features]
    # if output.ndim == 1:
    #     output = output.unsqueeze(0)  # Add batch dimension if missing
    #
    # # Step 1: Reverse unit sphere normalization
    # max_radius = data_batch['max_radius'].to(device)
    # print("output: ", output)
    # print("max_radius: ", max_radius)
    # if max_radius.ndim == 0:
    #     max_radius = max_radius.unsqueeze(0)  # Ensure max_radius is at least 1D for broadcasting
    #
    # # Apply normalization reversal safely using broadcasting
    # output[:, 0] = output[:, 0] * max_radius
    #########################################
    output = output.to(device)  # Ensure tensor is on the device

    # print("output 1: ", output)


    if output.ndim == 1:
        output = output.unsqueeze(0)  # Add batch dimension if missing

    max_radius = data_batch['max_radius'].to(device)
    # print("max_radius 1: ", max_radius)
    if max_radius.ndim == 0 or (max_radius.ndim == 1 and max_radius.shape[0] == 1):
        max_radius = max_radius.expand(output.shape[0])  # Expand max_radius to match output's batch size

    # print("output 2: ", output)
    # print("max_radius 2: ", max_radius)

    # Apply normalization reversal
    output[:, 0] *= max_radius  # Safe broadcasting
    ########################################3

    # Step 2: Convert from polar (radius, sin(theta), cos(theta)) to Cartesian coordinates
    radius = output[:, 0]
    sin_theta = output[:, 1]
    cos_theta = output[:, 2]

    if global_config.args.three_dim:
        # Additional columns for 3D coordinates: sin_phi and cos_phi
        sin_phi = output[:, 3]
        cos_phi = output[:, 4]

        # Calculate theta from sin_theta and cos_theta for clarity in following the formulas
        theta = torch.atan2(sin_theta, cos_theta)

        # Correct conversion for 3D
        x = radius * torch.sin(theta) * cos_phi  # Note the correction in angle usage
        y = radius * torch.sin(theta) * sin_phi
        z = radius * cos_theta  # cos(theta) is directly used

        # Stack x, y, and z coordinates horizontally to form the Cartesian coordinate triplets
        cartesian_coords = torch.stack((x, y, z), dim=1)

    else:
        # Calculate 2D Cartesian coordinates
        x = radius * cos_theta
        y = radius * sin_theta

        # Stack x and y coordinates horizontally to form the Cartesian coordinate pairs
        cartesian_coords = torch.stack((x, y), dim=1)

    return cartesian_coords.clone().detach()


def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def filter_data(data, base_shapes, experiments_path):
    """
    Filter the data based on the dataset column and save filtered data to disk.

    Args:
        data (pd.DataFrame): The input data.
        base_shapes (list): List of base shapes to filter by.
        experiments_path (str): Path to save the filtered data.

    Returns:
        dict: A dictionary of filtered DataFrames.
    """
    filtered_data = {}

    for base_shape in base_shapes:
        # Exact matching for base shape and base shape all jammed
        exact_base = data['dataset'] == base_shape
        exact_base_all_jammed = data['dataset'] == f"{base_shape}_all_jammed"
        filtered_base = data[exact_base | exact_base_all_jammed]

        if not filtered_base.empty:
            filtered_data[f'{base_shape}'] = filtered_base
            filtered_base.to_pickle(os.path.join(experiments_path, f'{base_shape}_dataset.pkl'))

        # Exact matching for base shape jammer outside region and base shape all jammed jammer outside region
        exact_jammer_outside = data['dataset'] == f"{base_shape}_jammer_outside_region"
        exact_jammer_outside_all_jammed = data['dataset'] == f"{base_shape}_all_jammed_jammer_outside_region"
        filtered_jammer_outside = data[exact_jammer_outside | exact_jammer_outside_all_jammed]

        if not filtered_jammer_outside.empty:
            filtered_data[f'{base_shape}_jammer_outside'] = filtered_jammer_outside
            filtered_jammer_outside.to_pickle(os.path.join(experiments_path, f'{base_shape}_jammer_outside_dataset.pkl'))

    return filtered_data

def split_datasets(data, experiments_path):
    """
    Preprocess and split the dataset into train, validation, and test sets with stratification based on jammer distance categories.

    Args:
        data (pd.DataFrame): The input data.
        experiments_path (str): Path to save the split datasets.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test datasets.
    """
    if global_config.args.dynamic:
        # Stratify on the distance category
        train_idx, val_test_idx = train_test_split(
            data.index,  # Use DataFrame index to ensure correct referencing
            test_size=0.3,
            random_state=100
        )

        # Split the test set into validation and test
        val_idx, test_idx = train_test_split(
            val_test_idx,
            test_size=0.6667,  # 20% test / 30% total = 0.6667
            random_state=100
        )
    else:
        # Work on a copy to avoid SettingWithCopyWarning when modifying data
        data = data.copy()

        # Calculate centroid for each entry by processing the node_positions list of lists
        data.loc[:, 'centroid_x'] = data['node_positions'].apply(lambda positions: np.mean([pos[0] for pos in positions]))
        data.loc[:, 'centroid_y'] = data['node_positions'].apply(lambda positions: np.mean([pos[1] for pos in positions]))

        # Extract x and y coordinates from jammer_position
        data.loc[:, 'jammer_x'] = data['jammer_position'].apply(lambda pos: pos[0][0])
        data.loc[:, 'jammer_y'] = data['jammer_position'].apply(lambda pos: pos[0][1])

        # Calculate distance between jammer and centroid
        data.loc[:, 'jammer_distance'] = data.apply(
            lambda row: calculate_distance(
                row['jammer_x'], row['jammer_y'], row['centroid_x'], row['centroid_y']
            ),
            axis=1
        )

        # Create dynamic bins based on min and max jammer distances
        num_bins = 7
        min_distance = data['jammer_distance'].min()
        max_distance = data['jammer_distance'].max()
        bin_edges = np.linspace(min_distance, max_distance, num=num_bins + 1)  # Create bin edges
        bin_labels = [f'{int(bin_edges[i])}-{int(bin_edges[i + 1])}m' for i in range(num_bins)]  # Create bin labels

        # Bin distances into categories
        data['distance_category'] = pd.cut(data['jammer_distance'], bins=bin_edges, labels=bin_labels, include_lowest=True)

        # Stratify on the distance category
        train_idx, val_test_idx = train_test_split(
            data.index,  # Use DataFrame index to ensure correct referencing
            test_size=0.3,
            stratify=data['distance_category'],  # Stratify on distance categories
            random_state=100
        )

        # Split the test set into validation and test
        val_idx, test_idx = train_test_split(
            val_test_idx,
            test_size=0.6667,  # 20% test / 30% total = 0.6667
            stratify=data.loc[val_test_idx, 'distance_category'],  # Stratify on distance categories
            random_state=100
        )

    # Convert indices back to DataFrame subsets
    train_df = data.loc[train_idx]
    val_df = data.loc[val_idx]
    test_df = data.loc[test_idx]

    return train_df, val_df, test_df

def process_data(data, experiments_path):
    """
    Process the data by filtering and splitting it.

    Args:
        data (pd.DataFrame): The input data.
        experiments_path (str): Path to save the processed data.

    Returns:
        dict: A dictionary of filtered and split datasets.
    """
    if global_config.args.dynamic:
        split_datasets_dict = {}
        train_df, val_df, test_df = split_datasets(data, experiments_path)
        split_datasets_dict['dynamic'] = {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
    else:
        # Define base shapes for filtering
        base_shapes = ['circle', 'triangle', 'rectangle', 'random']

        # Step 1: Filter the data
        filtered_data = filter_data(data, base_shapes, experiments_path)

        # Step 2: Split each filtered dataset
        split_datasets_dict = {}
        for key, filtered_df in filtered_data.items():
            train_df, val_df, test_df = split_datasets(filtered_df, experiments_path)
            split_datasets_dict[key] = {
                'train': train_df,
                'validation': val_df,
                'test': test_df
            }

    return split_datasets_dict


def save_datasets(combined_train_df, combined_val_df, combined_test_df, experiments_path, data_class):
    """
    Process the combined train, validation, and test data, and save them to disk as .pkl files.
    If the files already exist, they will not be saved again.

    Args:
        combined_train_df (pd.DataFrame): DataFrame containing combined training data.
        combined_val_df (pd.DataFrame): DataFrame containing combined validation data.
        combined_test_df (pd.DataFrame): DataFrame containing combined test data.
        experiments_path (str): The path where the processed data will be saved.
        data_class (str): The class or name of the dataset to be used in the file names.
    """
    # Define file paths
    train_file_path = os.path.join(experiments_path, f'{data_class}_train_dataset.pkl')
    val_file_path = os.path.join(experiments_path, f'{data_class}_val_dataset.pkl')
    test_file_path = os.path.join(experiments_path, f'{data_class}_test_dataset.pkl')

    # Check if files already exist
    if not os.path.exists(train_file_path):
        logging.info("Saving training data")
        with open(train_file_path, 'wb') as f:
            pickle.dump(combined_train_df, f)

    if not os.path.exists(val_file_path):
        logging.info("Saving validation data")
        with open(val_file_path, 'wb') as f:
            pickle.dump(combined_val_df, f)

    if not os.path.exists(test_file_path):
        logging.info("Saving test data")
        with open(test_file_path, 'wb') as f:
            pickle.dump(combined_test_df, f)


def bin_nodes(node_df, grid_meters=1):
    """
    Bin nodes by averaging positions and noise levels within each grid cell for both polar/spherical and Cartesian coordinates,
    and merge results into a single DataFrame.

    Args:
        node_df (pd.DataFrame): DataFrame containing node data in both polar and Cartesian coordinates.
        grid_meters (int): The size of each grid cell for binning.

    Returns:
        pd.DataFrame: Binned nodes with averaged positions and other features for Cartesian and polar coordinates.
    """
    # Handle Cartesian coordinates
    node_df['x_bin'] = (node_df['x'] // grid_meters).astype(int)
    node_df['y_bin'] = (node_df['y'] // grid_meters).astype(int)
    node_df['z_bin'] = (node_df['z'] // grid_meters).astype(int)

    binned_cartesian = node_df.groupby(['x_bin', 'y_bin', 'z_bin']).agg({
        'x': 'mean',
        'y': 'mean',
        'z': 'mean',
        'noise_level': 'mean'
    }).reset_index()

    # Drop the bin columns as they are no longer needed
    binned_cartesian.drop(columns=['x_bin', 'y_bin', 'z_bin'], inplace=True)

    # Handle Polar coordinates by converting to Cartesian first
    node_df['x_polar'] = node_df['r'] * node_df['sin_theta'] * node_df['cos_phi']
    node_df['y_polar'] = node_df['r'] * node_df['sin_theta'] * node_df['sin_phi']
    node_df['z_polar'] = node_df['r'] * node_df['cos_theta']

    node_df['x_polar_bin'] = (node_df['x_polar'] // grid_meters).astype(int)
    node_df['y_polar_bin'] = (node_df['y_polar'] // grid_meters).astype(int)
    node_df['z_polar_bin'] = (node_df['z_polar'] // grid_meters).astype(int)

    binned_polar = node_df.groupby(['x_polar_bin', 'y_polar_bin', 'z_polar_bin']).agg({
        'x_polar': 'mean',
        'y_polar': 'mean',
        'z_polar': 'mean'
    }).reset_index()

    # Convert averaged Cartesian coordinates back to polar
    # Correct calculation of r, theta, and phi
    binned_polar['r'] = np.sqrt(binned_polar['x_polar'] ** 2 + binned_polar['y_polar'] ** 2 + binned_polar['z_polar'] ** 2)
    # binned_polar['theta'] = np.arccos(binned_polar['z_polar'] / binned_polar['r'])
    binned_polar['theta'] = np.arctan2(np.sqrt(binned_polar['x_polar'] ** 2 + binned_polar['y_polar'] ** 2), binned_polar['z_polar'])
    binned_polar['phi'] = np.arctan2(binned_polar['y_polar'], binned_polar['x_polar'])

    # calculate sin and cos of theta and phi if needed for further processing
    binned_polar['sin_theta'] = np.sin(binned_polar['theta'])
    binned_polar['cos_theta'] = np.cos(binned_polar['theta'])
    binned_polar['sin_phi'] = np.sin(binned_polar['phi'])
    binned_polar['cos_phi'] = np.cos(binned_polar['phi'])

    # Drop unnecessary columns
    binned_polar.drop(columns=['x_polar_bin', 'y_polar_bin', 'z_polar_bin', 'x_polar', 'y_polar', 'z_polar'], inplace=True)

    # Merge the two DataFrames
    binned = pd.concat([binned_cartesian, binned_polar], axis=1)

    # Drop rows with NaN values
    binned = binned.dropna()

    # Sort by noise_level and keep the top max_nodes
    binned = binned.sort_values(by='noise_level', ascending=False).head(global_config.args.max_nodes)

    # Ensure at least 3 nodes are returned
    if len(binned) < 3:
        return node_df  # Fall back to the original dataset
    else:
        return binned


def add_jammed_column(data, threshold=-55):
    """
    Adds a column 'jammed_at' to the dataset indicating the index of the sample where jamming is detected based on
    noise levels exceeding a specified threshold.

    Args:
        data (pd.DataFrame): The dataset containing 'node_noise' columns, where each entry is a list of noise measurements.
        threshold (int, optional): The noise level that must be exceeded to consider the signal as being jammed.
                                   Defaults to -55 dBm.

    Returns:
        pd.DataFrame: The modified DataFrame with an additional 'jammed_at' column indicating the index of exceeded threshold.
    """
    data['jammed_at'] = None
    for i, noise_list in enumerate(data['node_noise']):
        # Check if noise_list is a valid non-empty list
        if not isinstance(noise_list, list) or len(noise_list) == 0:
            raise ValueError(f"Invalid or empty node_noise list at row {i}")

        count = 0
        jammed_index = None  # Store the index of the noise > threshold

        for idx, noise in enumerate(noise_list):
            if noise > threshold:
                count += 1
                # Save the index of the third noise sample that exceeds the threshold
                if count == 3:
                    jammed_index = idx + 1
                    break

        # Save the index of the "jammed" sample or handle no jamming detected
        if jammed_index is not None:
            data.at[i, 'jammed_at'] = jammed_index
        else:
            raise ValueError(f"No sufficient jammed noise samples found for row {i}")

    return data


def load_data(data_class, experiments_path=None):
    """
    Loads or processes data for a specified class of experiments, handling both the inference and training scenarios.
    For inference, the function loads preprocessed datasets directly. For non-inference uses, it processes raw data,
    converts it, and optionally handles dynamic data aspects like jamming detection before splitting it into train,
    validation, and test sets.

    Args:
        data_class (str): A string identifier for the data class being loaded, which dictates which subset of data
                          is processed or loaded.
        experiments_path (str, optional): The base path where preprocessed data files are stored or will be saved.
                                          If not provided, the default directory set in global configuration is used.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing DataFrames for the train, validation,
                                                         and test datasets, respectively.
    """
    if global_config.args.inference:
        logging.info("Loading data for inference...")

        # Paths to the datasets
        test_path = os.path.join(experiments_path, f"{data_class}_test_dataset.pkl")

        # Check if all datasets exist before loading
        if os.path.exists(test_path):
            train_df = None
            val_df = None
            test_df = pd.read_pickle(test_path)
            logging.info("Loaded datasets")
        else:
            # Load the dataset from the pickle file
            with open(global_config.args.dataset_path, "rb") as f:
                data_list = []
                try:
                    # Read each dictionary entry in the list and add to data_list
                    while True:
                        data_list.append(pickle.load(f))
                except EOFError:
                    pass  # End of file reached

            # Convert the list of dictionaries to a DataFrame
            if isinstance(data_list[0], pd.DataFrame):
                data = data_list[0]
            else:
                data = pd.DataFrame(data_list)

            # If the data needs type conversion, which might not be the case in inference
            if not global_config.args.dynamic:
                convert_data_type(data, load_saved_data=False)

            # Check if there are specific preprocessing steps for dynamic data
            if global_config.args.dynamic:
                data = add_jammed_column(data, threshold=-55)

            # Reset index in preparation for further processing or splitting
            data.reset_index(drop=True, inplace=True)

            # Directly treat all the data as a test dataset since we're in inference mode
            train_df = None
            val_df = None
            test_df = data

        # Convert to polar coordinates (spherical)
        convert_to_polar(test_df)

        return train_df, val_df, test_df
    else:
        datasets = [global_config.args.dataset_path]
        for dataset in datasets:
            logging.info("Loading data...")

            # Load the interpolated dataset from the pickle file
            with open(dataset, "rb") as f:
                data_list = []
                try:
                    # Read each dictionary entry in the list and add to data_list
                    while True:
                        data_list.append(pickle.load(f))
                except EOFError:
                    pass  # End of file reached

            # Convert the list of dictionaries to a DataFrame
            if isinstance(data_list[0], pd.DataFrame):
                data = data_list[0]
            else:
                data = pd.DataFrame(data_list)

            if not global_config.args.dynamic:
                convert_data_type(data, load_saved_data=False)

            # Identify point at which noise floor exceeded threshold
            if global_config.args.dynamic:
                data = add_jammed_column(data, threshold=-55)

            # Create train test splits
            data.reset_index(inplace=True)
            split_datasets_dict = process_data(data, experiments_path)
            train_df = split_datasets_dict[data_class]['train']
            val_df = split_datasets_dict[data_class]['validation']
            test_df = split_datasets_dict[data_class]['test']

            # Process and save the combined data
            save_datasets(train_df, val_df, test_df, experiments_path, data_class)

            # Convert coordinates to polar (spherical)
            convert_to_polar(train_df)
            convert_to_polar(val_df)
            convert_to_polar(test_df)

        return train_df, val_df, test_df


def get_params_hash():
    """
    Generates an MD5 hash based on the current configuration parameters stored in `global_config.args`.

    Returns:
        str: An MD5 hash representing the serialized configuration parameters.
    """
    # Convert argparse.Namespace to a dictionary
    params_dict = vars(global_config.args)
    # Serialize the dictionary with sorted keys to ensure consistent order.
    params_str = json.dumps(params_dict, sort_keys=True)
    # Compute and return the MD5 hash of the serialized string.
    return hashlib.md5(params_str.encode()).hexdigest()


def create_data_loader(train_data, val_data, test_data, experiment_path):
    """
    Create data loaders using the TemporalGraphDataset instances for training, validation, and testing sets.
    Args:
        train_data (pd.DataFrame): DataFrame containing the training data.
        val_data (pd.DataFrame): DataFrame containing the validation data.
        test_data (pd.DataFrame): DataFrame containing the testing data.
    Returns:
        tuple: Three DataLoaders for the training, validation, and testing datasets.
    """
    deg_histogram = None

    # Generate a unique identifier for the current params
    params_hash = get_params_hash()
    cache_path = os.path.join(experiment_path, f"data_loader_{params_hash}.pkl")
    os.makedirs(experiment_path, exist_ok=True)

    if os.path.exists(cache_path):
        # Load cached data loaders
        with open(cache_path, 'rb') as f:
            train_loader, val_loader, test_loader = pickle.load(f)
        logging.info("Loaded cached data loaders...")
    else:
        # Create data loaders and save them if cache doesn't exist
        logging.info("Creating data loaders...")
        train_loader, val_loader, test_loader = generate_data_loaders(train_data, val_data, test_data)

        # # Save data loaders to cache
        # with open(cache_path, 'wb') as f:
        #     pickle.dump((train_loader, val_loader, test_loader), f)
        # logging.info("Saved data loaders")

    if global_config.args.model == 'PNA':
        deg_histogram = compute_degree_histogram(train_loader)

    if global_config.args.inference:
        return None, None, test_loader, None

    return train_loader, val_loader, test_loader, deg_histogram


def generate_data_loaders(train_data, val_data, test_data):
    """
    Creates data loaders for training, validation, and test datasets using the TemporalGraphDataset class.
    This function configures each data loader with specific settings for batch size, shuffling, memory optimization,
    and the number of worker processes based on the dataset type (training vs. validation/test) and global configurations.

    Args:
        train_data (pd.DataFrame): The training dataset.
        val_data (pd.DataFrame): The validation dataset.
        test_data (pd.DataFrame): The test dataset.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing DataLoader objects for the training,
                                                   validation, and test datasets respectively.
    """
    if global_config.args.inference:
        train_loader = None
        val_loader = None
        test_dataset = TemporalGraphDataset(test_data, test=True, discretization_coeff=global_config.args.test_discrite_coeff)
        test_loader = DataLoader(test_dataset, batch_size=global_config.args.test_batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)
    else:
        train_dataset = TemporalGraphDataset(train_data, test=False)
        train_loader = DataLoader(train_dataset, batch_size=global_config.args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=global_config.args.num_workers)

        val_dataset = TemporalGraphDataset(val_data, test=True, discretization_coeff=global_config.args.val_discrite_coeff)
        val_loader = DataLoader(val_dataset, batch_size=global_config.args.test_batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)

        test_dataset = TemporalGraphDataset(test_data, test=True, discretization_coeff=global_config.args.test_discrite_coeff)
        test_loader = DataLoader(test_dataset, batch_size=global_config.args.test_batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)

    return train_loader, val_loader, test_loader


def compute_degree_histogram(data_loader):
    """
    Computes the degree histogram for all graphs in a dataset loaded through a DataLoader. This histogram
    represents the frequency of each degree value across all nodes in the dataset. Used only for PNA model.

    Args:
        data_loader (DataLoader): A DataLoader object that loads the graphs from which the degrees are calculated.

    Returns:
        torch.Tensor: A tensor representing the degree histogram. Each index i in the tensor corresponds to the
                      number of nodes with degree i in the dataset.
    """
    max_degree = 0
    deg_histogram = None
    for data in data_loader:
        d = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long)
        batch_max_degree = d.max().item()
        max_degree = max(max_degree, batch_max_degree)
        if deg_histogram is None:
            deg_histogram = torch.bincount(d, minlength=max_degree + 1)
        else:
            if batch_max_degree > deg_histogram.numel() - 1:
                new_histogram = torch.zeros(batch_max_degree + 1, dtype=deg_histogram.dtype)
                new_histogram[:deg_histogram.numel()] = deg_histogram
                deg_histogram = new_histogram
            deg_histogram += torch.bincount(d, minlength=deg_histogram.numel())
    return deg_histogram


def safe_convert_list(row: str, data_type: str):
    """
    Safely convert a string representation of a list to an actual list,
    with type conversion tailored to specific data types including handling
    for 'states' which are extracted and stripped of surrounding quotes.

    Args:
        row (str): String representation of a list.
        data_type (str): The type of data to convert ('jammer_pos', 'drones_pos', 'node_noise', 'states').

    Returns:
        List: Converted list or an empty list if conversion fails.
    """
    try:
        if data_type == 'jammer_position':
            result = row.strip('[').strip(']').split(', ')
            return [[float(pos) for pos in result]]
        elif data_type == 'node_positions':
            result = row.strip('[').strip(']').split('], [')
            return [[float(num) for num in elem.split(', ')] for elem in result]
        elif data_type == 'node_noise':
            result = row.strip('[').strip(']').split(', ')
            return [float(noise) for noise in result]
        elif data_type == 'node_rssi':
            result = row.strip('[').strip(']').split(', ')
            return [float(rssi) for rssi in result]
        elif data_type == 'node_states':
            result = row.strip('[').strip(']').split(', ')
            return [int(state) for state in result]
        elif data_type == 'timestamps':
            result = row.strip('[').strip(']').split(', ')
            return [float(time) for time in result]
        elif data_type == 'angle_of_arrival':
            result = row.strip('[').strip(']').split(', ')
            return [float(aoa) for aoa in result]
        elif data_type == 'jammed_at':
            return int(row)
        elif data_type == 'jammer_power':
            return float(row)
        elif data_type == 'num_samples':
            return float(row)
        elif data_type == 'sigma':
            return float(row)
        elif data_type == 'jammer_power':
            return float(row)
        else:
            raise ValueError("Unknown data type")
    except (ValueError, SyntaxError, TypeError) as e:
        return []  # Return an empty list if there's an error


def plot_graph(positions, edge_index, node_features, edge_weights=None, jammer_positions=None, show_weights=False, perc_completion=None, id=None, jammer_power=None):
    """
    Plots a graph with optional annotations for nodes, edges, and additional elements.

    Args:
        positions (array-like or torch.Tensor): Coordinates for nodes in the graph.
        edge_index (array-like or torch.Tensor): Indices indicating the start and end nodes for each edge.
        node_features (array-like or torch.Tensor): An array or tensor containing features for each node, used to annotate
                                                    nodes in the plot.
        edge_weights (array-like or torch.Tensor, optional): Weights for each edge.
        jammer_positions (array-like or torch.Tensor, optional): Coordinates for jammers to be marked on the graph.
        show_weights (bool, optional): If True, edge weights are shown in the plot. Default is False.
        perc_completion (float, optional): Percentage of completion of the trajectory or process, shown in the plot title.
        id (int or str, optional): An identifier for the graph, used in the plot title.
        jammer_power (float, optional): Power of the jammer, not currently used in plotting but can be included for future use.
    """
    G = nx.Graph()

    # Check if positions are tensors and convert them
    if isinstance(positions, torch.Tensor):
        positions = positions.numpy()
    elif isinstance(positions, list) and isinstance(positions[0], torch.Tensor):
        positions = [pos.numpy() for pos in positions]

    # Ensure node_features is a numpy array for easier handling
    if isinstance(node_features, torch.Tensor):
        node_features = node_features.numpy()

    # Add nodes with features and positions
    for i, pos in enumerate(positions):
        G.add_node(i, pos=(pos[0], pos[1]), noise=node_features[i][2])

    # Convert edge_index to numpy array if it's a tensor
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()
    edge_index = edge_index.T  # Ensure edge_index is transposed if necessary

    # Position for drawing, ensuring positions are tuples
    pos = {i: (p[0], p[1]) for i, p in enumerate(positions)}

    # Add edges to the graph
    if edge_weights is not None:
        edge_weights = edge_weights.numpy() if isinstance(edge_weights, torch.Tensor) else edge_weights
        for idx, (start, end) in enumerate(edge_index):
            G.add_edge(start, end, weight=edge_weights[idx])
    else:
        for start, end in edge_index:
            G.add_edge(start, end)

    # Draw the graph nodes
    nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=20)  # Use a default color

    # Draw the last index node with a different color (e.g., red)
    last_node_index = len(positions) - 1
    nx.draw_networkx_nodes(G, pos, nodelist=[last_node_index], node_color='red', node_size=60)

    # Draw the edges
    if show_weights and edge_weights is not None:
        nx.draw_networkx_edges(G, pos, width=0.5)
    else:
        nx.draw_networkx_edges(G, pos)

    # Optionally draw jammer positions
    if jammer_positions is not None:
        if isinstance(jammer_positions, torch.Tensor):
            jammer_positions = jammer_positions.numpy()
        for jp in jammer_positions:
            plt.scatter(jp[0], jp[1], color='red', s=50, marker='x', label='Jammer')

    # Additional plot settings
    if perc_completion is not None:
        plt.title(f"Graph {id} - {perc_completion:.2f}% trajectory since start", fontsize=15)
    else:
        plt.title(f"Graph {id}", fontsize=15)

    plt.axis('off')
    plt.show()


def weighted_centroid_localization(drones_pos, drones_noise):
    """
    Calculates the weighted centroid of drone positions based on their respective noise levels. This method
    applies a logarithmic transformation to the noise levels to use as weights, reflecting the importance
    of each drone's position in determining the centroid.

    Args:
        drones_pos (np.array): A numpy array where each row represents the coordinates of a drone.
        drones_noise (list or np.array): A list or 1D numpy array of noise measurements for each drone, which
                                        influence the weighting of their positions.

    Returns:
        list: The coordinates of the weighted centroid, converted to a list format.
    """
    weights = np.array([10 ** (noise / 10) for noise in drones_noise])
    weighted_sum = np.dot(weights, drones_pos) / np.sum(weights)
    return weighted_sum.tolist()


def create_torch_geo_data(instance: Instance) -> Data:
    """
    Constructs a PyTorch Geometric Data object from an instance of the dataset. This function integrates multiple
    preprocessing steps to prepare graph data for use in graph-based machine learning models.

    The function performs weighted centroid localization for nodes, applies various normalizations to the data,
    calculates node features combining multiple attributes, and constructs a graph with edges based on nearest
    neighbors.

    Args:
        instance (Instance): An instance object containing attributes such as node positions, noise levels, and
                             other related features necessary for constructing the graph.

    Returns:
        Data: A PyTorch Geometric Data object that includes node features, edge indices, edge weights, target
              variables, and additional metadata relevant for the model training or inference.
    """
    # Calculate Weighted Centroid Localization
    centroid_cartesian = weighted_centroid_localization(instance.node_positions_cart, instance.node_noise)

    # Wrap the centroid in a list to match the expected input format for cartesian_to_polar
    centroid_polar = cartesian_to_polar([centroid_cartesian])

    # Convert the polar coordinates to cyclical coordinates
    centroid_spherical = angle_to_cyclical(centroid_polar)

    # Append centroid to node positions
    instance.node_positions_cart = np.vstack([instance.node_positions_cart, centroid_cartesian])
    instance.node_positions = np.vstack([instance.node_positions, centroid_spherical])

    # Set noise for WCL node
    weighted_noise = weighted_centroid_localization(instance.node_noise, instance.node_noise)
    instance.node_noise = np.append(instance.node_noise, weighted_noise)  # Append high noise value for the centroid node

    # Normalize data instance data
    apply_min_max_normalization_instance(instance) # For noise floor and cartesian coords
    apply_unit_sphere_normalization(instance) # For polar coord

    # Create node features
    node_features = np.concatenate([
        instance.node_positions,
        instance.node_noise[:, None],
        instance.node_positions_cart
    ], axis=1)

    # Convert to 2D tensor
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32)


    positions = instance.node_positions_cart
    num_samples = positions.shape[0]
    if global_config.args.num_neighbors == 'fc':
        num_edges = positions.shape[0] - 1
    else:
        num_edges = global_config.args.num_neighbors

    # Initialize NearestNeighbors with more neighbors than needed to calculate full connections later
    nbrs = NearestNeighbors(n_neighbors=num_edges + 1, algorithm='auto').fit(positions)
    distances, indices = nbrs.kneighbors(positions)

    edge_index, edge_weight = [], []
    alpha = 1.0  # decay parameter

    # Add edges based on KNN for all nodes except the last
    for i in range(num_samples - 1):  # Exclude the last node in this loop
        for j in range(1, num_edges + 1):  # Skip the first index (self-loop), limit to num_edges
            distance = distances[i, j]
            weight = (np.exp(-distance) * (np.e - np.exp(distance))) / (np.e - 1)
            edge_index.extend([[i, indices[i, j]], [indices[i, j], i]])
            edge_weight.extend([weight, weight])

    # Handling the last node separately (directed to supernode)
    last_node_index = num_samples - 1
    # Add an edge from every other node to the last node
    for i in range(num_samples):
        if i != last_node_index:
            distance = np.linalg.norm(positions[last_node_index] - positions[i])  # Calculate the Euclidean distance
            weight = (np.exp(-distance) * (np.e - np.exp(distance))) / (np.e - 1)
            edge_index.append([i, last_node_index])
            edge_weight.append(weight)

    # Add self-loops for all nodes
    for i in range(num_samples):
        edge_index.append([i, i])
        edge_weight.append(1.0)  # Uniform weight for self-loops

    # Convert edge_index and edge_weight to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # Add data.y
    jammer_positions = np.array(instance.jammer_position).reshape(-1, global_config.args.out_features)
    y = torch.tensor(jammer_positions, dtype=torch.float)

    # Plot graph
    if global_config.args.plot_network:
        plot_graph(positions=instance.node_positions, node_features=node_features_tensor,  edge_index=edge_index, edge_weights=edge_weight, show_weights=True, perc_completion=instance.perc_completion)

    # Convert WCL centroid prediction to tensor
    wcl_pred_tensor = torch.tensor(instance.node_positions[-1], dtype=torch.float32).unsqueeze(0)

    # Create the Data object
    data = Data(x=node_features_tensor, edge_index=edge_index, edge_weight=edge_weight, y=y, wcl_pred=wcl_pred_tensor)

    # Store additional instance informaltion to Data object
    data.max_radius = torch.tensor(instance.max_radius, dtype=torch.float)
    data.perc_completion = torch.tensor(instance.perc_completion, dtype=torch.float)
    data.pl_exp = torch.tensor(instance.pl_exp, dtype=torch.float)
    data.sigma = torch.tensor(instance.sigma, dtype=torch.float)
    data.jtx = torch.tensor(instance.jammer_power, dtype=torch.float)
    data.num_samples = torch.tensor(instance.num_samples, dtype=torch.float)

    return data
