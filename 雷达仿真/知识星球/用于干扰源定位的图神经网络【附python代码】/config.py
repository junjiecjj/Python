params = {
    'model': 'GAT',
    'learning_rate': 0.0006703373871168542,
    'weight_decay': 0.00001,
    'test_batch_size': 8,
    'batch_size': 8,
    'dropout_rate': 0.5,
    'num_heads': 4,
    'num_layers': 8,
    'hidden_channels': 128,
    'out_channels': 128,
    'in_channels': 22,
    'out_features': 5,  # 3 jammer pos (x, y, z) # 5 (r, sin(theta), cos(theta), sin(phi), cos(phi))
    'max_epochs': 300,
    '3d': True,
    'required_features': ['node_positions', 'node_noise'],  # node_positions, polar_coordinates, node_noise, node_rssi
    'additional_features': ['weighted_centroid_radius','weighted_centroid_sin_theta', 'weighted_centroid_cos_theta', 'weighted_centroid_sin_az', 'weighted_centroid_cos_az', 'dist_to_wcl', 'median_noise', 'max_noise', 'noise_differential', 'vector_x', 'vector_y', 'vector_z', 'rate_of_change_signal_strength'],
    'num_neighbors': 3,
    'activation': False,
    'max_nodes': 1000,
    'experiments_folder': 'static/newr/gat/3d/newp1/newest_wcl/sn_wn_grpool_snweight_sndir_3mlp_newloss/',
    'dataset_path': 'data/controlled_path_data_1000_3d_original.pkl',  # combined_fspl_log_distance.csv',  # combined_urban_area.csv # linear_path_static_new_1to10k_updated_aoa.pkl
    'dynamic': True,
    'downsample': True,
    'train_per_class': True,
    'inference': False,
    'reproduce': True,
    'plot_network': False,
    'val_discrite_coeff': 0.1,
    'test_discrite_coeff': 0.1,  # disable discritization -> step_size = 1
    'num_workers': 16,
    'aug': ['drop_node'],
}

