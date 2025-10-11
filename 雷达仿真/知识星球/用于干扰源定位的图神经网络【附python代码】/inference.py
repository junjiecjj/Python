import pandas as pd
import subprocess
import pickle
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


# def calculate_noise(distance, P_tx_jammer, G_tx_jammer, path_loss_exponent, shadowing, pl0=32, d0=1):
#     # Prevent log of zero if distance is zero by replacing it with a very small positive number
#     d = np.where(distance == 0, np.finfo(float).eps, distance)
#     # Path loss calculation
#     path_loss_db = pl0 + 10 * path_loss_exponent * np.log10(d / d0)
#     # Apply shadowing if sigma is not zero
#     if shadowing != 0:
#         path_loss_db += np.random.normal(0, shadowing, size=d.shape)
#     return P_tx_jammer + G_tx_jammer - path_loss_db


# def generate_dummy_data(num_nodes, jammer_position, P_tx_jammer, G_tx_jammer, path_loss_exponent, shadowing):
#     # TODO: maybe instead make this load a row from test set and then use it
#     a = 10.0  # Base radius for the spiral
#     b = 10  # Growth rate for the spiral
#     a_z = 1  # Base height for the spiral
#     b_z = 10  # Growth rate for the height
#
#     # Generate theta values for the spiral
#     theta = np.linspace(0, 2 * np.pi, num_nodes)  # Two full rotations for the spiral
#     r = a + b * theta  # Radius grows with theta
#     r_z = a_z + b_z * theta  # Height grows with theta
#
#     # Convert polar coordinates to Cartesian coordinates
#     x = r * np.cos(theta) + jammer_position[0]
#     y = r * np.sin(theta) + jammer_position[1]
#     z = r_z + jammer_position[2]  # Add z component
#
#     # Stack positions into a 2D array
#     node_positions = np.vstack((x, y, z)).T.tolist()
#
#     # Calculate Euclidean distances from each node to the jammer in 3D space
#     distances = np.sqrt((x - jammer_position[0]) ** 2 + (y - jammer_position[1]) ** 2 + (z - jammer_position[2]) ** 2)
#
#     # Calculate noise for each node
#     node_noise = calculate_noise(distances, P_tx_jammer, G_tx_jammer, path_loss_exponent, shadowing)
#
#     # Create a DataFrame to store the data
#     data = {
#         'num_samples': num_nodes,
#         'node_positions': node_positions,
#         'node_noise': node_noise.tolist(),
#         'pl_exp': path_loss_exponent,  # Consistent with the input
#         'sigma': shadowing,  # Consistent with the input
#         'jammer_power': P_tx_jammer,  # Consistent with the input
#         'jammer_position': jammer_position,  # Consistent with the input
#         'jammer_gain': G_tx_jammer,  # Consistent with the input
#         'dataset': 'dynamic'
#     }
#
#     # Save data to .pkl file
#     filename = '/home/dania/gnn_jammer_localization/data/flight_data.pkl'
#     save_data_as_pkl(data, filename)
#
#     return data

# def save_data_as_pkl(data, filename):
#     with open(filename, 'wb') as pkl_file:  # 'ab' mode to append binary data
#         pickle.dump(data, pkl_file)

def plot_positions(data, jammer_position, predicted_position):
    node_positions = np.array(data['node_positions'])

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the nodes
    ax.scatter(node_positions[:, 0], node_positions[:, 1], node_positions[:, 2],
               c='blue', label='Nodes', s=1, alpha=0.5)

    # Plot the jammer position
    ax.scatter(jammer_position[0], jammer_position[1], jammer_position[2],
               c='red', marker='x', s=200, label='Jammer')

    # Plot the predicted position
    ax.scatter(predicted_position[0], predicted_position[1], predicted_position[2],
               c='green', marker='o', s=200, label='Prediction', alpha=0.8)

    # Add labels and title
    ax.set_title('Node and Jammer Positions in 3D')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')

    # Add a legend
    ax.legend()
    plt.show()

def calculate_rmse(predicted_position, actual_position):
    rmse = np.sqrt(mean_squared_error(actual_position, predicted_position))
    return rmse

def gnn():
    # Define the command and arguments
    script = 'main.py'

    args = [
        '--inference', 'True',
        '--dataset_path', '/home/dania/gnn_jammer_localization/experiments/dynamic_test_dataset.pkl',
        '--batch_size', '8',
        '--test_batch_size', '8'
    ]

    # Convert args list to string and prepare the full command
    args = [str(arg) for arg in args]  # Ensure all elements are strings
    command = ['python', script] + args

    # Execute the command
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Load the CSV file
    file_path = 'experiments/predictions_GAT_dynamic_inference.csv'
    df = pd.read_csv(file_path)

    # Convert the split strings into floats
    predictions = [float(num) for num in df['Prediction'][0].strip('[]').split()]

    print(predictions)

    return predictions


# Example usage
# num_samples = 5000
# jammer_pos = [200, 200, 20]
# jammer_ptx =  20
# jammer_gtx = 1
# plexp = 3.0
# sigma = 2
# print(f"Ptx jammer: {jammer_ptx}, Gtx jammer: {jammer_gtx}, PL: {plexp}, Sigma: {sigma}")

# data = generate_dummy_data(num_samples, jammer_pos, jammer_ptx, jammer_gtx, plexp, sigma)
# data_original = data.copy()
predicted_jammer_pos = gnn()
# plot_positions(data_original, jammer_pos, predicted_jammer_pos)
# rmse = calculate_rmse(predicted_jammer_pos, jammer_pos)
# print(f"RMSE: {round(rmse, 2)} m")
