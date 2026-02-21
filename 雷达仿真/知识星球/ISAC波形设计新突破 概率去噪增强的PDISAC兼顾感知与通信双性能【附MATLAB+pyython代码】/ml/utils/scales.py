import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import resize


def scale_complex_matrices(matrices, a=0, b=1, scale_magnitude=True):
    """
    Scale a list of complex matrices to the range [a, b].

    Parameters:
    - matrices: List of numpy arrays (complex-valued matrices).
    - a: Lower bound of the target range.
    - b: Upper bound of the target range.
    - scale_magnitude: If True, scale based on magnitude of complex numbers.
                      If False, scale real and imaginary parts separately.

    Returns:
    - List of scaled complex matrices.
    """

    if not isinstance(matrices, list):
        matrices = [matrices]  # Ensure input is a list

    scaled_matrices = []

    if scale_magnitude:
        # Compute global min and max of magnitudes across all matrices
        magnitudes = [np.abs(matrix).ravel() for matrix in matrices]
        global_min = min([np.min(mag) for mag in magnitudes])
        global_max = max([np.max(mag) for mag in magnitudes])

        # Avoid division by zero
        if global_max == global_min:
            if len(matrices) == 1:
                return matrices[0]
            return [matrix.copy() for matrix in matrices]  # No scaling needed

        for matrix in matrices:
            # Scale the magnitude to [a, b]
            magnitude = np.abs(matrix)
            scaled_magnitude = a + (b - a) * (magnitude - global_min) / (
                global_max - global_min
            )
            # Preserve the phase
            phase = np.angle(matrix)
            # Reconstruct complex numbers
            scaled_matrix = scaled_magnitude * np.exp(1j * phase)
            scaled_matrices.append(scaled_matrix)

    else:
        # Scale real and imaginary parts separately
        real_parts = [np.real(matrix).ravel() for matrix in matrices]
        imag_parts = [np.imag(matrix).ravel() for matrix in matrices]
        global_min_real = min([np.min(real) for real in real_parts])
        global_max_real = max([np.max(real) for real in real_parts])
        global_min_imag = min([np.min(imag) for imag in imag_parts])
        global_max_imag = max([np.max(imag) for imag in imag_parts])

        # Avoid division by zero
        if global_max_real == global_min_real or global_max_imag == global_min_imag:
            if len(matrices) == 1:
                return matrices[0]
            return [matrix.copy() for matrix in matrices]  # No scaling needed

        for matrix in matrices:
            real_part = np.real(matrix)
            imag_part = np.imag(matrix)
            # Scale real and imaginary parts
            scaled_real = a + (b - a) * (real_part - global_min_real) / (
                global_max_real - global_min_real
            )
            scaled_imag = a + (b - a) * (imag_part - global_min_imag) / (
                global_max_imag - global_min_imag
            )
            # Reconstruct complex matrix
            scaled_matrix = scaled_real + 1j * scaled_imag
            scaled_matrices.append(scaled_matrix)

    if len(matrices) == 1:
        return scaled_matrices[0]

    return scaled_matrices


def resize_matrix(matrix, new_hw=(256, 256)):
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.ndim != 2
        or not np.iscomplexobj(matrix)
    ):
        raise ValueError("Input matrix must be a 2D complex NumPy array")

    # Extract the new dimensions
    new_h, new_w = new_hw

    if matrix.shape[-1] == 1:
        new_w = 1

    # Resize real and imaginary parts separately
    resized_real = resize(matrix.real, (new_h, new_w), order=1, preserve_range=True)
    resized_imag = resize(matrix.imag, (new_h, new_w), order=1, preserve_range=True)

    # Combine them back into a complex matrix
    resized_matrix = resized_real + 1j * resized_imag

    return resized_matrix


class MinMaxScaler:
    def __init__(self, feature_range=(-1, 1), device="cpu"):
        """
        Args:
            feature_range (tuple or None): Desired range for normalization, default (-1, 1).
                                          If None, no normalization is applied.
            device (str): PyTorch device ('cpu' or 'cuda')
        """
        self.feature_range = feature_range  # Can be None to skip normalization
        self.device = torch.device(device)
        self.original_shape = None
        self.transformed_shape = None
        self.shape_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.min_val = None
        self.max_val = None
        self.residual = None  # Store residual for accurate recovery

    def _to_tensor(self, data):
        """Convert data to PyTorch tensor"""
        if isinstance(data, np.ndarray):
            return torch.FloatTensor(data).to(self.device)
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            raise ValueError("Input must be numpy array or torch tensor")

    def _to_numpy(self, data):
        """Convert tensor to numpy"""
        return data.cpu().numpy()

    def _find_closest_above(self, value):
        """Find the smallest value in shape_list that is >= input value"""
        for shape in self.shape_list:
            if shape >= value:
                return shape
        return self.shape_list[-1]

    def _resize(self, data, target_h, target_w):
        """Resize data using bicubic interpolation"""
        if len(data.shape) == 4:
            return F.interpolate(
                data, size=(target_h, target_w), mode="bicubic", align_corners=True
            )
        else:
            data = data.unsqueeze(0)
            resized = F.interpolate(
                data, size=(target_h, target_w), mode="bicubic", align_corners=True
            )
            return resized.squeeze(0)

    def fit_transform(self, data, return_numpy=True):
        """Transform data with optional normalization and resizing, storing residual"""
        data = self._to_tensor(data)
        self.original_shape = data.shape
        h, w = self.original_shape[-2:]

        # Normalize only if feature_range is specified
        if self.feature_range is not None:
            self.min_val = torch.min(data)
            self.max_val = torch.max(data)
            if self.max_val != self.min_val:
                normalized = (data - self.min_val) / (self.max_val - self.min_val) * (
                    self.feature_range[1] - self.feature_range[0]
                ) + self.feature_range[0]
            else:
                normalized = data * 0 + self.feature_range[0]
        else:
            normalized = data  # Skip normalization

        # Resize to target shape
        h_target = self._find_closest_above(h)
        w_target = self._find_closest_above(w)
        transformed = self._resize(normalized, h_target, w_target)
        self.transformed_shape = transformed.shape

        # Compute residual for accurate recovery
        transformed_back = self._resize(transformed, h, w)
        self.residual = normalized - transformed_back  # Error from resizing

        if return_numpy:
            transformed = self._to_numpy(transformed)
        return transformed

    def inverse_transform(
        self, transformed_data, return_numpy=True, original_shape=None
    ):
        """Reverse the transformation with residual correction"""
        transformed_data = self._to_tensor(transformed_data)

        if original_shape is not None:
            self.original_shape = original_shape

        if self.original_shape is None:
            raise ValueError("Scaler must be fitted first using fit_transform.")

        orig_h, orig_w = self.original_shape[-2:]

        # Resize back to original shape
        resized_back = self._resize(transformed_data, orig_h, orig_w)

        # Apply residual correction
        if self.residual is not None:
            corrected = resized_back + self.residual
        else:
            corrected = resized_back

        # Denormalize only if feature_range was specified
        if self.feature_range is not None:
            if self.max_val != self.min_val:
                recovered = (corrected - self.feature_range[0]) / (
                    self.feature_range[1] - self.feature_range[0]
                ) * (self.max_val - self.min_val) + self.min_val
            else:
                recovered = corrected * 0 + self.min_val
        else:
            recovered = corrected  # No denormalization

        if return_numpy:
            recovered = self._to_numpy(recovered)
        return recovered


# Example usage
if __name__ == "__main__":
    data = torch.rand(2, 255, 256)  # Example data tensor

    # Test with normalization
    scaler_with_norm = MinMaxScaler(feature_range=None, device="cpu")
    transformed_data_norm = scaler_with_norm.fit_transform(data, return_numpy=True)
    # transformed_data_norm += np.random.normal(loc=0, scale=np.sqrt(noise_variance / 2), size=transformed_data_norm.shape)
    recovered_data_norm = scaler_with_norm.inverse_transform(
        transformed_data_norm, return_numpy=True
    )

    print("Transformed shape with normalization:", transformed_data_norm.shape)
    print("Recovered shape with normalization:", recovered_data_norm.shape)
    # Check if original and recovered data match
    assert np.allclose(data.numpy(), recovered_data_norm, atol=1e-5), (
        "Data mismatch after normalization and recovery"
    )
