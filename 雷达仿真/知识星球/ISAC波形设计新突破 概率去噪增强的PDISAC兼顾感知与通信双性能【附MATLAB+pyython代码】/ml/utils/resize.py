import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from typing import Union, Tuple, Optional, Literal


class ComplexMatrixResizer:
    """
    A class for resizing complex matrices with high-quality interpolation.
    Supports both NumPy and PyTorch tensors.
    """

    def __init__(
        self, method: Literal["bilinear", "bicubic", "lanczos", "fourier"] = "fourier"
    ):
        """
        Initialize the ComplexMatrixResizer.

        Args:
            method: Interpolation method to use. Options:
                - 'bilinear': Fast, smooth results
                - 'bicubic': Higher quality, smoother
                - 'lanczos': Best quality for downsampling
                - 'fourier': Frequency domain resizing (best for preserving spectral content)
        """
        self.method = method
        self.original_matrix = None
        self.original_shape = None
        self.is_torch = False
        self.device = None

    def scale(
        self, X: Union[np.ndarray, torch.Tensor], target_shape: Tuple[int, int]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Scale the complex matrix to the target shape.

        Args:
            X: Input complex matrix of shape (M, N)
            target_shape: Target shape (A, B)

        Returns:
            Resized complex matrix of shape (A, B)
        """
        # Store original for reverse operation
        self.original_matrix = X.copy() if isinstance(X, np.ndarray) else X.clone()
        self.original_shape = X.shape
        self.is_torch = isinstance(X, torch.Tensor)

        if self.is_torch:
            self.device = X.device
            return self._scale_torch(X, target_shape)
        else:
            return self._scale_numpy(X, target_shape)

    def _scale_numpy(self, X: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Scale NumPy complex matrix."""
        if self.method == "fourier":
            return self._fourier_resize_numpy(X, target_shape)

        # Separate real and imaginary parts
        real_part = X.real
        imag_part = X.imag

        # Choose interpolation order based on method
        order_map = {"bilinear": 1, "bicubic": 3, "lanczos": 5}
        order = order_map.get(self.method, 3)

        # Calculate zoom factors
        zoom_factors = (target_shape[0] / X.shape[0], target_shape[1] / X.shape[1])

        # Resize real and imaginary parts separately
        real_resized = ndimage.zoom(
            real_part, zoom_factors, order=order, mode="reflect", prefilter=True
        )
        imag_resized = ndimage.zoom(
            imag_part, zoom_factors, order=order, mode="reflect", prefilter=True
        )

        # Combine back into complex matrix
        return real_resized + 1j * imag_resized

    def _scale_torch(
        self, X: torch.Tensor, target_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """Scale PyTorch complex tensor."""
        if self.method == "fourier":
            return self._fourier_resize_torch(X, target_shape)

        # Separate real and imaginary parts
        real_part = X.real.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        imag_part = X.imag.unsqueeze(0).unsqueeze(0)

        # Map method to torch interpolation mode
        mode_map = {
            "bilinear": "bilinear",
            "bicubic": "bicubic",
            "lanczos": "bicubic",  # PyTorch doesn't have Lanczos, use bicubic
        }
        mode = mode_map.get(self.method, "bicubic")

        # Resize using interpolate
        real_resized = F.interpolate(
            real_part, size=target_shape, mode=mode, align_corners=True
        )
        imag_resized = F.interpolate(
            imag_part, size=target_shape, mode=mode, align_corners=True
        )

        # Remove batch and channel dimensions and combine
        real_resized = real_resized.squeeze(0).squeeze(0)
        imag_resized = imag_resized.squeeze(0).squeeze(0)

        return torch.complex(real_resized, imag_resized)

    def _fourier_resize_numpy(
        self, X: np.ndarray, target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize using FFT-based method (best for preserving frequency content).
        """
        # Take 2D FFT
        fft = np.fft.fft2(X)
        fft_shifted = np.fft.fftshift(fft)

        # Create output array
        output_fft = np.zeros(target_shape, dtype=np.complex128)

        # Calculate crop/pad dimensions
        in_h, in_w = X.shape
        out_h, out_w = target_shape

        # Center crop or pad in frequency domain
        if out_h <= in_h and out_w <= in_w:
            # Downsampling: crop center frequencies
            h_start = (in_h - out_h) // 2
            w_start = (in_w - out_w) // 2
            output_fft = fft_shifted[
                h_start : h_start + out_h, w_start : w_start + out_w
            ]
        else:
            # Upsampling: pad with zeros
            h_start = (out_h - in_h) // 2
            w_start = (out_w - in_w) // 2
            h_end = h_start + in_h
            w_end = w_start + in_w
            output_fft[h_start:h_end, w_start:w_end] = fft_shifted

        # Inverse FFT
        output_fft_shifted = np.fft.ifftshift(output_fft)
        output = np.fft.ifft2(output_fft_shifted)

        # Scale to preserve energy
        scale_factor = (out_h * out_w) / (in_h * in_w)
        return output * scale_factor

    def _fourier_resize_torch(
        self, X: torch.Tensor, target_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Resize using FFT-based method for PyTorch tensors.
        """
        # Take 2D FFT
        fft = torch.fft.fft2(X)
        fft_shifted = torch.fft.fftshift(fft)

        # Create output tensor
        output_fft = torch.zeros(target_shape, dtype=torch.complex128, device=X.device)

        # Calculate crop/pad dimensions
        in_h, in_w = X.shape
        out_h, out_w = target_shape

        # Center crop or pad in frequency domain
        if out_h <= in_h and out_w <= in_w:
            # Downsampling: crop center frequencies
            h_start = (in_h - out_h) // 2
            w_start = (in_w - out_w) // 2
            output_fft = fft_shifted[
                h_start : h_start + out_h, w_start : w_start + out_w
            ]
        else:
            # Upsampling: pad with zeros
            h_start = (out_h - in_h) // 2
            w_start = (out_w - in_w) // 2
            h_end = h_start + in_h
            w_end = w_start + in_w
            output_fft[h_start:h_end, w_start:w_end] = fft_shifted

        # Inverse FFT
        output_fft_shifted = torch.fft.ifftshift(output_fft)
        output = torch.fft.ifft2(output_fft_shifted)

        # Scale to preserve energy
        scale_factor = (out_h * out_w) / (in_h * in_w)
        return output * scale_factor

    def reverse(
        self, X_scaled: Union[np.ndarray, torch.Tensor], target_shape: Tuple[int, int]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Resize a scaled complex matrix back to a larger shape.
        This does not guarantee exact recovery of the original matrix,
        but upscales using the same interpolation/Fourier method.

        Args:
            X_scaled: Input scaled matrix (complex, numpy or torch)
            target_shape: Shape to upscale to (e.g., original size)

        Returns:
            Resized complex matrix with shape = target_shape
        """
        if isinstance(X_scaled, np.ndarray):
            if self.method == "fourier":
                return self._fourier_resize_numpy(X_scaled, target_shape)
            else:
                return self._scale_numpy(X_scaled, target_shape)
        elif isinstance(X_scaled, torch.Tensor):
            if self.method == "fourier":
                return self._fourier_resize_torch(X_scaled, target_shape)
            else:
                return self._scale_torch(X_scaled, target_shape)
        else:
            raise TypeError("Input must be np.ndarray or torch.Tensor")

    def scale_with_antialiasing(
        self, X: Union[np.ndarray, torch.Tensor], target_shape: Tuple[int, int]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Scale with additional anti-aliasing for better quality when downsampling.

        Args:
            X: Input complex matrix
            target_shape: Target shape (A, B)

        Returns:
            Resized complex matrix with anti-aliasing applied
        """
        self.original_matrix = X.copy() if isinstance(X, np.ndarray) else X.clone()
        self.original_shape = X.shape
        self.is_torch = isinstance(X, torch.Tensor)

        # Check if downsampling
        downsample = (target_shape[0] < X.shape[0]) or (target_shape[1] < X.shape[1])

        if downsample and isinstance(X, np.ndarray):
            # Apply Gaussian filter before downsampling
            sigma = max(X.shape[0] / target_shape[0], X.shape[1] / target_shape[1]) / 2
            real_part = ndimage.gaussian_filter(X.real, sigma=sigma)
            imag_part = ndimage.gaussian_filter(X.imag, sigma=sigma)
            X_filtered = real_part + 1j * imag_part
            return self._scale_numpy(X_filtered, target_shape)
        elif downsample and isinstance(X, torch.Tensor):
            # For PyTorch, the interpolate function handles anti-aliasing internally
            # when using bicubic mode
            return self._scale_torch(X, target_shape)
        else:
            # No anti-aliasing needed for upsampling
            return self.scale(X, target_shape)


# Example usage and testing
if __name__ == "__main__":
    # Create a sample complex matrix
    M, N = 1280, 1280
    X_numpy = np.random.randn(M, N) + 1j * np.random.randn(M, N)

    # Test with NumPy
    resizer_np = ComplexMatrixResizer(method="bicubic")

    # Scale to larger size
    X_scaled_up = resizer_np.scale(X_numpy, (256, 256))
    print(f"Original shape: {X_numpy.shape}")
    print(f"Scaled up shape: {X_scaled_up.shape}")

    # Get original back
    X_original = resizer_np.reverse()
    print(f"Recovered original shape: {X_original.shape}")
    print(f"Original preserved: {np.allclose(X_original, X_numpy)}")

    # # Scale to smaller size with anti-aliasing
    # resizer_np2 = ComplexMatrixResizer(method="lanczos")
    # X_scaled_down = resizer_np2.scale_with_antialiasing(X_numpy, (128, 128))
    # print(f"Scaled down shape: {X_scaled_down.shape}")

    # # Test with PyTorch
    # X_torch = torch.randn(M, N, dtype=torch.complex64)
    # resizer_torch = ComplexMatrixResizer(method="fourier")

    # X_torch_scaled = resizer_torch.scale(X_torch, (512, 512))
    # print(f"\nPyTorch - Original shape: {X_torch.shape}")
    # print(f"PyTorch - Scaled shape: {X_torch_scaled.shape}")

    # # Test different methods
    # methods = ["bilinear", "bicubic", "lanczos", "fourier"]
    # for method in methods:
    #     resizer = ComplexMatrixResizer(method=method)
    #     result = resizer.scale(X_numpy, (1280, 1280))
    #     print(f"\nMethod '{method}' - Output shape: {result.shape}")
    #     print(
    #         f"Method '{method}' - Magnitude range: [{np.abs(result).min():.4f}, {np.abs(result).max():.4f}]"
    #     )
