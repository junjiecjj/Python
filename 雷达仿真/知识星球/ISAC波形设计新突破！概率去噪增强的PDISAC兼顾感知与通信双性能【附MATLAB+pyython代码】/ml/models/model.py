import time
import torch
from torch import nn


class DenoiseModel(nn.Module):
    def __init__(self, model: nn.Module, domain: str) -> None:
        super(DenoiseModel, self).__init__()
        self.domain = domain
        self.model = model

    def _preprocess(self, x):
        # x: (batch, fast, slow)
        if self.domain == "time":
            # FFT along slow-time axis (dim=2), shift
            rd_map = torch.fft.fftshift(torch.fft.fft(x, dim=2), dim=2)
            return rd_map
        elif self.domain == "freq":
            # Doppler FFT along slow-time axis (dim=2)
            doppler_freq = torch.fft.fftshift(torch.fft.fft(x, dim=2), dim=2)
            # IFFT along fast-time axis (dim=1)
            rd_map = torch.fft.ifft(doppler_freq, dim=1)
            return rd_map
        else:
            raise ValueError(f"Unknown domain: {self.domain}")

    def _postprocess(self, rd_map):
        # rd_map: (batch, fast, slow)
        if self.domain == "time":
            # Inverse of FFT (dim=2) with shift
            time_data = torch.fft.ifft(torch.fft.ifftshift(rd_map, dim=2), dim=2)
            return time_data
        elif self.domain == "freq":
            # Undo Doppler processing
            doppler_time = torch.fft.fft(rd_map, dim=1)  # FFT along fast-time
            time_data = torch.fft.ifft(torch.fft.ifftshift(doppler_time, dim=2), dim=2)
            return time_data
        else:
            raise ValueError(f"Unknown domain: {self.domain}")

    def convert_to_complex(self, x):
        return torch.complex(x[:, 0, ...], x[:, 1, ...])

    def reverse_from_complex(self, x):
        real_part = x.real
        imag_part = x.imag
        return torch.stack([real_part, imag_part], dim=1)

    def forward(
        self, Y: torch.Tensor = None, S: torch.Tensor = None, Z: torch.Tensor = None
    ):
        if isinstance(S, torch.Tensor) and isinstance(Z, torch.Tensor):
            Y = self.reverse_from_complex(self._preprocess(self.convert_to_complex(Y)))
            # S = self.reverse_from_complex(self._preprocess(self.convert_to_complex(S)))
            Z = self.reverse_from_complex(self._preprocess(self.convert_to_complex(Z)))

            output_rd, kls = self.model(Y=Y, S=S, Z=Z)

            output = self.reverse_from_complex(
                self._postprocess(self.convert_to_complex(output_rd))
            )
            return output, output_rd, kls
        elif isinstance(S, torch.Tensor):
            Y = self.reverse_from_complex(self._preprocess(self.convert_to_complex(Y)))
            # S = self.reverse_from_complex(self._preprocess(self.convert_to_complex(S)))

            output_rd = self.model(Y=Y, S=S)

            output = self.reverse_from_complex(
                self._postprocess(self.convert_to_complex(output_rd))
            )
            return output, output_rd

        elif isinstance(Z, torch.Tensor):
            Y = self.reverse_from_complex(self._preprocess(self.convert_to_complex(Y)))
            Z = self.reverse_from_complex(self._preprocess(self.convert_to_complex(Z)))

            output_rd, kls = self.model(Y=Y, Z=Z)

            output = self.reverse_from_complex(
                self._postprocess(self.convert_to_complex(output_rd))
            )
            return output, output_rd, kls
        else:
            Y = self.reverse_from_complex(self._preprocess(self.convert_to_complex(Y)))

            output_rd = self.model(Y=Y)

            output = self.reverse_from_complex(
                self._postprocess(self.convert_to_complex(output_rd))
            )
            return output, output_rd

    def predict(self, Y: torch.Tensor = None, S: torch.Tensor = None):
        if isinstance(S, torch.Tensor):
            Y = self.reverse_from_complex(self._preprocess(self.convert_to_complex(Y)))
            # S = self.reverse_from_complex(self._preprocess(self.convert_to_complex(S)))

            output = self.model.predict(Y=Y, S=S)
        else:
            Y = self.reverse_from_complex(self._preprocess(self.convert_to_complex(Y)))
            output = self.model.predict(Y=Y)

        output = self.reverse_from_complex(
            self._postprocess(self.convert_to_complex(output))
        )
        return output

    def distribution(
        self,
        Y: torch.Tensor = None,
        S: torch.Tensor = None,
        Z: torch.Tensor = None,
        distributions: dict = {},
    ):
        if isinstance(S, torch.Tensor):
            Y = self.reverse_from_complex(self._preprocess(self.convert_to_complex(Y)))
            # S = self.reverse_from_complex(self._preprocess(self.convert_to_complex(S)))
            Z = self.reverse_from_complex(self._preprocess(self.convert_to_complex(Z)))

            distributions[r"$p(\mathbf{P}_{\rm prbs, rd})$"] = S
            distributions[r"$p(\mathbf{Y}_{\rm sen, rd}^{\rm prbs})$"] = Y
            distributions[r"$p(\mathbf{\tilde{Z}}_{\rm sen, rd}^{\rm prbs})$"] = Z

            output_rd, kls = self.model.distribution(
                Y=Y, S=S, Z=Z, distributions=distributions
            )
        else:
            Y = self.reverse_from_complex(self._preprocess(self.convert_to_complex(Y)))
            Z = self.reverse_from_complex(self._preprocess(self.convert_to_complex(Z)))

            distributions[r"$p(\mathbf{Y}_{\rm sen, rd}^{\rm prbs})$"] = Y
            distributions[r"$p(\mathbf{\tilde{Z}}_{\rm sen, rd}^{\rm prbs})$"] = Z

            output_rd, kls = self.model.distribution(
                Y=Y, Z=Z, distributions=distributions
            )

        output = self.reverse_from_complex(
            self._postprocess(self.convert_to_complex(output_rd))
        )

        distributions[r"$p(\mathbf{\hat{Z}}_{\rm sen, rd}^{\rm prbs})$"] = output_rd
        distributions[r"$p(\mathbf{\hat{Z}}_{\rm sen, fast}^{\rm prbs})$"] = output

        return distributions


class WrapperModel1(nn.Module):
    def __init__(self, original_model):
        super(WrapperModel1, self).__init__()
        self.model = original_model

    def forward(self, x1, y1):
        return self.model.predict(x1, y1)


class WrapperModel2(nn.Module):
    def __init__(self, original_model):
        super(WrapperModel2, self).__init__()
        self.model = original_model

    def forward(self, x1):
        return self.model.predict(x1)


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, inputs, num_runs=100, warmup=10, device="cpu"):
    """Measure average inference time"""
    model.eval()
    model.to(device)

    # Move inputs to device
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.predict(**inputs)

    # Timing
    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model.predict(**inputs)

    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs * 1000  # Convert to ms

    return avg_time


def get_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024**2)
    return size_mb


if __name__ == "__main__":
    from models.bases import dncnn_y, dncnn_ys, pdnet_y, pdnet_ys

    domain = "freq"
    device = "mps"
    print(f"Using device: {device}\n")

    c, h, w = (2, 255, 256)

    results = []

    print("=" * 80)
    print("TESTING AND BENCHMARKING MODELS")
    print("=" * 80)

    #################
    # Test PDNet YS #
    #################
    print("\n1. Testing PDNet YS (freq domain)...")
    Y = torch.randn(1, c, h, w)
    S = torch.randn(1, c, h, w)
    Z = torch.randn(1, c, h, w)

    current_model = pdnet_ys.PDNet(input_shape=(c, h, w), hidden_channel=16, level=3)
    model = DenoiseModel(model=current_model, domain=domain)

    output, *args = model(Y=Y, S=S, Z=Z)
    print(f"   Output shape: {output.shape}")

    output = model.predict(Y=Y, S=S)
    print(f"   Prediction shape: {output.shape}")

    params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    inference_time = measure_inference_time(model, {"Y": Y, "S": S}, device=device)

    results.append(
        {
            "Model": "PDNet YS",
            "Parameters": params,
            "Size (MB)": size_mb,
            "Inference Time (ms)": inference_time,
        }
    )

    ################
    # Test PDNet Y #
    ################
    print("\n2. Testing PDNet Y (freq domain)...")
    Y = torch.randn(1, c, h, w)
    Z = torch.randn(1, c, h, w)

    current_model = pdnet_y.PDNet(input_shape=(c, h, w), hidden_channel=16, level=3)
    model = DenoiseModel(model=current_model, domain=domain)

    output, *args = model(Y=Y, Z=Z)
    print(f"   Output shape: {output.shape}")

    output = model.predict(Y=Y)
    print(f"   Prediction shape: {output.shape}")

    params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    inference_time = measure_inference_time(model, {"Y": Y}, device=device)

    results.append(
        {
            "Model": "PDNet Y",
            "Parameters": params,
            "Size (MB)": size_mb,
            "Inference Time (ms)": inference_time,
        }
    )

    #################
    # Test DnCNN YS #
    #################
    print("\n3. Testing DnCNN YS (time domain)...")
    Y = torch.randn(1, c, h, w)
    S = torch.randn(1, c, h, w)  # Fixed: Changed from (1, c, h, 1) to (1, c, h, w)

    current_model = dncnn_ys.DnCNN(depth=10, img_channels=2)
    model = DenoiseModel(model=current_model, domain=domain)

    output, *args = model(Y=Y, S=S)
    print(f"   Output shape: {output.shape}")

    output = model.predict(Y=Y, S=S)
    print(f"   Prediction shape: {output.shape}")

    params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    inference_time = measure_inference_time(model, {"Y": Y, "S": S}, device=device)

    results.append(
        {
            "Model": "DnCNN YS",
            "Parameters": params,
            "Size (MB)": size_mb,
            "Inference Time (ms)": inference_time,
        }
    )

    ################
    # Test DnCNN Y #
    ################
    print("\n4. Testing DnCNN Y (time domain)...")
    Y = torch.randn(1, c, h, w)

    current_model = dncnn_y.DnCNN(depth=10, img_channels=2)
    model = DenoiseModel(model=current_model, domain=domain)

    output, *args = model(Y=Y)
    print(f"   Output shape: {output.shape}")

    output = model.predict(Y=Y)
    print(f"   Prediction shape: {output.shape}")

    params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    inference_time = measure_inference_time(model, {"Y": Y}, device=device)

    results.append(
        {
            "Model": "DnCNN Y",
            "Parameters": params,
            "Size (MB)": size_mb,
            "Inference Time (ms)": inference_time,
        }
    )

    # Print results table
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Model':<15} {'Parameters':<15} {'Size (MB)':<12} {'Inference (ms)':<15}")
    print("-" * 80)

    for result in results:
        print(
            f"{result['Model']:<15} {result['Parameters']:<15,} "
            f"{result['Size (MB)']:<12.2f} {result['Inference Time (ms)']:<15.2f}"
        )

    print("=" * 80)
    print(f"\nDevice used for inference timing: {device}")
    print(f"Input shape: ({c}, {h}, {w})")
    print("Batch size: 1")
    print("Number of timing runs: 100 (after 10 warmup runs)")
