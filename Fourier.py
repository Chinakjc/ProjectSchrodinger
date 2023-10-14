import numpy as np
import Quadrature
import Domain


class FourierTransform:

    @classmethod
    def forward_transform_1D2(cls, function_values: np.ndarray, engine="numpy") -> np.ndarray:
        """Perform the forward Fourier transform for a 1D function."""
        if engine == "numpy":
            return np.fft.fft(function_values)
        elif engine == "quadrature":
            N = len(function_values)
            result = np.zeros(N, dtype=complex)
            geo = {"x0": [0, 1]}
            space_step = 1.0 / N

            for freq in range(N):
                def integrand(x):
                    xx = x[0:-1]
                    return function_values[(xx * N).astype(int)] * np.exp(-2j * np.pi * freq * xx)

                result[freq] = Quadrature.Quadrature.rectangle_method(geo, space_step, integrand)

            return result * N
        else:
            raise ValueError(f"Unknown engine: {engine}")

    @classmethod
    def forward_transform_1D(cls, function_values: np.ndarray, engine="numpy") -> np.ndarray:
        """Perform the forward Fourier transform for a 1D function."""
        if engine == "numpy":
            return np.fft.fft(function_values)
        elif engine == "quadrature":
            N = len(function_values)
            result = np.zeros(N, dtype=complex)
            geo = {"x0": [0, 1]}
            space_step = 1.0 / N
            xx = Domain.Domain(geo, space_step).get_discretized_points(0)[0:-1]

            for freq in range(N):
                freq_coefs = np.exp(-2j * np.pi * freq * xx)
                integrand_values = function_values * freq_coefs
                # Compute the integral using rectangle_method_1D_N
                result[freq] = Quadrature.Quadrature.rectangle_method_1D_N(space_step, integrand_values)

            return result * N
        else:
            raise ValueError(f"Unknown engine: {engine}")

    @classmethod
    def forward_transform(cls, function_values: np.ndarray, engine="numpy") -> np.ndarray:
        """Perform the forward Fourier transform."""
        if engine == "numpy":
            return np.fft.fftn(function_values)

        else:
            raise ValueError(f"Unknown engine: {engine}")

    @classmethod
    def inverse_transform(cls, transformed_values: np.ndarray, engine="numpy") -> np.ndarray:
        """Perform the inverse Fourier transform."""
        if engine == "numpy":
            return np.fft.ifftn(transformed_values)
        else:
            raise ValueError(f"Unknown engine: {engine}")

    @classmethod
    def derivative_transform_1D(cls, fourier_coeffs: np.ndarray) -> np.ndarray:
        """
        Compute the Fourier transform of the derivative of a function given its Fourier coefficients.

        Parameters:
        - fourier_coeffs (np.ndarray): Fourier coefficients of the original function.

        Returns:
        - np.ndarray: Fourier coefficients of the derivative of the function.
        """
        N = len(fourier_coeffs)
        # Create an array of frequency values.
        # Note: This assumes the input is a 1D Fourier transform.
        k_values = np.fft.fftfreq(N) * N
        return np.pi * 1j * k_values * fourier_coeffs
