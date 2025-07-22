import numpy as np
from scipy.integrate import simpson

def phi(u_values):
    """
    Define the base function φ(u). Replace this with any function you like.
    """
    return np.exp(-u_values)  # Example: exponential decay

def compute_approx_integral_of_psi(
    max_time: float,
    time_step: float = 1e-3,
    max_convolutions: int = 50
) -> float:
    """
    Approximate ∫₀^t ψ(u) du where ψ(u) = sum_{n=1}^∞ φ^{(*n)}(u)

    Parameters:
    - max_time: upper limit of integration (t)
    - time_step: resolution of the time discretization (Δu)
    - max_convolutions: how many convolutions to include in the sum

    Returns:
    - Approximate value of the integral
    """
    # Time grid from 0 to max_time
    time_grid = np.arange(0, max_time + time_step, time_step)

    # Evaluate φ(u) on the time grid
    phi_values = phi(time_grid)

    # Initialize ψ(u) as a zero array of the same length
    psi_values = np.zeros_like(time_grid)

    # The current convolution result starts as φ itself
    current_convolution = phi_values.copy()

    for n in range(1, max_convolutions + 1):
        # Define time grid for the current convolution result
        convolution_time_grid = np.arange(0, len(current_convolution) * time_step, time_step)

        # Interpolate the convolution to match the target time grid
        current_convolution_on_grid = np.interp(
            time_grid,
            convolution_time_grid,
            current_convolution,
            left=0.0,
            right=0.0
        )

        # Add the interpolated convolution to ψ(u)
        psi_values += current_convolution_on_grid

        # Prepare for next iteration: convolve current result with φ
        current_convolution = np.convolve(current_convolution, phi_values) * time_step

    # Finally, approximate the integral of ψ(u) from 0 to t
    integral_of_psi = simpson(psi_values, time_grid)

    return integral_of_psi

compute_approx_integral_of_psi(max_time=10, time_step=1e-3)
