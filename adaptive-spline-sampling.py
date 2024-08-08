from ouroboros.helpers.spline import Spline
import numpy as np
import matplotlib.pyplot as plt


def generate_sample_curve_helix(
    start_z=-10, end_z=10, radius=1, z_rate=1, num_points=100
) -> np.ndarray:
    """
    Generates a sample helix curve for testing purposes.

    Parameters
    ----------
    start_z : float
        The starting z value.
    end_z : float
        The ending z value.
    radius : float
        The radius of the helix.
    z_rate : float
        The rate at which the z value changes.
    num_points : int
        The number of points to generate.

    Returns
    -------
    np.ndarray
        The sample helix curve (num_points, 3).
    """
    t = np.linspace(start_z, end_z, num_points)
    x = np.cos(t) * radius
    y = np.sin(t) * radius
    z = t * z_rate

    return np.vstack((x, y, z)).T


def generate_sample_curve_parabola(start_x=-10, end_x=10, num_points=100) -> np.ndarray:
    """
    Generates a sample parabola curve for testing purposes.

    Parameters
    ----------
    start_x : float
        The starting x value.
    end_x : float
        The ending x value.
    num_points : int
        The number of points to generate.

    Returns
    -------
    np.ndarray
        The sample parabola curve (num_points, 3).
    """
    t = np.linspace(start_x, end_x, num_points)
    x = t
    # y = (t**2) / (end_x - start_x)  # Normalize y to match x scale
    y = (t**2 * np.cos(t)) / (end_x - start_x)  # Normalize y to match x scale
    z = np.zeros_like(t)

    return np.vstack((x, y, z)).T


def calculate_spline_curvature(spline: Spline, t: np.ndarray) -> np.ndarray:
    """
    Calculate the curvature of a spline at a given set of points.

    Parameters
    ----------
    spline : Spline
        The spline to evaluate.
    t : np.ndarray
        The points at which to evaluate the curvature.

    Returns
    -------
    np.ndarray
        The curvature at each point.
    """
    # Calculate the first and second derivatives
    first_derivative = spline(t, derivative=1).T
    second_derivative = spline(t, derivative=2).T

    # Calculate the curvature
    numerator = np.linalg.norm(np.cross(first_derivative, second_derivative), axis=1)
    denominator = np.linalg.norm(first_derivative) ** 3
    curvature = numerator / denominator

    return curvature


def calculate_curvature_parameterization(spline: Spline, t: np.ndarray) -> np.ndarray:
    """
    Calculate the curvature parameterization of a spline at a given set of points.

    Parameters
    ----------
    spline : Spline
        The spline to evaluate.
    t : np.ndarray
        The points at which to evaluate the curvature parameterization.

    Returns
    -------
    np.ndarray
        The curvature parameterization at each point.
    """
    # Calculate the curvature
    curvature = calculate_spline_curvature(spline, t)

    # Calculate the first derivative of the curvature
    first_derivative = spline(t, derivative=1).T
    len_term = np.linalg.norm(first_derivative)

    # Calculate the cumulative sum of the curvature
    curvature_sum = np.cumsum(curvature * len_term)

    return curvature_sum


def calculate_arc_length(spline: Spline, t: np.ndarray) -> np.ndarray:
    """
    Calculate the cumulative arc length of a spline at a given set of points.

    Parameters
    ----------
    spline : Spline
        The spline to evaluate.
    t : np.ndarray
        The points at which to evaluate the arc length.

    Returns
    -------
    np.ndarray
        The arc length at each point.
    """

    # def arc_length_integrand(t):
    #     first_derivative = spline(t, derivative=1).T
    #     return np.linalg.norm(first_derivative, axis=1)

    # arc_length = np.zeros_like(t)
    # for i in range(1, len(t)):
    #     arc_length[i] = np.trapz(arc_length_integrand(t[: i + 1]), t[: i + 1])

    # Evaluate the B-spline derivative
    dx, dy, dz = spline(t, derivative=1)

    # Compute the cumulative arc length using the derivatives
    dists = np.linalg.norm([dx, dy, dz], axis=0)
    arc_length = np.cumsum(dists * np.diff(t, prepend=0))

    return arc_length


def adaptive_curvature_parameterization(
    spline: Spline, sample_params: np.ndarray, calculation_params: np.ndarray, ratio=1
) -> np.ndarray:
    """
    Sample spline parameters based on both curvature and arc length.

    Based on: https://doi.org/10.1016/j.cagd.2017.11.004

    Parameters
    ----------
    spline : Spline
        The spline to sample.
    sample_params : np.ndarray
        The points at which to sample the spline.
    calculation_params : np.ndarray
        The points at which to calculate the curvature and arc length.
    ratio : float
        The ratio between the curvature and arc length components.

    Returns
    -------
    np.ndarray
        The sampled parameters.
    """

    # Convert ratio to weights for the components
    curvature_weight = ratio / (1 + ratio)
    arc_length_weight = 1 - curvature_weight

    arc_length = calculate_arc_length(spline, calculation_params)
    curvature = calculate_curvature_parameterization(spline, calculation_params)

    arc_length_component = arc_length / arc_length[-1]
    curvature_component = curvature / curvature[-1]

    hybrid_parameter = (
        arc_length_weight * arc_length_component
        + curvature_weight * curvature_component
    )

    # Invert the parameterization axes
    # Note: This is done because we want more points where the curvature is high
    # and less points where the curvature is low. This is the opposite of the
    # original parameterization.
    return np.interp(sample_params, hybrid_parameter, calculation_params)


def adaptive_curvature_sampling(
    spline: Spline, n_sample_points=100, n_calculation_points=10000, ratio=1
) -> np.ndarray:
    """
    Sample spline points based on both curvature and arc length.

    Based on: https://doi.org/10.1016/j.cagd.2017.11.004

    Parameters
    ----------
    spline : Spline
        The spline to sample.
    t : np.ndarray
        The points at which to evaluate the spline.
    ratio : float
        The ratio between the curvature and arc length components.

    Returns
    -------
    np.ndarray
        The sampled points.
    """

    sample_params = np.linspace(0, 1, n_sample_points)
    calculation_params = np.linspace(0, 1, n_calculation_points)

    parameters = adaptive_curvature_parameterization(
        spline, sample_params, calculation_params, ratio=ratio
    )

    # Evaluate the spline at the sampled parameters
    return spline(parameters)


if __name__ == "__main__":
    DIST_BETWEEN_POINTS = 2
    RATIO = 1  # E.G. 1/2 for bias towards arc length, 2 for bias towards curvature

    # Generate a sample helix curve
    # curve = generate_sample_curve_helix(
    #     start_z=0, end_z=20, z_rate=10, radius=5, num_points=100
    # )
    curve = generate_sample_curve_parabola(start_x=-20, end_x=20, num_points=300)

    # Fit a spline to the curve
    spline = Spline(curve, degree=3)

    # Evaluate the spline
    t = np.linspace(0, 1, 1000)
    new_points = spline(t)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("Adaptive Spline Sampling")

    # Plot the arc length
    arc_length = calculate_arc_length(spline, t)
    ax3 = fig.add_subplot(222)
    ax3.plot(arc_length, t, "b-", label="Cumulative Arc Length")
    ax3.set_xlabel("Cumulative Arc Length")
    ax3.set_ylabel("t")
    ax3.legend()

    # Plot the hybrid curvature sampling
    hybrid_parameter = adaptive_curvature_parameterization(spline, t, t, ratio=RATIO)
    ax5 = fig.add_subplot(223)
    ax5.plot(hybrid_parameter, t, "g-", label="Adaptive Curvature Sampling")
    ax5.set_xlabel("Adaptive Curvature Sampling")
    ax5.set_ylabel("t")
    ax5.legend()

    # Plot the curvature parameterization
    curvature_parameterization = calculate_curvature_parameterization(spline, t)
    ax4 = fig.add_subplot(224)
    ax4.plot(curvature_parameterization, t, "r-", label="Cumulative Curvature")
    ax4.set_xlabel("Cumulative Curvature")
    ax4.set_ylabel("t")
    ax4.legend()

    # Sample the spline using adaptive curvature sampling
    total_arc_length = arc_length[-1]
    num_points = int(total_arc_length / DIST_BETWEEN_POINTS)

    hybrid_points = adaptive_curvature_sampling(
        spline, n_sample_points=num_points, ratio=RATIO
    )

    # Plot the original and interpolated curves
    ax = fig.add_subplot(221, projection="3d")
    ax.set_aspect("equal")

    # Determine the axis limits
    min_val = np.min(new_points)
    max_val = np.max(new_points)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_zlim(min_val, max_val)

    # ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], "r-", label="Original Curve")
    ax.plot(
        new_points[0], new_points[1], new_points[2], "b-", label="Interpolated Curve"
    )
    ax.plot(
        hybrid_points[0],
        hybrid_points[1],
        hybrid_points[2],
        "go",
        label="Adaptive Curve",
        markersize=4,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    ax.elev = 70
    ax.azim = -80

    plt.show()

    # Save the figure
    fig.savefig("adaptive-spline-sampling.png")
