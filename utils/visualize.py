import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(
    history,
    label,
    xlabel="Episode",
    ylabel="Value",
    title="Training History",
    save_path="temp/temp.jpg",
    smooth_type="ma",
    smooth_param=1,
):
    """
    Plots the training history of a model with optional smoothing.

    Args:
        history (list): Training values.
        label (str): Legend label.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
        title (str): Plot title.
        save_path (str): Path to save the figure.
        smooth_type (str): Type of smoothing ('ma' for moving average, 'ema' for exponential moving average).
        smooth_param (int or float): Smoothing parameter (window size for 'ma', alpha for 'ema').
    """
    assert history is not None, "History data must be provided."
    assert label is not None, "Label must be provided."
    assert smooth_type in ["ma", "ema"], "smooth_type must be 'ma' or 'ema'"

    plt.figure(figsize=(10, 6))

    x = np.arange(len(history))
    history = np.array(history)

    if smooth_param and smooth_param != 1:
        if smooth_type == "ma":
            assert isinstance(smooth_param, int) and smooth_param > 1, "For 'ma', smooth_param must be an integer > 1"
            kernel = np.ones(smooth_param) / smooth_param
            padded = np.pad(history, (smooth_param // 2, smooth_param - 1 - smooth_param // 2), mode="edge")
            smoothed = np.convolve(padded, kernel, mode="valid")
        elif smooth_type == "ema":
            assert (
                isinstance(smooth_param, float) and 0 < smooth_param < 1
            ), "For 'ema', smooth_param must be a float in (0, 1)"
            alpha = float(smooth_param)
            smoothed = [history[0]]
            for val in history[1:]:
                smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
            smoothed = np.array(smoothed)
        plt.plot(x, history, color="#87CEEB", label=label, alpha=0.6)  # original line
        plt.plot(x, smoothed, color="#0051CB", linewidth=1.5, label=f"Smoothed")  # smoothed line
    else:
        plt.plot(x, smoothed, color="#0051CB", linewidth=1.5, label=label)  # original line

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    # Example usage
    data = np.random.randn(10000).cumsum()
    noise = np.random.randn(10000) * 10
    data += noise
    plot_training_history(data, label="Cumulative Sum", save_path=None, smooth_type="ma", smooth_param=50)
