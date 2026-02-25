'''
 Author: Dr. Souradeep Dutta
 
 This code is provided for demonstration and instructional purposes
 for CPEN 355 at the University of British Columbia (UBC).
 
 Course website:
 https://souradeep-dutta-01.github.io/ubc-cpen-355-website
 
'''
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def plot_results(X, y, w_true, b_true, w_learned, b_learned, history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    pos = y > 0
    neg = y < 0
    ax.scatter(X[pos, 0], X[pos, 1], c="tab:blue", label="y=+1")
    ax.scatter(X[neg, 0], X[neg, 1], c="tab:orange", label="y=-1")

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    xs = np.array([x_min, x_max])

    if w_true is not None and b_true is not None and abs(w_true[1]) > 1e-8:
        ys_true = -(w_true[0] * xs + b_true) / w_true[1]
        ax.plot(xs, ys_true, "k--", label="true separator")

    if w_learned is not None and b_learned is not None and abs(w_learned[1]) > 1e-8:
        ys_learned = -(w_learned[0] * xs + b_learned) / w_learned[1]
        ax.plot(xs, ys_learned, "r-", label="learned separator")

    ax.set_title("Dataset and Separators")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()

    ax2 = axes[1]
    ax2.plot(history, color="tab:green")
    ax2.set_title("Training Curve (Avg Hinge Loss)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")

    fig.tight_layout()

    output_path = Path("Figures") / "training-curve.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)


def animate_training(X, y, w_true, b_true, param_trace, output_path=None):
    if not param_trace:
        raise ValueError("param_trace must be non-empty.")

    first = param_trace[0]
    if isinstance(first, (float, int)):
        fig, ax = plt.subplots(figsize=(6, 5))
        line, = ax.plot([], [], color="tab:green")
        title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

        ax.set_title("Training Curve (Animated)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        def init():
            line.set_data([], [])
            title.set_text("")
            return line, title

        def update(frame):
            xs = np.arange(frame + 1)
            ys = np.array(param_trace[: frame + 1])
            line.set_data(xs, ys)
            title.set_text(f"Epoch {frame + 1} | Loss {param_trace[frame]:.4f}")
            ax.relim()
            ax.autoscale_view()
            return line, title

        ani = animation.FuncAnimation(
            fig, update, frames=len(param_trace), init_func=init, blit=True
        )
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        pos = y > 0
        neg = y < 0
        ax.scatter(X[pos, 0], X[pos, 1], c="tab:blue", label="y=+1")
        ax.scatter(X[neg, 0], X[neg, 1], c="tab:orange", label="y=-1")

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        xs = np.array([x_min, x_max])

        if w_true is not None:
            true_b = 0.0 if b_true is None else b_true
        else:
            true_b = None

        if w_true is not None and true_b is not None and abs(w_true[1]) > 1e-8:
            ys_true = -(w_true[0] * xs + true_b) / w_true[1]
            ax.plot(xs, ys_true, "k--", label="true separator")

        learned_line, = ax.plot([], [], "r-", label="learned separator")
        title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

        ax.set_title("Learning Progress")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend()

        def init():
            learned_line.set_data([], [])
            title.set_text("")
            return learned_line, title

        total_updates = len(param_trace)
        target_frames = max(50, total_updates)
        repeat = max(1, target_frames // total_updates)
        frame_indices = [i for i in range(total_updates) for _ in range(repeat)]

        def update(frame):
            idx = frame_indices[frame]
            w1, w2, b, loss = param_trace[idx]
            if abs(w2) > 1e-8:
                ys = -(w1 * xs + b) / w2
                learned_line.set_data(xs, ys)
            else:
                learned_line.set_data([], [])
            title.set_text(f"Update {idx + 1}/{total_updates} | Mistake {loss:.4f}")
            return learned_line, title

        ani = animation.FuncAnimation(
            fig, update, frames=len(frame_indices), init_func=init, blit=True
        )

    if output_path is None:
        output_path = Path("Figures") / "training-progress.mp4"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=8)
    ani.save(output_path, writer=writer)
    plt.close(fig)


def plot_decision_regions(fn, xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), samples=800, seed=0):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(xlim[0], xlim[1], size=samples)
    ys = rng.uniform(ylim[0], ylim[1], size=samples)
    values = np.array([fn(x, y) for x, y in zip(xs, ys)], dtype=float)

    pos = values > 0
    neg = values < 0

    plt.figure(figsize=(6, 5))
    plt.scatter(xs[pos], ys[pos], s=8, c="red", alpha=0.3, label="f(x) > 0")
    plt.scatter(xs[neg], ys[neg], s=8, c="blue", alpha=0.3, label="f(x) < 0")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Regions (Random Sampling)")
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    output_path = Path("Figures") / "decision_regions.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)


def animate_decision_boundaries(
    X,
    y,
    w_trace,
    feature_fn=None,
    score_fn=None,
    output_path="Figures/decision_boundaries.mp4",
    grid_size=150,
):
    if not w_trace:
        raise ValueError("w_trace must be non-empty.")

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    pos = y > 0
    neg = y < 0
    ax.scatter(X[pos, 0], X[pos, 1], c="tab:blue", label="y=+1", s=15)
    ax.scatter(X[neg, 0], X[neg, 1], c="tab:orange", label="y=-1", s=15)
    ax.set_title("Decision Boundary Over Time")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()

    region = np.zeros((grid_size, grid_size), dtype=float)
    im = ax.imshow(
        region,
        origin="lower",
        extent=(x_min, x_max, y_min, y_max),
        cmap="coolwarm",
        alpha=0.3,
        vmin=-1,
        vmax=1,
    )
    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def scores_for_w(w):
        if score_fn is not None:
            scores = np.array([score_fn(p, w) for p in points], dtype=float)
        elif feature_fn is None:
            scores = points @ w
        else:
            scores = np.array([np.dot(w, feature_fn(p)) for p in points], dtype=float)
        return scores.reshape(grid_size, grid_size)

    def init():
        im.set_data(region)
        title.set_text("")
        return im, title

    def update(frame):
        w = w_trace[frame]
        scores = scores_for_w(w)
        region = np.where(scores > 0.0, 1.0, -1.0)
        im.set_data(region)
        title.set_text(f"Update {frame + 1}/{len(w_trace)}")
        return im, title

    ani = animation.FuncAnimation(
        fig, update, frames=len(w_trace), init_func=init, blit=True
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=8)
    ani.save(output_path, writer=writer)
    plt.close(fig)
