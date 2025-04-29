import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

_cmap = colors.ListedColormap([
    "#000000",   # Black
    "#0074D9",   # Blue
    "#FF4136",   # Red
    "#2ECC40",   # Green
    "#FFDC00",   # Yellow
    "#AAAAAA",   # Grey
    "#F012BE",   # Fuchsia
    "#FF851B",   # Orange
    "#7FDBFF",   # Teal
    "#870C25"    # Brown
])

_grid_color = "#555555"

_norm = colors.Normalize(vmin=0, vmax=9)

def draw_grid(ax, matrix, title=""):
    """Draw a single grid on the given axis with grid lines."""
    ax.imshow(matrix, cmap=_cmap, norm=_norm)
    ax.set_title(title)
    ax.axis('off')
    
    rows, cols = matrix.shape
    # Draw horizontal grid lines
    for row in range(0, rows + 1):
        ax.plot([0-0.5, cols-0.5], [row-0.5, row-0.5], color=_grid_color, lw=1)
    # Draw vertical grid lines
    for col in range(0, cols + 1):
        ax.plot([col-0.5, col-0.5], [0-0.5, rows-0.5], color=_grid_color, lw=1)

def plot_task(id, task, test, title=""):
    rows = len(task)
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(10,7))

    title = title if title else f"Task {id}"
    fig.suptitle(title, fontsize=16, y=1.05)

    axes = axes.flatten()
    for i, pair in enumerate(task):
        mat_inp = np.array(pair['input'])
        mat_out = np.array(pair['output'])

        # Draw input grid
        draw_grid(axes[i * cols], mat_inp, f"Training Input Grid {i+1}")
        
        # Draw output grid
        draw_grid(axes[i * cols + 1], mat_out, f"Training Output Grid {i+1}")

    # Draw test input grid
    mat_inp = np.array(test['input'])
    draw_grid(axes[cols - 1], mat_inp, "Test Input Grid")

    # Hide unused axes
    for i in range(1, rows):
        axes[i*cols + cols-1].axis('off')

    plt.tight_layout()
    plt.show()