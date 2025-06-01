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

def draw_task(task, task_id="", title="", output_recon=None, output_prediction=None):
    training_pair_count = len(task['train'])
    rows = max(training_pair_count, 4)
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(10,7))
    title = title if title else f"Task {task_id}"
    fig.suptitle(title, fontsize=16, y=1.05)

    axes = axes.flatten()
    for i, (input, output) in enumerate(task['train']):
        mat_inp = np.array(input)
        mat_out = np.array(output)

        draw_grid(axes[i * cols], mat_inp, f"Training Input Grid {i+1}")
        
        draw_grid(axes[i * cols + 1], mat_out, f"Training Output Grid {i+1}")

    test_input, test_output = task['test'][0]
    mat_inp = np.array(test_input)
    draw_grid(axes[cols - 1], mat_inp, "Test Input Grid")


    mat_out = np.array(test_output)
    draw_grid(axes[2*cols - 1], mat_out, "Test Output Grid")
    
    used_output_rows = 2

    if output_recon is not None:
        mat_recon = np.array(output_recon)
        used_output_rows += 1
        draw_grid(axes[used_output_rows*cols - 1], mat_recon, "Reconstructed Test Output Grid")

    if output_prediction is not None:
        mat_pred = np.array(output_prediction)
        used_output_rows += 1
        draw_grid(axes[used_output_rows*cols - 1], mat_pred, "Predicted Output Grid")
        
    for i in range(used_output_rows + 1, rows):
        axes[i*cols -1].axis('off')
    
    for i in range(training_pair_count, rows):
        for j in range(cols - 1):
            axes[j + i*cols].axis('off')

    plt.tight_layout()
    plt.show()


def plot_losses(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
