import matplotlib.pyplot as plt
from config import epochs, save_folder
from os import path
from glob import glob
import numpy as np


def get_loss_values(loss_fn):
    losses = [0.0] * epochs
    folder = path.join(save_folder, loss_fn)
    try:
        for i in range(epochs):
            fs = glob(path.join(folder, f"checkpoint_{i}_*.tar"))
            assert len(fs) == 1, f"{len(fs)} file(s) exist for checkpoint {i} [loss: {loss_fn}]"
            loss = path.basename(fs[0]).replace(f"checkpoint_{i}_", "").replace(".tar", "")
            losses[i] = float(loss)
    except Exception as ex:
        print(f"Failed reading files or values from folder: {folder}...")
        print(ex)

    return losses


def visualize_all():
    epochs_val = range(epochs)

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('error', color=color)
    ax1.plot(epochs_val, get_loss_values("mse"), label='MSE')
    ax1.plot(epochs_val, get_loss_values("rmse"), label='RMSE')
    ax1.plot(epochs_val, get_loss_values("idiv"), label='I-divergence')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(r'error ($d_{IS}$)', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs_val, get_loss_values("dis"), label=r'$d_{IS}$')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')
    ax2.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.show()


def visualize(loss_fn, label=None):
    if not label:
        label = loss_fn
    epochs_val = range(epochs)
    plt.figure()
    if loss_fn == "dis":
        plt.plot(epochs_val, [np.log(d) for d in get_loss_values(loss_fn)], label=label)
        # plt.yscale("log")
    else:
        plt.plot(epochs_val, get_loss_values(loss_fn), label=label)
    plt.title('Reconstruction Error vs Epoch - Using ' + label)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel(f"error ({label})")
    plt.show()


if __name__ == "__main__":
    visualize("idiv", r"$\mathit{I}$-divergence")
    visualize("rmse", r"RMSE")
    visualize("mse", r"MSE")
    visualize("dis", r"$d_{IS}$")
