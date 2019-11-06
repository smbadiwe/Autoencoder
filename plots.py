import matplotlib.pyplot as plt
from config import epochs, save_folder
from os import path
from glob import glob
from imageio import imread, imsave
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
    plt.plot(epochs_val, get_loss_values(loss_fn), label=label)
    plt.title('Reconstruction Error vs Epoch - Using ' + label)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel(f"error ({label})")
    plt.show()


def visualize_reconstructions(image_num=0):

    nrows, ncols, i = 5, 5, 1
    fig = plt.figure(figsize=(30, 30))

    def pre_plot(file_, index):
        img_ = imread(file_)
        fig.add_subplot(nrows, ncols, index)
        ax1_ = plt.gca().axes
        ax1_.set_frame_on(False)
        ax1_.get_xaxis().set_ticklabels([])
        ax1_.get_yaxis().set_ticklabels([])
        ax1_.get_xaxis().set_ticks([])
        ax1_.get_yaxis().set_ticks([])
        return img_, ax1_
    # print(plt.rcParams)
    # settings = {'font.size': 10, 'axes.labelsize': 48, 'figure.frameon': False}

    plt.rcParams.update({'axes.labelsize': 48})
    orig_file = f"images/{image_num}_image.png"
    img, ax1 = pre_plot(orig_file, i)

    ax1.get_xaxis().set_visible(False)
    plt.ylabel("Original")
    plt.imshow(img, interpolation='nearest')
    i = 6
    for lfn in ["mse", "rmse", "idiv", "dis"]:
        for fn in [0, 20, 60, 115, None]:
            if fn == 0 or lfn == "dis":
                plt.rcParams.update({'font.size': 48})
            img_file = glob(f"images/{image_num}_out_{lfn}_{f'_{fn}' if fn is not None else 'BEST'}_*.png")[0]

            img, ax1 = pre_plot(img_file, i)
            if fn == 0:
                plt.ylabel(lfn)
            else:
                ax1.get_yaxis().set_visible(False)
            if lfn == "dis":
                plt.xlabel(f"epoch {fn if fn is not None else 'BEST'}")
            else:
                ax1.get_xaxis().set_visible(False)
            plt.imshow(img, interpolation='nearest')
            i += 1
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()
    plt.savefig("test.png")  # , dpi=300, bbox_inches='tight',


if __name__ == "__main__":
    # visualize("idiv", r"$\mathit{I}$-divergence")
    # visualize("rmse", r"RMSE")
    # visualize("mse", r"MSE")
    # visualize("dis", r"$d_{IS}$")
    visualize_reconstructions()
