import matplotlib.pyplot as plt
from config import epochs, save_folder
from os import path
from glob import glob
from imageio import imread
from utils import get_checkpoint_folder, get_demo_output_image


def get_loss_values(loss_fn, shrink):
    losses = [0.0] * epochs
    folder = get_checkpoint_folder(loss_fn=loss_fn, shrink=shrink)
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


def visualize(loss_fn, shrink, label=None):
    if not label:
        label = loss_fn
    shr_text = f" [SHR-{shrink}]" if shrink else ""
    epochs_val = range(epochs)
    plt.figure()
    plt.plot(epochs_val, get_loss_values(loss_fn=loss_fn, shrink=shrink), label=label)
    plt.title(f'Reconstruction Error vs Epoch - Using {label}{shr_text}')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel(f"error ({label})")
    plt.show()


def pre_plot(fig, nrows, ncols, file_, index):
    img_ = imread(file_)
    fig.add_subplot(nrows, ncols, index)
    ax1_ = plt.gca().axes
    ax1_.set_frame_on(False)
    ax1_.get_xaxis().set_ticklabels([])
    ax1_.get_yaxis().set_ticklabels([])
    ax1_.get_xaxis().set_ticks([])
    ax1_.get_yaxis().set_ticks([])
    return img_, ax1_


def visualize_x_and_x_hat(loss_fn, image_num=0, shrink=0):
    ck = 'BEST_*'
    img_f = get_demo_output_image(loss_fn=loss_fn, shrink=shrink, sample_idx=image_num, ck=ck)
    img_file = glob(img_f)
    if len(img_file) == 0:
        print(img_f, "does not exist. Aborting.")
        return None

    nrows, ncols = 2, 1
    fig = plt.figure(figsize=(30, 30))
    plt.rcParams.update({'axes.labelsize': 128})

    orig_file = f"images/{image_num}_image.png"
    img, ax1 = pre_plot(fig=fig, nrows=nrows, ncols=ncols, file_=orig_file, index=1)
    ax1.get_xaxis().set_visible(False)
    plt.ylabel("X")
    plt.imshow(img, interpolation='nearest')

    img, ax1 = pre_plot(fig=fig, nrows=nrows, ncols=ncols, file_=img_file[0], index=2)
    ax1.get_xaxis().set_visible(False)
    plt.ylabel(r"$\hat{X}$")
    plt.imshow(img, interpolation='nearest')

    plt.show()


def visualize_reconstructions(image_num=0, shrink=0):

    nrows, ncols, i = 5, 5, 1
    fig = plt.figure(figsize=(30, 30))

    # print(plt.rcParams)
    # settings = {'font.size': 10, 'axes.labelsize': 48, 'figure.frameon': False}

    plt.rcParams.update({'axes.labelsize': 48})
    orig_file = f"images/{image_num}_image.png"
    img, ax1 = pre_plot(fig=fig, nrows=nrows, ncols=ncols, file_=orig_file, index=i)

    ax1.get_xaxis().set_visible(False)
    plt.ylabel("Original")
    plt.imshow(img, interpolation='nearest')
    i = 6
    for lfn in ["mse", "rmse", "idiv", "dis"]:
        for fn in [0, 20, 60, 115, None]:
            # img_file = glob(f"images/{image_num}_out_{lfn}_{f'_{fn}' if fn is not None else 'BEST'}_*.png")[0]
            ck = f'_{fn}_*' if fn is not None else 'BEST_*'
            img_f = get_demo_output_image(loss_fn=lfn, shrink=shrink, sample_idx=image_num, ck=ck)
            img_file = glob(img_f)
            if len(img_file) == 0:
                print(img_f, "does not exist. Moving on")
                continue

            if fn == 0 or lfn == "dis":
                plt.rcParams.update({'font.size': 48})

            img, ax1 = pre_plot(fig=fig, nrows=nrows, ncols=ncols, file_=img_file[0], index=i)
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
    plt.savefig(f"sample-{image_num}_shr-{shrink}_outs.png")  # , dpi=300, bbox_inches='tight',


if __name__ == "__main__":
    # visualize("idiv", r"$\mathit{I}$-divergence")
    # visualize("rmse", r"RMSE")
    # visualize("mse", r"MSE")
    # visualize("dis", r"$d_{IS}$")
    for ir in range(0, 3):
        # visualize_reconstructions(image_num=ir, shrink=0)
        visualize_x_and_x_hat(loss_fn="mse", image_num=ir, shrink=1)
