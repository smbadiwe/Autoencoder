import matplotlib.pyplot as plt
from config import epochs, save_folder
from os import path, listdir


def get_loss_values(loss_fn):
    losses = [0.0] * epochs
    folder = path.join(save_folder, loss_fn)
    try:
        files = [f for f in listdir(folder) if f.startswith("checkpoint_")]
        assert len(files) == epochs
        files.sort()
        for i in range(epochs):
            loss = files[i].replace(f"checkpoint_{i}_", "").replace(".tar", "")
            losses[i] = float(loss)
    except Exception as ex:
        print(f"Failed reading files or values from folder: {folder}...")
        print(ex)
    return losses


def visualize():
    epochs_val = range(epochs)

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('error', color=color)
    ax1.plot(epochs_val, get_loss_values("mse"), label='MSE')
    ax1.plot(epochs_val, get_loss_values("rmse"), label='RMSE')
    ax1.plot(epochs_val, get_loss_values("idiv"), label='I-divergence')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(r'error ($d_{IS}$)', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs_val, get_loss_values("dis"), label=r'$d_{IS}$')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    # plt.figure()
    # plt.plot(epochs_val, get_loss_values("mse"), label='MSE')
    # plt.plot(epochs_val, get_loss_values("rmse"), label='RMSE')
    # plt.plot(epochs_val, get_loss_values("idiv"), label='I-divergence')
    # plt.title('Reconstruction Error vs Epoch')
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    visualize()
