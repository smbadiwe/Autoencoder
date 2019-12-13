from os import makedirs, path
import sys
from config import *
from glob import glob


def get_shrink_value_and_loss_from_input(default_sh=0, default_lf="rmse"):
    argv = sys.argv
    try:
        for k, v in enumerate(argv):
            if v == '-sh':
                default_sh = int(argv[k + 1])
            if v == '-lf':
                default_lf = argv[k + 1]
    except Exception as ex:
        print(ex, f"Setting shrink value [sh] to {default_sh} and loss function [lf] to {default_lf}")
    if default_lf not in {"idiv", "mse", "rmse", "dis"}:
        raise Exception(f"Invalid loss. Values must be 'idiv', 'mse', 'rmse' or 'dis'.")
    print(f"Using shrink value [sh]: {default_sh} and loss function [lf]: {default_lf}")
    return default_sh, default_lf


def ensure_folder(folder):
    if not path.exists(folder):
        makedirs(folder)


def adjust_learning_rate(optimizer, shrink_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


class ExpoAverageMeter(object):
    # Exponential Weighted Average Meter
    def __init__(self, beta=0.9):
        self.reset(beta)

    def reset(self, beta):
        self.beta = beta
        self.val = 0
        self.avg = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.avg = self.beta * self.avg + (1 - self.beta) * self.val


def get_demo_output_image(loss_fn, shrink, sample_idx, ck):
    prepend = f"shrink-{shrink}_" if shrink else ""
    return f'images/{prepend}{sample_idx}_out_{loss_fn}_{ck}.png'


def get_checkpoint_folder(loss_fn, shrink):
    if shrink:
        return path.join(save_folder, f"shrink{shrink}", loss_fn)
    return path.join(save_folder, loss_fn)


def get_fn_list():
    if epochs == 30:
        fn_list = [0, 5, 15, 25, None]
    else:
        fn_list = [0, 20, 60, 115, None]
    return fn_list


def get_last_saved_checkpoint_number(loss_fn, shrink):
    directory = get_checkpoint_folder(loss_fn=loss_fn, shrink=shrink)
    ensure_folder(directory)
    fs = glob(path.join(directory, f"checkpoint_*.tar"))
    return len(fs)  # checkpoint is numbered from 0 to len - 1. No gaps.
    # if len(fs) == 0:
    #     return 0
    # fs = [int(path.basename(ff).split("_")[1]) for ff in fs]
    # return max(fs)


def save_checkpoint(epoch, model, optimizer, loss_fn, val_loss, is_best, shrink):
    directory = get_checkpoint_folder(loss_fn=loss_fn, shrink=shrink)
    ensure_folder(directory)
    state = {'model': model,
             'optimizer': optimizer}
    filename = path.join(directory, 'checkpoint_{0}_{1:.3f}.tar'.format(epoch, val_loss))
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, path.join(directory, 'BEST_checkpoint.tar'))
