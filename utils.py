from os import makedirs, path
import sys
from config import *


def get_shrink_value_from_input(default=0):
    argv = sys.argv
    try:
        for k, v in enumerate(argv):
            if v == '-sh':
                default = int(argv[k + 1])
    except Exception as ex:
        print(ex, f"Setting shrink value to {default}")
    print(f"Using shrink value: {default}")
    return default


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
        self.reset()

    def reset(self):
        self.beta = 0.9
        self.val = 0
        self.avg = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.avg = self.beta * self.avg + (1 - self.beta) * self.val


def get_checkpoint_folder(loss_fn, shrink):
    if shrink:
        return path.join(save_folder, f"shrink{shrink}", loss_fn)
    return path.join(save_folder, loss_fn)


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
