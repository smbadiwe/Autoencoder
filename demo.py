from os import path, listdir
import random

import cv2 as cv
import numpy as np
import torch
from imageio import imread, imsave
from PIL import Image
from config import device, save_folder, imsize
from utils import ensure_folder


def imresize(arr, dim_tuple):
    return np.array(Image.fromarray(arr).resize(size=dim_tuple))


def main(loss_fn="mse", checkpoint_file='BEST_checkpoint', num_test_samples=10):
    """

    :param loss_fn: The loss used. Here's it'll be used as sub-directory to get the actual model.
    :param checkpoint_file: # model checkpoint file name, without directory or extension
    :param num_test_samples:
    :return:
    """
    print(f'checkpoint: {checkpoint_file}')
    # Load model
    checkpoint = torch.load(f"{save_folder}/{loss_fn}/{checkpoint_file}.tar")
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    ensure_folder('images')
    test_path = 'images'
    test_images = [path.join(test_path, f) for f in listdir(test_path) if f.endswith('_image.png')]

    n_test_img = len(test_images)
    if n_test_img < num_test_samples:
        test_path = 'data/test/'
        test_images = [path.join(test_path, f) for f in listdir(test_path) if f.endswith('.jpg')]

        samples = random.sample(test_images, num_test_samples - n_test_img)
        for i, fpath in enumerate(samples):
            # Read images
            img = imread(fpath)
            img = imresize(img, (imsize, imsize))
            fname = f'images/{n_test_img + i}_image.png'
            imsave(fname, img)
            test_images.append(fname)

    imgs = torch.zeros([num_test_samples, 3, imsize, imsize], dtype=torch.float, device=device)

    for i, fpath in enumerate(test_images):
        # Read images
        img = imread(fpath)
        img = imresize(img, (imsize, imsize))
        img = img.transpose(2, 0, 1)
        assert img.shape == (3, imsize, imsize)
        assert np.max(img) <= 255
        imgs[i] = torch.FloatTensor(img / 255.)

    imgs = torch.tensor(imgs)

    with torch.no_grad():
        preds = model(imgs)

    ck = checkpoint_file.replace("checkpoint", "")
    for i in range(num_test_samples):
        out = preds[i]
        out = out.cpu().numpy()
        out = np.transpose(out, (1, 2, 0))
        out = out * 255.
        out = np.clip(out, 0, 255)
        out = out.astype(np.uint8)
        out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
        cv.imwrite(f'images/{i}_out_{loss_fn}_{ck}.png', out)


if __name__ == '__main__':
    main()
