from os import listdir
import random
import cv2 as cv
import numpy as np
from imageio import imread, imsave
from PIL import Image
from utils import *


def imresize(arr, dim_tuple):
    return np.array(Image.fromarray(arr).resize(size=dim_tuple))


def main(loss_fn="rmse", checkpoint_file=None, num_test_samples=5, shrink=0):
    """

    :param loss_fn: The loss used. Here's it'll be used as sub-directory to get the actual model.
    :param checkpoint_file: # model checkpoint: number or file name without directory or extension
    :param num_test_samples:
    :param shrink:
    :return:
    """
    if checkpoint_file is None:
        checkpoint_file = 'BEST_checkpoint'
    folder = get_checkpoint_folder(loss_fn=loss_fn, shrink=shrink)
    try:
        checkpoint_file += 0  # test if param is int
        fs = glob(path.join(folder, f"checkpoint_{checkpoint_file}_*.tar"))
        if len(fs) == 0:
            raise FileNotFoundError(f"Chackpoint {checkpoint_file} model file does not exist")
        if len(fs) > 1:
            raise FileNotFoundError(f"Chackpoint {checkpoint_file} file is ambiguous. Not sure what you're looking for")
        checkpoint_file = fs[0]
    except TypeError:
        checkpoint_file = path.join(folder, f"{checkpoint_file}.tar")
    # print(f'checkpoint: {checkpoint_file}')
    # Load model
    checkpoint = torch.load(checkpoint_file)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    ensure_folder('images')
    test_path = 'images'
    test_images = [path.join(test_path, f) for f in listdir(test_path) if f.endswith('_image.png')]

    n_test_img = len(test_images)
    if n_test_img < num_test_samples:
        print(f"n_test_img: {n_test_img}. num_test_samples: {num_test_samples}")
        test_path = 'data/test/'
        test_images_orig = [path.join(test_path, f) for f in listdir(test_path) if f.endswith('.jpg')]

        samples = random.sample(test_images_orig, num_test_samples - n_test_img)
        for i, fpath in enumerate(samples):
            # Read images
            img = imread(fpath)
            img = imresize(img, (imsize, imsize))
            fname = f'images/{n_test_img + i}_image.png'
            imsave(fname, img)
            test_images.append(fname)

    imgs = torch.zeros([num_test_samples, 3, imsize, imsize], dtype=torch.float, device=device)
    test_images.sort()
    print(f"\ntest images: {test_images}\n")
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

    ck = path.basename(checkpoint_file).replace("checkpoint", "").replace(".tar", "")
    for i in range(num_test_samples):
        out = preds[i]
        out = out.cpu().numpy()
        out = np.transpose(out, (1, 2, 0))
        out = out * 255.
        out = np.clip(out, 0, 255)
        out = out.astype(np.uint8)
        out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
        cv.imwrite(get_demo_output_image(loss_fn=loss_fn, shrink=shrink, sample_idx=i, ck=ck), out)


if __name__ == '__main__':
    # main(loss_fn='dis', checkpoint_file=0, num_test_samples=1)
    sh, lf = get_shrink_value_and_loss_from_input()
    for lfn in ["mse", "rmse", "idiv", "dis"]:
        for fn in [0, 20, 60, 115, None]:
            main(loss_fn=lfn, checkpoint_file=fn, num_test_samples=3, shrink=sh)
            # main(loss_fn=lfn, checkpoint_file=fn, num_test_samples=1)
