import pathlib
import glob
import os
import cv2
from tqdm import tqdm


def main():
    root_path = r'D:\AIML-Image Data\SBB\Data\Version5\Test9\Bridge'
    dst_path = r'D:\AIML-Image Data\SBB\Data\Version5\Test9\png_exports\bridge'

    for im_path in tqdm(glob.glob(os.path.join(root_path, '**/*.bmp'), recursive=True)):
        im = cv2.imread(im_path, 0)
        filename = pathlib.Path(im_path).stem
        dst_im_path = os.path.join(dst_path, '{0}.png'.format(filename))
        im = cv2.resize(im, (56, 56))
        cv2.imwrite(dst_im_path, im)


if __name__ == '__main__':
    main()
