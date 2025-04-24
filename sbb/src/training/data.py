import os
import glob

from tqdm import tqdm
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf


class SBBDataLoader:
    @staticmethod
    def load_im_from_path(im_path: str, im_size: int) -> np.ndarray:
        """_summary_
        loads the image from a given path
        Args:
            im_path (str): _description_
            im_size (int): _description_

        Returns:
            np.ndarray: _description_
        """
        raw_im = cv2.imread(im_path, 0)
        resized = cv2.resize(raw_im, (im_size, im_size))
        preprocessed = (resized / 255.).astype(np.float32)
        return np.expand_dims(preprocessed, axis=-1)
    
    @staticmethod
    def create_generators_softmax(src_path: str, im_size: int, val_split=0.10, random_seed=42):
        """_summary_
        creates training and validation generator from the provided dataset
        the directory has to be organized in the following way:
        - root
            -> bridge
                -> 0.png
                -> 1.png
                -> .
                -> .
            -> non_bridge
                -> 0.png
                -> 1.png
                -> .
                -> .
        Args:
            src_path (str): root path of the directory in which the dataset is stored
            im_size (int): size of the image which is read
        """
        BATCH_SIZE = 128
        SHUFFLE_BUFFER_SIZE = 100
        
        bridge_dir = os.path.join(src_path, 'Bridge')
        non_bridge_dir = os.path.join(src_path, 'Non-Bridge')
        
        bridge_paths = glob.glob(os.path.join(bridge_dir, '*.bmp'))
        non_bridge_paths = glob.glob(os.path.join(non_bridge_dir, '*.bmp'))
        
        print('[INFO] found {0} bridge images'.format(len(bridge_paths)))
        print('[INFO] found {0} non_bridge images'.format(len(non_bridge_paths)))
        n_bridge_repeats = len(non_bridge_paths) // len(bridge_paths)
        print('[INFO] balancing bridge and non bridges')
        bridge_paths = n_bridge_repeats * bridge_paths
        print('[INFO] adjusted for {0} bridge images'.format(len(bridge_paths)))
        print('[INFO] adjusted for {0} non_bridge images'.format(len(non_bridge_paths)))
        
        x_bridge = [SBBDataLoader.load_im_from_path(x, im_size) for x in tqdm(bridge_paths)]
        x_non_bridge = [SBBDataLoader.load_im_from_path(x, im_size) for x in tqdm(non_bridge_paths)]

        def create_sbb_entry(index: int):
            tmp = np.zeros(2)
            tmp[index] = 1
            return tmp
        
        # y_bridge = list(np.ones(len(bridge_paths)))
        # y_non_bridge = list(np.zeros(len(non_bridge_paths)))
        y_bridge = [create_sbb_entry(index=1) for x in range(len(bridge_paths))]
        y_non_bridge = [create_sbb_entry(index=0) for x in range(len(non_bridge_paths))]

        x = np.array(x_bridge + x_non_bridge)
        y = np.array(y_bridge + y_non_bridge)

        print('[INFO] y_bridge_shape', y.shape)

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_split) # , random_state=42)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        
        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        test_dataset = test_dataset.batch(BATCH_SIZE)

        return train_dataset, test_dataset


    @staticmethod
    def create_generators(src_path: str, im_size: int, val_split=0.10, random_seed=42):
        """_summary_
        creates training and validation generator from the provided dataset
        the directory has to be organized in the following way:
        - root
            -> bridge
                -> 0.png
                -> 1.png
                -> .
                -> .
            -> non_bridge
                -> 0.png
                -> 1.png
                -> .
                -> .
        Args:
            src_path (str): root path of the directory in which the dataset is stored
            im_size (int): size of the image which is read
        """
        BATCH_SIZE = 128
        SHUFFLE_BUFFER_SIZE = 100
        
        bridge_dir = os.path.join(src_path, 'Bridge')
        non_bridge_dir = os.path.join(src_path, 'Non-Bridge')
        
        bridge_paths = glob.glob(os.path.join(bridge_dir, '*.bmp'))
        non_bridge_paths = glob.glob(os.path.join(non_bridge_dir, '*.bmp'))
        
        print('[INFO] found {0} bridge images'.format(len(bridge_paths)))
        print('[INFO] found {0} non_bridge images'.format(len(non_bridge_paths)))
        n_bridge_repeats = len(non_bridge_paths) // len(bridge_paths)
        print('[INFO] balancing bridge and non bridges')
        bridge_paths = n_bridge_repeats * bridge_paths
        print('[INFO] adjusted for {0} bridge images'.format(len(bridge_paths)))
        print('[INFO] adjusted for {0} non_bridge images'.format(len(non_bridge_paths)))
        
        x_bridge = [SBBDataLoader.load_im_from_path(x, im_size) for x in bridge_paths]
        x_non_bridge = [SBBDataLoader.load_im_from_path(x, im_size) for x in non_bridge_paths]
        
        y_bridge = list(np.ones(len(bridge_paths)))
        y_non_bridge = list(np.zeros(len(non_bridge_paths)))
        
        x = np.array(x_bridge + x_non_bridge)
        y = np.array(y_bridge + y_non_bridge)
        
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_split) # , random_state=42)
        
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        
        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        test_dataset = test_dataset.batch(BATCH_SIZE)

        
        return train_dataset, test_dataset
