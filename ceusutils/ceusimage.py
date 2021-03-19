import glob
import re
import os
import numpy as np
import sys
from pathlib import Path

from tensorflow.python.keras.preprocessing.image_dataset import paths_and_labels_to_dataset
from tensorflow.python.keras.preprocessing import dataset_utils
from tensorflow.python.keras.layers.preprocessing import image_preprocessing

WHITELIST_FORMATS = ('.bmp', '.gif', '.jpeg', '.jpg', '.png')

class CeusImagesGenerator:

    def __init__(self, path, one_out = None, class_disease = ''):
        try:
            if not os.path.exists(path):
                raise FileExistsError
            self._path = path
            self._patients_set = set()
            self._classes = {}
            self._patient_id = one_out
            self._training_records = None
            self._validation_records = None
            if not class_disease:
                raise ValueError
            self._class_disease = class_disease

        except FileExistsError:
            print("Path is not valid...")
            sys.exit(1)
        except ValueError:
            print('A disease class is required...')
            sys.exit(1)

    @property
    def classes(self):
        """
        :return: a dictionary with classes from ceus dataset folder
        """
        if not self._classes:
            for idx, item in enumerate(glob.glob(self._path + '/*')):
                self._classes[os.path.basename(item)] = idx
        return self._classes

    @staticmethod
    def patients_sets(dir_data):
        """
        :param dir_data: absolute or relative path of dataset
        :return: set of patients identified in ceus dataset folder
        """
        patients_dict = {}
        regex = re.compile(r'\d+')

        fullpath = str(Path(dir_data))
        fullpath += '/*'
        for subfolder in glob.glob(fullpath):
            patient_set = set()
            for item in glob.glob(subfolder + '/*.jpg'):
                lst = [int(x) for x in regex.findall(os.path.basename(item))]
                patient_set.add(lst[0])
            patients_dict[os.path.basename(subfolder)] = list(patient_set)

        return patients_dict

    def get_files_by_patient(self):
        """
        Validation records

        :return: list of filenames corresponding to provided patient
        """
        
        # return list only for required id instead of ids that start with digit id
        path_disease = os.path.join(self._path, self._class_disease)
        patient_file_pattern = f'{path_disease}\\{self._class_disease}{str(self._patient_id)}[A-Za-z]*.jpg'
        return [item for item in glob.glob(patient_file_pattern)]


    def get_files_without_patient(self):
        """
        Train records
        :return: list of filenames without files for provided patient
        """

        # Modified by Mihai Andreescu 15/12/2020
        # return list only for required id instead of ids that start with digit id
        excl_set = set(self.get_files_by_patient())
        patients = set(glob.glob(self._path +'/*/*.jpg')) - excl_set

        result = list(patients)
        self._training_records = len(result)

        return result

    def image_dataset_from_files(self, label_mode='int',
                                 color_mode='rgb',
                                 batch_size=32,
                                 image_size=(256, 256),
                                 shuffle=True,
                                 seed=None,
                                 validation_split=None,
                                 subset=None,
                                 interpolation='bilinear'):

        if label_mode not in {'int', 'categorical', 'binary', None}:
            raise ValueError(
                '`label_mode` argument must be one of "int", "categorical", "binary", '
                'or None. Received: %s' % (label_mode,))
        if color_mode == 'rgb':
            num_channels = 3
        elif color_mode == 'rgba':
            num_channels = 4
        elif color_mode == 'grayscale':
            num_channels = 1
        else:
            raise ValueError(
                '`color_mode` must be one of {"rbg", "rgba", "grayscale"}. '
                'Received: %s' % (color_mode,))

        interpolation = image_preprocessing.get_interpolation(interpolation)
        if validation_split:
            dataset_utils.check_validation_split_arg(
                validation_split, subset, shuffle, seed)

        if seed is None:
            seed = np.random.randint(1e6)
        image_paths, labels, class_names = self.index_files(shuffle=shuffle, seed=seed, subset=subset)
        if subset == 'training':
            print("Remaining pictures", len(image_paths))
        else:
            print("Excluded pictures", len(image_paths))

        if label_mode == 'binary' and len(class_names) != 2:
            raise ValueError(
                'When passing `label_mode="binary", there must exactly 2 classes. '
                'Found the following classes: %s' % (class_names,))

        image_paths, labels = dataset_utils.get_training_or_validation_split(
            image_paths, labels, validation_split, subset)

        dataset = paths_and_labels_to_dataset(
            image_paths=image_paths,
            image_size=image_size,
            num_channels=num_channels,
            labels=labels,
            label_mode=label_mode,
            num_classes=len(class_names),
            interpolation=interpolation)
        if shuffle:
            # Shuffle locally at each iteration
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
        dataset = dataset.batch(batch_size)
        # Users may need to reference `class_names`.
        dataset.class_names = class_names
        return dataset

    def index_files(self, shuffle=True, seed=None, subset = None):

        labels_list = []
        file_paths = self.get_files_without_patient() if subset == 'training' \
            else self.get_files_by_patient()

        for file_path in file_paths:
            labels_list.append(self.classes[os.path.basename(os.path.dirname(file_path))])

        if shuffle:
            # Shuffle globally to erase macro-structure
            if seed is None:
                seed = np.random.randint(1e6)
            rng = np.random.RandomState(seed)
            rng.shuffle(file_paths)
            rng = np.random.RandomState(seed)
            rng.shuffle(labels_list)
        return file_paths, labels_list, self.classes.keys()

if __name__ == '__main__':

    data_dir = 'E:/MY/My Databases/MEDICAL/CEUS/UMF/DBV50LEZIUNI'

    # get ids of patients for each class
    p_dict = CeusImagesGenerator.patients_sets(data_dir)
    for item in p_dict.items():
        print(item)

    # create generator object
    datagen = CeusImagesGenerator(data_dir, one_out=1, class_disease='FNH')

    print(len(datagen.get_files_by_patient()))
    print(len(datagen.get_files_without_patient()))

    batch_size = 16
    img_height = 180
    img_width = 180
    ds_training = datagen.image_dataset_from_files(
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    print(len(ds_training) * batch_size)



