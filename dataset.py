import numpy as np
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import ssl
from typing import Tuple, Dict
import random
from tqdm import tqdm


class Dataset:
    def __init__(self) -> None:
        self.dataset_dtype: str = "float32"
        self.train_data_by_class: Dict = {}
        self.test_data_by_class: Dict = {}

        self.training_dataset: np.ndarray = None
        self.training_labels: np.ndarray = None
        self.validation_dataset: np.ndarray = None
        self.validation_labels: np.ndarray = None
        self.testing_dataset: np.ndarray = None
        self.testing_labels: np.ndarray = None

        self.load_datasets()

    def load_datasets(self) -> None:
        """
        Loads training, validation and testing dataset from CIFAR-10.
        """
        print("Loading CIFAR-10 dataset...")
        ssl._create_default_https_context = ssl._create_unverified_context

        # Load CIFAR-10 dataset
        (training_dataset, training_labels), (testing_dataset, testing_labels) = (
            cifar10.load_data()
        )

        # Normalize CIFAR-10 dataset and one-hot encode labels
        training_dataset = training_dataset.astype(self.dataset_dtype) / 255
        testing_dataset = testing_dataset.astype(self.dataset_dtype) / 255
        training_labels = to_categorical(training_labels, 10)
        testing_labels = to_categorical(testing_labels, 10)

        # Generate paired datasets
        self.train_data_by_class = self.separate_data_by_class(
            training_dataset, training_labels
        )
        self.test_data_by_class = self.separate_data_by_class(
            testing_dataset, testing_labels
        )
        paired_train_set, paired_train_labels = self.generate_paired_data(True)
        paired_test_set, paired_test_labels = self.generate_paired_data(False)

        # Use testing dataset as validation dataset
        self.training_dataset = paired_train_set
        self.training_labels = paired_train_labels
        self.validation_dataset = paired_test_set
        self.validation_labels = paired_test_labels
        self.testing_dataset = paired_test_set
        self.testing_labels = paired_test_labels

    def separate_data_by_class(self, dataset: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Generates a dictionary of class to img key-value pairs.
        """
        print("Separating out data by class...")
        category_dict = {}

        if len(labels.shape) > 1 and labels.shape[1] > 1:
            labels = np.argmax(labels, axis=1)

        categories = np.unique(labels)
        for category in categories:
            category_dict[category] = dataset[labels == category]

        return category_dict

    def generate_paired_data(self, train: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates pairs of genuine/fake images from dataset.
        """
        print(f"Generating paired data: train is {train}...")
        paired_imgs = []
        paired_labels = []
        if train:
            dataset = self.train_data_by_class
        else:
            dataset = self.test_data_by_class

        output_categories = dataset.keys()

        for category in tqdm(output_categories):
            imgs = dataset[category]
            if len(imgs) % 2 != 0:
                imgs = imgs[:-1]

            # Sequential pairs
            real_img_pairs = [(imgs[i], imgs[i + 1]) for i in range(0, len(imgs), 2)]
            real_label_pairs = [1] * len(real_img_pairs)
            paired_imgs.extend(real_img_pairs)
            paired_labels.extend(real_label_pairs)

            # Generate equal num fake pairs, paired with randomly chosen class
            fake_set_indices = np.random.choice(
                len(imgs), size=len(real_img_pairs), replace=False
            )
            fake_img_set = imgs[fake_set_indices]
            other_categories = [i for i in output_categories if i != category]
            for img in fake_img_set:
                other_category = random.choice(other_categories)
                other_img = random.choice(dataset[other_category])
                paired_imgs.append((img, other_img))
                paired_labels.append(0)

        # Shuffle pairs
        indices = np.random.permutation(len(paired_imgs))
        paired_imgs = np.array(paired_imgs)[indices]
        paired_labels = np.array(paired_labels)[indices]

        return paired_imgs, paired_labels
