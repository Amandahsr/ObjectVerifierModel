from keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam
import keras.backend as kb
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from dataset import Dataset


class SiameseModel:
    def __init__(self):
        self.input_img_shape: Tuple[int, int, int] = (32, 32, 3)
        self.activation_function: str = "relu"
        self.weights: str = None
        self.include_top: bool = False
        self.margin: float = 1.5

        self.layer14_dense_units: int = 4096
        self.layer15_dense_units: int = 4096

        # Training params
        self.dataset: Dataset = Dataset()
        self.num_epochs: int = 15
        self.batch_size: int = 512
        self.dropout_rate: float = 0.5
        self.learning_rate: float = 0.0001
        self.optimizer: Adam = Adam(learning_rate=self.learning_rate)
        self.model: Sequential = self.initialize_siamese_model()

        self.train_status: bool = False
        self.shuffle_training_data: bool = True
        self.training_history: Dict = None
        self.test_loss_history: List[float] = []
        self.test_loss: float = None
        self.test_accuracy: float = None
        self.pred_labels: np.ndarray = None
        self.true_labels: np.ndarray = None
        self.test_confusion_matrix: pd.DataFrame = None

    def initialize_vggnet_model(self):
        vggnet_model = VGG16(
            weights=self.weights,
            include_top=self.include_top,
            input_shape=self.input_img_shape,
        )

        for layer in vggnet_model.layers:
            layer.trainable = True

        output = vggnet_model.output
        output = layers.Flatten()(output)
        output = layers.Dense(
            self.layer14_dense_units, activation=self.activation_function
        )(output)
        output = layers.Dropout(self.dropout_rate)(output)
        output = layers.Dense(
            self.layer14_dense_units, activation=self.activation_function
        )(output)

        model = models.Model(inputs=vggnet_model.input, outputs=output)

        return model

    def euclidean_dist(self, vectors: Tuple[np.ndarray, np.ndarray]) -> float:
        """
        Returns euclidean distance between two vectors
        """
        x_vector, y_vector = vectors
        sum_squared = kb.sum(kb.square(x_vector - y_vector), axis=1, keepdims=True)

        return kb.sqrt(kb.maximum(sum_squared, kb.epsilon()))

    def contrastive_loss(self, y_true, y_pred):
        y_true = kb.cast(y_true, self.dataset.dataset_dtype)
        square_pred = kb.square(y_pred)
        margin_square = kb.square(kb.maximum(self.margin - y_pred, 0))

        return kb.mean(y_true * square_pred + (1 - y_true) * margin_square)

    def initialize_siamese_model(self):
        print("Building siamese network...")
        input_a = layers.Input(shape=self.input_img_shape)
        input_b = layers.Input(shape=self.input_img_shape)

        vggnet_model = self.initialize_vggnet_model()

        model_a = vggnet_model(input_a)
        model_b = vggnet_model(input_b)

        eu_dist = layers.Lambda(lambda x: self.euclidean_dist(x))([model_a, model_b])

        siamese_model = models.Model([input_a, input_b], eu_dist)
        siamese_model.compile(
            loss=self.contrastive_loss, optimizer=self.optimizer, metrics=["accuracy"]
        )

        # Check model layers
        print("Model architecture:")
        print(siamese_model.summary())

        return siamese_model

    def train_model(self) -> None:
        """
        Trains model and caches training history.
        """
        print("Training siamese model...")
        training_history = self.model.fit(
            [self.dataset.training_dataset[:, 0], self.dataset.training_dataset[:, 1]],
            self.dataset.training_labels,
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            shuffle=self.shuffle_training_data,
            validation_data=(
                [
                    self.dataset.validation_dataset[:, 0],
                    self.dataset.validation_dataset[:, 1],
                ],
                self.dataset.validation_labels,
            ),
            # callbacks=[Callback()],
            # verbose=0,
        )

        self.train_status = True
        self.training_history = training_history

    def eval_model(self) -> None:
        """
        Returns categorical cross entropy results, predicted and true labels for testing dataset.
        """
        print("Evaluating model")
        if not self.train_status:
            raise ValueError("Model has not been trained.")

        # loss function results
        self.test_loss, self.test_accuracy = self.model.evaluate(
            [self.dataset.testing_dataset[:, 0], self.dataset.testing_dataset[:, 1]],
            self.dataset.testing_labels,
        )

        # Classification results, convert prob to class labels
        predictions = self.model.predict(
            [self.dataset.testing_dataset[:, 0], self.dataset.testing_dataset[:, 1]]
        )

        # Generate confusion matrix
        self.pred_labels = np.argmax(predictions, axis=1)
        self.true_labels = np.argmax(self.dataset.testing_labels, axis=1)
        self.test_confusion_matrix = self.generate_confusion_matrix(
            self.pred_labels, self.true_labels
        )

    def generate_confusion_matrix(
        self, pred_labels: np.ndarray, true_labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Generates a confusion matrix.
        """
        print("Generating confusion matrix")
        matrix = confusion_matrix(true_labels, pred_labels)

        # Categories are indexed in sequential order
        classification_categories = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

        matrix_df = pd.DataFrame(
            matrix, index=classification_categories, columns=classification_categories
        )

        return matrix_df
