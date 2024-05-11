from typing import Dict, Optional, Tuple
import flwr as fl
import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.utils import shuffle

# server address = {IP_ADDRESS}:{PORT}
server_address = "10.12.101.102:5050"
classes = ["face1","face2","face3","face4","face5","face6","face7","face8","face9","face10"]
class_labels = {classes: i for i, classes in enumerate(classes)}
number_of_classes = len(classes)
# defining image size
IMAGE_SIZE = (128, 128)

federatedLearningcounts = 4
local_client_epochs = 3
local_client_batch_size = 8

def main() -> None:

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        alpha=1.0,
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=10,
        classifier_activation="softmax"
    )
    # freeze the layers in the base model so they don't get updated
    base_model.trainable = False

    # define classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    # create the final model
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    # compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(
            model.get_weights()),
    )

    # start Flower server (SSL-enabled) for X rounds of federated learning
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=federatedLearningcounts),
        strategy=strategy
    )

def load_dataset():
    # defining the directory with the server's test images. We only use the test images!
    directory = "datasets/dataset_server"
    sub_directories = ["test", "train"]

    loaded_dataset = []
    for sub_directory in sub_directories:
        path = os.path.join(directory, sub_directory)
        images = []
        labels = []

        print("Server dataset loading {}".format(sub_directory))

        for folder in os.listdir(path):
            label = class_labels[folder]

            # iterate through each image in the folder
            for file in os.listdir(os.path.join(path,folder)):
                # get path name of the image
                img_path = os.path.join(os.path.join(path, folder), file)

                try:
                    # open and resize the image
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Skipping corrupted image: {img_path}")
                        continue

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, IMAGE_SIZE)

                    # append the image and its corresponding label to loaded_dataset
                    images.append(image)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')

        loaded_dataset.append((images, labels))

    return loaded_dataset


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # load data and model here to avoid the overhead of doing it in `evaluate` itself
    (training_images, training_labels), (test_images, test_labels) = load_dataset()
    print("[Server] test_images shape:", test_images.shape)
    print("[Server] test_labels shape:", test_labels.shape)

    # the `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        print("======= server round %s/%s evaluate() ===== " %(server_round, federatedLearningcounts))
        # update model with the latest parameters
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
        print("======= server round %s/%s accuracy : %s =======" %(server_round, federatedLearningcounts,accuracy))

        if (server_round == federatedLearningcounts):
            # save the decentralized ML model locally on the server computer
            print("Saving updated model locally..")
            # model.save('saved_models/mobilenetv2.h5')  # save model in .h5 format
            model.save('saved_models/omg.h5')      # save model in SavedModel format

            # test the updated model


        return loss, {"accuracy": accuracy}
    return evaluate

def fit_config(server_round: int):
    # return training configuration dict for each round
    config = {
        "batch_size": local_client_batch_size,
        "local_epochs": local_client_epochs,
    }
    return config

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round."""
    # val_steps = 5 if server_round < 4 else 10
    val_steps = 4
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
