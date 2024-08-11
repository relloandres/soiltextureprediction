import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import csv
import json
import time
import torchvision


class ModelTrainer:
    AVAILABLE_MODELS = ["TwoConvLayer", "FourConvLayer"]
    TRAINING_MANDATORY_FIELDS = [
        "n_epochs",
        "batch_size",
        "optimizer_name",
        "optimizer_params",
    ]
    SGD_MANDATORY_FIELDS = ["lr", "momentum", "weight_decay"]
    ADAM_MANDATORY_FIELDS = ["lr"]

    def validate_training_info(self, training_info):
        # Validate main fields
        for field in ModelTrainer.TRAINING_MANDATORY_FIELDS:
            if field not in training_info:
                # Raising a ValueError with a custom error message
                raise ValueError("Field {} is missing in model_structure".format(field))

        # Validate optimizer fields
        if training_info["optimizer_name"] == "SGD":
            for field in ModelTrainer.SGD_MANDATORY_FIELDS:
                if field not in training_info["optimizer_params"]:
                    # Raising a ValueError with a custom error message
                    raise ValueError(
                        "Field {} is missing in optimizer_params".format(field)
                    )
        elif training_info["optimizer_name"] == "Adam":
            for field in ModelTrainer.ADAM_MANDATORY_FIELDS:
                if field not in training_info["optimizer_params"]:
                    # Raising a ValueError with a custom error message
                    raise ValueError(
                        "Field {} is missing in optimizer_params".format(field)
                    )
        else:
            raise ValueError(
                "Optimzer {} is not supported".format(training_info["optimizer_name"])
            )

        return True

    def create_optimizer(self, model, training_info):
        # Validate optimizer fields
        if training_info["optimizer_name"] == "SGD":
            return torch.optim.SGD(
                model.parameters(),
                lr=training_info["optimizer_params"]["lr"],
                momentum=training_info["optimizer_params"]["momentum"],
                weight_decay=training_info["optimizer_params"]["weight_decay"],
            )

        elif training_info["optimizer_name"] == "Adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=training_info["optimizer_params"]["lr"],
                weight_decay=training_info["optimizer_params"]["weight_decay"],
            )

        else:
            raise ValueError(
                "Optimzer {} is not supported".format(training_info["optimizer_name"])
            )

    def instantiate_model(self):
        if self.model_class in globals():
            class_instance = globals()[self.model_class](self.model_structure).to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            return class_instance
        else:
            raise ValueError("Class '{}' not found.".format(self.model_class))

    def train_one_epoch(self, training_loader):
        epoch_cumulative_loss = 0

        for data in training_loader:
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            current_batch_loss = self.loss_fn(outputs, labels)
            current_batch_loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Add batch loss to get final epoch loss
            epoch_cumulative_loss += current_batch_loss.item() * inputs.size(0)

        return epoch_cumulative_loss

    def start_training(
        self,
        training_loader,
        training_samples_n,
        validation_loader,
        validation_samples_n,
        loss_threshold,
        loss_decrease_threshold,
        epoch_logs_n=100,
    ):
        best_avg_vloss = 1_000_000.0
        current_time = int(time.time())

        # Print start message
        print(
            "Starting training for model_{}_{} | Training Set: {} Validation Set {}".format(
                self.model_name, current_time, training_samples_n, validation_samples_n
            )
        )

        # Save model metadata
        model_meta = {
            "model_class": self.model_class,
            "model_name": self.model_name,
            "model_structure": self.model_structure,
            "training_info": self.training_info,
            "data_info": self.data_info,
        }
        with open(
            self.saving_info["save_model_meta_path"]
            + "/model_{}_{}.json".format(self.model_name, current_time),
            "w",
        ) as json_file:
            json.dump(model_meta, json_file)

        # Train model
        model_losses = []
        for epoch in range(self.training_info["n_epochs"]):
            # Epoch start timestamp
            epoch_start_timestamp = time.time()

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            current_epoch_tloss = self.train_one_epoch(training_loader)

            # We don't need gradients on to do reporting
            self.model.train(False)

            # Calculate validation loss
            current_epoch_vloss = 0.0
            for vdataloader in validation_loader:
                vinputs, vlabels = vdataloader
                voutputs = self.model(vinputs)
                vloss = self.loss_fn(voutputs, vlabels)
                current_epoch_vloss += vloss.item() * vinputs.size(0)

            # Epoch end timestamp
            epoch_time = int(time.time() - epoch_start_timestamp)

            # Calculate mean losses
            current_epoch_avg_tloss = current_epoch_tloss / training_samples_n
            current_epoch_avg_vloss = current_epoch_vloss / validation_samples_n
            model_losses.append(
                [current_epoch_avg_tloss, current_epoch_avg_vloss, epoch_time]
            )

            # Print progress
            if epoch % epoch_logs_n == 0:
                print(
                    "EPOCH {} TRAIN {} VALID {} TIME_PER_EPOCH {}".format(
                        self.current_epoch_number,
                        round(current_epoch_avg_tloss, 6),
                        round(current_epoch_avg_vloss, 6),
                        epoch_time,
                    )
                )

            # Track best performance, and save the model's state if needed
            if self.saving_info["save_model"] == 1:
                if (current_epoch_avg_vloss < loss_threshold) & (
                    current_epoch_avg_vloss
                    < (1 - loss_decrease_threshold) * best_avg_vloss
                ):
                    best_avg_vloss = current_epoch_avg_vloss
                    model_path = self.saving_info[
                        "save_model_path"
                    ] + "/model_{}_{}".format(
                        self.model_name, current_time, self.current_epoch_number
                    )
                    torch.save(self.model.state_dict(), model_path)

            # Update epoch number
            self.current_epoch_number += 1

        # Save training information
        with open(
            self.saving_info["save_model_meta_path"]
            + "/model_{}.csv".format(self.model_name),
            "w",
            newline="",
        ) as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows([["avg_loss", "avg_vloss", "epoch_time"]])
            csvwriter.writerows(model_losses)

        print(
            "Finishing training for model_{}_{}".format(self.model_name, current_time)
        )
        print("-----------------------------------------------------------------")

    def __init__(
        self,
        model_class,
        model_name,
        model_structure,
        training_info,
        data_info,
        saving_info,
        current_epoch_number=0,
    ):
        # Initialize variables
        self.model_class = model_class
        self.model_name = model_name
        self.model_structure = model_structure
        self.training_info = training_info
        self.data_info = data_info
        self.saving_info = saving_info
        self.current_epoch_number = current_epoch_number

        # Build Model
        self.model = self.instantiate_model()

        # Validate training info
        self.validate_training_info(training_info)

        # Build loss function
        if (data_info["label_type"] == "mineral-regression") or (
            data_info["label_type"] == "om-regression"
        ):
            self.loss_fn = nn.MSELoss()
        elif (data_info["label_type"] == "usda-classification") or (
            data_info["label_type"] == "custom-classification"
        ):
            if self.training_info["cross_entropy_loss_weights"] is not None:
                cross_entropy_loss_weights = torch.tensor(
                    self.training_info["cross_entropy_loss_weights"]
                ).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            else:
                cross_entropy_loss_weights = None
            self.loss_fn = nn.CrossEntropyLoss(cross_entropy_loss_weights)
        else:
            raise Exception("Unknown label type")

        # Build optimizer
        self.optimizer = self.create_optimizer(self.model, training_info)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNext2B(nn.Module):
    def buildBranch(self, model_type, label_type, weights_path):
        """
        This function returns 4 things:
            - Feature extraction block od MobilNetV3 Large model
            - Number of neurons used for the last fully connected layer
            - LayerNorm2 property
            - LayerNorm2 property
        """

        if weights_path == "default":
            if model_type == "tiny":
                branch = torchvision.models.convnext_tiny(
                    weights="ConvNeXt_Tiny_Weights.IMAGENET1K_V1"
                )
            elif model_type == "small":
                branch = torchvision.models.convnext_small(
                    weights="ConvNeXt_Small_Weights.IMAGENET1K_V1"
                )
            elif model_type == "base":
                branch = torchvision.models.convnext_base(
                    weights="ConvNeXt_Base_Weights.IMAGENET1K_V1"
                )
            elif model_type == "large":
                branch = torchvision.models.convnext_large(
                    weights="ConvNeXt_Large_Weights.IMAGENET1K_V1"
                )
            else:
                raise Exception("Unknown Model Type")

        else:
            if model_type == "tiny":
                branch = torchvision.models.convnext_tiny()
            elif model_type == "small":
                branch = torchvision.models.convnext_small()
            elif model_type == "base":
                branch = torchvision.models.convnext_base()
            elif model_type == "large":
                branch = torchvision.models.convnext_large()
            else:
                raise Exception("Unknown Model Type")

        normalize_layer_shape = branch.classifier[0].normalized_shape[
            0
        ]  # This corresponds to the number of neurons for the last linear layer
        normalize_layer_eps = branch.classifier[0].eps
        normalize_layer_elementwise_affine = branch.classifier[0].elementwise_affine

        # Load weights if needed
        if weights_path != "default":
            if label_type == "mineral-regression":
                branch.classifier = nn.Sequential(
                    LayerNorm2d(
                        normalize_layer_shape,
                        eps=normalize_layer_eps,
                        elementwise_affine=normalize_layer_elementwise_affine,
                    ),
                    nn.Flatten(1),
                    nn.Linear(
                        in_features=normalize_layer_shape, out_features=3, bias=True
                    ),
                    nn.Softmax(dim=1),
                )
            elif label_type == "om-regression":
                branch.classifier = nn.Sequential(
                    LayerNorm2d(
                        normalize_layer_shape,
                        eps=normalize_layer_eps,
                        elementwise_affine=normalize_layer_elementwise_affine,
                    ),
                    nn.Flatten(1),
                    nn.Linear(
                        in_features=normalize_layer_shape, out_features=4, bias=True
                    ),
                    nn.Softmax(dim=1),
                )
            elif label_type == "usda-classification":
                branch.classifier = nn.Sequential(
                    LayerNorm2d(
                        normalize_layer_shape,
                        eps=normalize_layer_eps,
                        elementwise_affine=normalize_layer_elementwise_affine,
                    ),
                    nn.Flatten(1),
                    nn.Linear(
                        in_features=normalize_layer_shape, out_features=3, bias=True
                    ),
                    nn.Softmax(dim=1),
                )
            elif label_type == "custom-classification":
                branch.classifier = nn.Sequential(
                    LayerNorm2d(
                        normalize_layer_shape,
                        eps=normalize_layer_eps,
                        elementwise_affine=normalize_layer_elementwise_affine,
                    ),
                    nn.Flatten(1),
                    nn.Linear(
                        in_features=normalize_layer_shape, out_features=3, bias=True
                    ),
                    nn.Softmax(dim=1),
                )

            branch.load_state_dict(
                torch.load(weights_path, map_location=torch.device("cpu"))
            )

        return (
            branch.features,
            normalize_layer_shape,
            normalize_layer_eps,
            normalize_layer_elementwise_affine,
        )

    def __init__(
        self,
        model_type,
        label_type,
        microscope_weights_path,
        camera_weights_path,
    ):
        super(ConvNext2B, self).__init__()

        (
            self.microscope_branch,
            microscope_normalize_layer_shape,
            microscope_normalize_layer_eps,
            microscope_normalize_layer_elementwise_affine,
        ) = self.buildBranch(model_type, label_type, microscope_weights_path)
        (
            self.camera_branch,
            camera_normalize_layer_shape,
            camera_normalize_layer_eps,
            camera_normalize_layer_elementwise_affine,
        ) = self.buildBranch(model_type, label_type, camera_weights_path)

        self.micro_avgpool = nn.AdaptiveAvgPool2d(1)
        self.camera_avgpool = nn.AdaptiveAvgPool2d(1)

        classifier_in_features = (
            microscope_normalize_layer_shape + camera_normalize_layer_shape
        )

        if label_type == "mineral-regression":
            self.classifier = nn.Sequential(
                LayerNorm2d(
                    classifier_in_features,
                    eps=microscope_normalize_layer_eps,
                    elementwise_affine=microscope_normalize_layer_elementwise_affine,
                ),
                nn.Flatten(1),
                nn.Linear(
                    in_features=classifier_in_features, out_features=3, bias=True
                ),
                nn.Softmax(dim=1),
            )
        elif label_type == "om-regression":
            self.classifier = nn.Sequential(
                LayerNorm2d(
                    classifier_in_features,
                    eps=microscope_normalize_layer_eps,
                    elementwise_affine=microscope_normalize_layer_elementwise_affine,
                ),
                nn.Flatten(1),
                nn.Linear(
                    in_features=classifier_in_features, out_features=4, bias=True
                ),
                nn.Softmax(dim=1),
            )
        elif label_type == "usda-classification":
            self.classifier = nn.Sequential(
                LayerNorm2d(
                    classifier_in_features,
                    eps=microscope_normalize_layer_eps,
                    elementwise_affine=microscope_normalize_layer_elementwise_affine,
                ),
                nn.Flatten(1),
                nn.Linear(
                    in_features=classifier_in_features, out_features=12, bias=True
                ),
                nn.Softmax(dim=1),
            )
        elif label_type == "custom-classification":
            self.classifier = nn.Sequential(
                LayerNorm2d(
                    classifier_in_features,
                    eps=microscope_normalize_layer_eps,
                    elementwise_affine=microscope_normalize_layer_elementwise_affine,
                ),
                nn.Flatten(1),
                nn.Linear(
                    in_features=classifier_in_features, out_features=13, bias=True
                ),
                nn.Softmax(dim=1),
            )

    def forward(self, x_micro, x_camera):
        x_micro = self.microscope_branch(x_micro)
        x_camera = self.camera_branch(x_camera)

        x_micro = self.micro_avgpool(x_micro)
        x_camera = self.camera_avgpool(x_camera)

        x = torch.cat((x_camera, x_micro), dim=1)
        x = self.classifier(x)

        return x


class ConvnextTrainer:
    TRAINING_MANDATORY_FIELDS = [
        "n_epochs",
        "batch_size",
        "optimizer_name",
        "optimizer_params",
    ]
    SGD_MANDATORY_FIELDS = ["lr", "momentum", "weight_decay"]
    ADAM_MANDATORY_FIELDS = ["lr"]

    def validate_training_info(self, training_info):
        # Validate main fields
        for field in ModelTrainer.TRAINING_MANDATORY_FIELDS:
            if field not in training_info:
                # Raising a ValueError with a custom error message
                raise ValueError("Field {} is missing in model_structure".format(field))

        # Validate optimizer fields
        if training_info["optimizer_name"] == "SGD":
            for field in ModelTrainer.SGD_MANDATORY_FIELDS:
                if field not in training_info["optimizer_params"]:
                    # Raising a ValueError with a custom error message
                    raise ValueError(
                        "Field {} is missing in optimizer_params".format(field)
                    )
        elif training_info["optimizer_name"] == "Adam":
            for field in ModelTrainer.ADAM_MANDATORY_FIELDS:
                if field not in training_info["optimizer_params"]:
                    # Raising a ValueError with a custom error message
                    raise ValueError(
                        "Field {} is missing in optimizer_params".format(field)
                    )
        else:
            raise ValueError(
                "Optimzer {} is not supported".format(training_info["optimizer_name"])
            )

        return True

    def create_optimizer(self, model, training_info):
        # Validate optimizer fields
        if training_info["optimizer_name"] == "SGD":
            return torch.optim.SGD(
                model.parameters(),
                lr=training_info["optimizer_params"]["lr"],
                momentum=training_info["optimizer_params"]["momentum"],
                weight_decay=training_info["optimizer_params"]["weight_decay"],
            )

        elif training_info["optimizer_name"] == "Adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=training_info["optimizer_params"]["lr"],
                weight_decay=training_info["optimizer_params"]["weight_decay"],
            )

        else:
            raise ValueError(
                "Optimzer {} is not supported".format(training_info["optimizer_name"])
            )

    def instantiate_model(
        self, model_type, label_type, pre_trained, train_only_last_layer
    ):
        if model_type == "tiny":
            if pre_trained:
                model = torchvision.models.convnext_tiny(
                    weights="ConvNeXt_Tiny_Weights.IMAGENET1K_V1"
                )
            else:
                model = torchvision.models.convnext_tiny()
        elif model_type == "small":
            if pre_trained:
                model = torchvision.models.convnext_small(
                    weights="ConvNeXt_Small_Weights.IMAGENET1K_V1"
                )
            else:
                model = torchvision.models.convnext_small()
        elif model_type == "base":
            if pre_trained:
                model = torchvision.models.convnext_base(
                    weights="ConvNeXt_Base_Weights.IMAGENET1K_V1"
                )
            else:
                model = torchvision.models.convnext_base()
        elif model_type == "large":
            if pre_trained:
                model = torchvision.models.convnext_large(
                    weights="ConvNeXt_Large_Weights.IMAGENET1K_V1"
                )
            else:
                model = torchvision.models.convnext_large()
        else:
            raise Exception("Unknown Model Type")

        if train_only_last_layer:
            # Freeze all layers (final layer changed later)
            for p in model.parameters():
                p.requires_grad = False

        # Change the final layer
        normalize_layer_shape = model.classifier[0].normalized_shape[0]
        normalize_layer_eps = model.classifier[0].eps
        normalize_layer_elementwise_affine = model.classifier[0].elementwise_affine
        if label_type == "mineral-regression":
            model.classifier = nn.Sequential(
                LayerNorm2d(
                    normalize_layer_shape,
                    eps=normalize_layer_eps,
                    elementwise_affine=normalize_layer_elementwise_affine,
                ),
                nn.Flatten(1),
                nn.Linear(in_features=normalize_layer_shape, out_features=3, bias=True),
                nn.Softmax(dim=1),
            )
        elif label_type == "om-regression":
            model.classifier = nn.Sequential(
                LayerNorm2d(
                    normalize_layer_shape,
                    eps=normalize_layer_eps,
                    elementwise_affine=normalize_layer_elementwise_affine,
                ),
                nn.Flatten(1),
                nn.Linear(in_features=normalize_layer_shape, out_features=4, bias=True),
                nn.Softmax(dim=1),
            )
        elif label_type == "usda-classification":
            model.classifier = nn.Sequential(
                LayerNorm2d(
                    normalize_layer_shape,
                    eps=normalize_layer_eps,
                    elementwise_affine=normalize_layer_elementwise_affine,
                ),
                nn.Flatten(1),
                nn.Linear(
                    in_features=normalize_layer_shape, out_features=12, bias=True
                ),
                nn.Softmax(dim=1),
            )
        elif label_type == "custom-classification":
            model.classifier = nn.Sequential(
                LayerNorm2d(
                    normalize_layer_shape,
                    eps=normalize_layer_eps,
                    elementwise_affine=normalize_layer_elementwise_affine,
                ),
                nn.Flatten(1),
                nn.Linear(
                    in_features=normalize_layer_shape, out_features=13, bias=True
                ),
                nn.Softmax(dim=1),
            )

        model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        return model

    def train_one_epoch(self, training_loader):
        epoch_cumulative_loss = 0

        for inputs, labels in training_loader:
            inputs = inputs.to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            labels = labels.to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            current_batch_loss = self.loss_fn(outputs, labels)
            current_batch_loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Add batch loss to get final epoch loss
            epoch_cumulative_loss += current_batch_loss.item() * inputs.size(0)

        return epoch_cumulative_loss

    def start_training(
        self,
        training_loader,
        training_samples_n,
        epoch_logs_n=100,
    ):
        # Print start message
        print(
            "Starting training for model_{} | Training Set: {}".format(
                self.model_name, training_samples_n
            )
        )

        # Save model metadata
        model_meta = {
            "model_class": self.model_class,
            "model_name": self.model_name,
            "training_info": self.training_info,
            "data_info": self.data_info,
        }
        with open(
            self.saving_info["save_model_meta_path"]
            + "/model_{}.json".format(self.model_name),
            "w",
        ) as json_file:
            json.dump(model_meta, json_file)

        # Train model
        model_losses = []
        for epoch in range(self.training_info["n_epochs"]):
            # Epoch start timestamp
            epoch_start_timestamp = time.time()

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            current_epoch_tloss = self.train_one_epoch(training_loader)

            # We don't need gradients on to do reporting
            self.model.train(False)

            # Epoch end timestamp
            epoch_time = int(time.time() - epoch_start_timestamp)

            # Calculate mean losses
            current_epoch_avg_tloss = current_epoch_tloss / training_samples_n
            model_losses.append([current_epoch_avg_tloss, epoch_time])

            # Print progress
            if epoch % epoch_logs_n == 0:
                print(
                    "EPOCH {} TRAIN {} TIME_PER_EPOCH {}".format(
                        self.current_epoch_number,
                        round(current_epoch_avg_tloss, 6),
                        epoch_time,
                    )
                )

            # Save model
            if epoch > self.training_info["save_from_epoch"]:
                model_path = self.saving_info[
                    "save_model_path"
                ] + "/model_{}_{}".format(self.model_name, self.current_epoch_number)
                torch.save(self.model.state_dict(), model_path)

            # Update epoch number
            self.current_epoch_number += 1

        # Save training information
        with open(
            self.saving_info["save_model_meta_path"]
            + "/model_{}.csv".format(self.model_name),
            "w",
            newline="",
        ) as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows([["avg_loss", "avg_vloss", "epoch_time"]])
            csvwriter.writerows(model_losses)

        print("Finishing training for model_{}".format(self.model_name))
        print("-----------------------------------------------------------------")

    def __init__(
        self,
        model_class,
        model_name,
        model_structure,
        training_info,
        data_info,
        saving_info,
        current_epoch_number=0,
    ):
        # Initialize variables
        self.model_class = model_class
        self.model_name = model_name
        self.model_structure = model_structure
        self.training_info = training_info
        self.data_info = data_info
        self.saving_info = saving_info
        self.current_epoch_number = current_epoch_number

        # Build Model
        self.model = self.instantiate_model(
            self.model_structure["model_type"],
            self.data_info["label_type"],
            self.model_structure["pre_trained"],
            self.training_info["train_only_last_layer"],
        )

        # Validate training info
        self.validate_training_info(training_info)

        # Build loss function
        if (self.data_info["label_type"] == "mineral-regression") or (
            self.data_info["label_type"] == "om-regression"
        ):
            self.loss_fn = nn.MSELoss()
        elif (self.data_info["label_type"] == "usda-classification") or (
            self.data_info["label_type"] == "custom-classification"
        ):
            if self.training_info["cross_entropy_loss_weights"] is not None:
                cross_entropy_loss_weights = torch.tensor(
                    self.training_info["cross_entropy_loss_weights"]
                ).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            else:
                cross_entropy_loss_weights = None
            self.loss_fn = nn.CrossEntropyLoss(cross_entropy_loss_weights)
        else:
            raise Exception("Unknown label type")

        # Build optimizer
        self.optimizer = self.create_optimizer(self.model, training_info)


class ConvNext2BTrainer:
    TRAINING_MANDATORY_FIELDS = [
        "n_epochs",
        "batch_size",
        "optimizer_name",
        "optimizer_params",
    ]
    SGD_MANDATORY_FIELDS = ["lr", "momentum", "weight_decay"]
    ADAM_MANDATORY_FIELDS = ["lr"]

    def validate_training_info(self, training_info):
        # Validate main fields
        for field in ModelTrainer.TRAINING_MANDATORY_FIELDS:
            if field not in training_info:
                # Raising a ValueError with a custom error message
                raise ValueError("Field {} is missing in model_structure".format(field))

        # Validate optimizer fields
        if training_info["optimizer_name"] == "SGD":
            for field in ModelTrainer.SGD_MANDATORY_FIELDS:
                if field not in training_info["optimizer_params"]:
                    # Raising a ValueError with a custom error message
                    raise ValueError(
                        "Field {} is missing in optimizer_params".format(field)
                    )
        elif training_info["optimizer_name"] == "Adam":
            for field in ModelTrainer.ADAM_MANDATORY_FIELDS:
                if field not in training_info["optimizer_params"]:
                    # Raising a ValueError with a custom error message
                    raise ValueError(
                        "Field {} is missing in optimizer_params".format(field)
                    )
        else:
            raise ValueError(
                "Optimzer {} is not supported".format(training_info["optimizer_name"])
            )

        return True

    def create_optimizer(self, model, training_info):
        # Validate optimizer fields
        if training_info["optimizer_name"] == "SGD":
            return torch.optim.SGD(
                model.parameters(),
                lr=training_info["optimizer_params"]["lr"],
                momentum=training_info["optimizer_params"]["momentum"],
                weight_decay=training_info["optimizer_params"]["weight_decay"],
            )

        elif training_info["optimizer_name"] == "Adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=training_info["optimizer_params"]["lr"],
                weight_decay=training_info["optimizer_params"]["weight_decay"],
            )

        else:
            raise ValueError(
                "Optimzer {} is not supported".format(training_info["optimizer_name"])
            )

    def instantiate_model(
        self,
        model_type,
        label_type,
        microscope_weights_path,
        camera_weights_path,
        train_only_last_layer,
    ):
        model = ConvNext2B(
            model_type,
            label_type,
            microscope_weights_path,
            camera_weights_path,
        )

        if train_only_last_layer:
            # Freeze all layers (final layer changed later)
            for p in model.parameters():
                p.requires_grad = False

        # Change the final layer
        normalize_layer_shape = model.classifier[0].normalized_shape[0]
        normalize_layer_eps = model.classifier[0].eps
        normalize_layer_elementwise_affine = model.classifier[0].elementwise_affine
        if label_type == "mineral-regression":
            model.classifier = nn.Sequential(
                LayerNorm2d(
                    normalize_layer_shape,
                    eps=normalize_layer_eps,
                    elementwise_affine=normalize_layer_elementwise_affine,
                ),
                nn.Flatten(1),
                nn.Linear(in_features=normalize_layer_shape, out_features=3, bias=True),
                nn.Softmax(dim=1),
            )
        elif label_type == "om-regression":
            model.classifier = nn.Sequential(
                LayerNorm2d(
                    normalize_layer_shape,
                    eps=normalize_layer_eps,
                    elementwise_affine=normalize_layer_elementwise_affine,
                ),
                nn.Flatten(1),
                nn.Linear(in_features=normalize_layer_shape, out_features=4, bias=True),
                nn.Softmax(dim=1),
            )
        elif label_type == "usda-classification":
            model.classifier = nn.Sequential(
                LayerNorm2d(
                    normalize_layer_shape,
                    eps=normalize_layer_eps,
                    elementwise_affine=normalize_layer_elementwise_affine,
                ),
                nn.Flatten(1),
                nn.Linear(
                    in_features=normalize_layer_shape, out_features=12, bias=True
                ),
                nn.Softmax(dim=1),
            )
        elif label_type == "custom-classification":
            model.classifier = nn.Sequential(
                LayerNorm2d(
                    normalize_layer_shape,
                    eps=normalize_layer_eps,
                    elementwise_affine=normalize_layer_elementwise_affine,
                ),
                nn.Flatten(1),
                nn.Linear(
                    in_features=normalize_layer_shape, out_features=13, bias=True
                ),
                nn.Softmax(dim=1),
            )

        model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        return model

    def train_one_epoch(self, training_loader):
        epoch_cumulative_loss = 0

        for camera_inputs, micro_inputs, labels in training_loader:
            camera_inputs = camera_inputs.to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            micro_inputs = micro_inputs.to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            labels = labels.to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(micro_inputs, camera_inputs)

            # Compute the loss and its gradients
            current_batch_loss = self.loss_fn(outputs, labels)
            current_batch_loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Add batch loss to get final epoch loss
            epoch_cumulative_loss += current_batch_loss.item() * micro_inputs.size(0)

        return epoch_cumulative_loss

    def start_training(
        self,
        training_loader,
        training_samples_n,
        epoch_logs_n=100,
    ):
        # Print start message
        print(
            "Starting training for model_{} | Training Set: {}".format(
                self.model_name, training_samples_n
            )
        )

        # Save model metadata
        model_meta = {
            "model_class": self.model_class,
            "model_name": self.model_name,
            "training_info": self.training_info,
            "data_info": self.data_info,
        }
        with open(
            self.saving_info["save_model_meta_path"]
            + "/model_{}.json".format(self.model_name),
            "w",
        ) as json_file:
            json.dump(model_meta, json_file)

        # Train model
        model_losses = []
        for epoch in range(self.training_info["n_epochs"]):
            # Epoch start timestamp
            epoch_start_timestamp = time.time()

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            current_epoch_tloss = self.train_one_epoch(training_loader)

            # We don't need gradients on to do reporting
            self.model.train(False)

            # Epoch end timestamp
            epoch_time = int(time.time() - epoch_start_timestamp)

            # Calculate mean losses
            current_epoch_avg_tloss = current_epoch_tloss / training_samples_n
            model_losses.append([current_epoch_avg_tloss, epoch_time])

            # Print progress
            if epoch % epoch_logs_n == 0:
                print(
                    "EPOCH {} TRAIN {} TIME_PER_EPOCH {}".format(
                        self.current_epoch_number,
                        round(current_epoch_avg_tloss, 6),
                        epoch_time,
                    )
                )

            # Save model
            if epoch > self.training_info["save_from_epoch"]:
                model_path = self.saving_info[
                    "save_model_path"
                ] + "/model_{}_{}".format(self.model_name, self.current_epoch_number)
                torch.save(self.model.state_dict(), model_path)
            # Update epoch number
            self.current_epoch_number += 1

        # Save training information
        with open(
            self.saving_info["save_model_meta_path"]
            + "/model_{}.csv".format(self.model_name),
            "w",
            newline="",
        ) as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows([["avg_loss", "avg_vloss", "epoch_time"]])
            csvwriter.writerows(model_losses)

        print("Finishing training for model_{}".format(self.model_name))
        print("-----------------------------------------------------------------")

    def __init__(
        self,
        model_class,
        model_name,
        model_structure,
        training_info,
        data_info,
        saving_info,
        current_epoch_number=0,
    ):
        # Initialize variables
        self.model_class = model_class
        self.model_name = model_name
        self.model_structure = model_structure
        self.training_info = training_info
        self.data_info = data_info
        self.saving_info = saving_info
        self.current_epoch_number = current_epoch_number

        # Build Model
        self.model = self.instantiate_model(
            self.model_structure["model_type"],
            self.data_info["label_type"],
            self.model_structure["microscope_weights_path"],
            self.model_structure["camera_weights_path"],
            self.training_info["train_only_last_layer"],
        )

        # Validate training info
        self.validate_training_info(training_info)

        # Build loss function
        if (self.data_info["label_type"] == "mineral-regression") or (
            self.data_info["label_type"] == "om-regression"
        ):
            self.loss_fn = nn.MSELoss()
        elif (self.data_info["label_type"] == "usda-classification") or (
            self.data_info["label_type"] == "custom-classification"
        ):
            if self.training_info["cross_entropy_loss_weights"] is not None:
                cross_entropy_loss_weights = torch.tensor(
                    self.training_info["cross_entropy_loss_weights"]
                ).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            else:
                cross_entropy_loss_weights = None
            self.loss_fn = nn.CrossEntropyLoss(cross_entropy_loss_weights)
        else:
            raise Exception("Unknown label type")

        # Build optimizer
        self.optimizer = self.create_optimizer(self.model, training_info)
