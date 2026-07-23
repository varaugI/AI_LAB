import copy
import json
import math
import random

from .layers import DenseLayer
from .losses import (
    mean_squared_error,
    mean_squared_error_derivative,
    binary_cross_entropy,
    binary_cross_entropy_derivative,
    categorical_cross_entropy,
    categorical_cross_entropy_derivative,
)
from .metrics import argmax, binary_accuracy, classification_accuracy
from .data_utils import train_test_split


_LOSSES = {
    "mse": (mean_squared_error, mean_squared_error_derivative),
    "bce": (binary_cross_entropy, binary_cross_entropy_derivative),
    "cce": (categorical_cross_entropy, categorical_cross_entropy_derivative),
}

_OPTIMIZERS = {"sgd", "momentum", "adam"}


class Sequential:
    """A small sequential neural-network model written using Python lists."""

    def __init__(self, layers, learning_rate=0.05, optimizer="sgd",
                 optimizer_params=None):
        if not layers:
            raise ValueError("Sequential requires at least one layer.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if optimizer not in _OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {tuple(sorted(_OPTIMIZERS))}.")

        self.layers = layers
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params or {}
        self.history = {}
        self._optimizer_step = 0

    def configure_training(self, optimizer=None, learning_rate=None, **optimizer_params):
        """Change optimizer settings before continuing training."""
        if optimizer is not None:
            if optimizer not in _OPTIMIZERS:
                raise ValueError(f"optimizer must be one of {tuple(sorted(_OPTIMIZERS))}.")
            self.optimizer = optimizer
        if learning_rate is not None:
            if learning_rate <= 0:
                raise ValueError("learning_rate must be positive.")
            self.learning_rate = learning_rate
            self.initial_learning_rate = learning_rate
        if optimizer_params:
            self.optimizer_params = optimizer_params
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.reset_optimizer_state()
        self._optimizer_step = 0

    def forward(self, inputs, training=False):
        output = inputs
        for layer in self.layers:
            if hasattr(layer, "is_training"):
                layer.is_training = training
            output = layer.forward(output)
        return output

    def backward(self, loss_gradient):
        self._optimizer_step += 1
        gradient = loss_gradient
        for layer in reversed(self.layers):
            if isinstance(layer, DenseLayer):
                gradient = layer.backward(
                    gradient,
                    learning_rate=self.learning_rate,
                    optimizer=self.optimizer,
                    optimizer_params=self.optimizer_params,
                    step=self._optimizer_step,
                )
            else:
                gradient = layer.backward(gradient)
        return gradient

    def train_on_batch(self, inputs, targets, loss_type="mse"):
        """Perform one optimizer update using a supplied batch."""
        if not inputs or len(inputs) != len(targets):
            raise ValueError("inputs and targets must be non-empty and equally sized.")
        if loss_type not in _LOSSES:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        loss_fn, gradient_fn = _LOSSES[loss_type]
        predictions = self.forward(inputs, training=True)
        loss = loss_fn(predictions, targets)
        self.backward(gradient_fn(predictions, targets))
        return loss

    def train_on_sample(self, input_row, target_row, loss_type="mse"):
        """Online learning: update the model from one new example."""
        return self.train_on_batch([input_row], [target_row], loss_type=loss_type)

    def _learning_rate_for_epoch(self, schedule, epoch, schedule_params):
        if schedule is None or schedule == "constant":
            return self.learning_rate
        if callable(schedule):
            value = schedule(epoch, self.learning_rate)
            if value <= 0:
                raise ValueError("A learning-rate schedule must return a positive value.")
            return value

        decay_rate = schedule_params.get("decay_rate", 0.95)
        minimum = schedule_params.get("min_learning_rate", 1e-6)
        base = schedule_params.get("base_learning_rate", self.initial_learning_rate)

        if schedule == "step":
            step_size = schedule_params.get("step_size", 50)
            if step_size <= 0:
                raise ValueError("step_size must be positive.")
            value = base * (decay_rate ** ((epoch - 1) // step_size))
        elif schedule == "exponential":
            value = base * (decay_rate ** (epoch - 1))
        elif schedule == "time":
            value = base / (1.0 + decay_rate * (epoch - 1))
        elif schedule == "cosine":
            total_epochs = schedule_params.get("total_epochs")
            if not total_epochs:
                raise ValueError("cosine schedule requires total_epochs.")
            progress = min(1.0, (epoch - 1) / max(1, total_epochs - 1))
            value = minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))
        elif schedule == "plateau":
            # Plateau is updated after validation below, not here.
            return self.learning_rate
        else:
            raise ValueError(
                "lr_schedule must be constant, step, exponential, time, cosine, "
                "plateau, a callable, or None."
            )
        return max(minimum, value)

    @staticmethod
    def _metric_value(metric, predictions, targets, loss_type):
        if callable(metric):
            return metric(predictions, targets)
        if metric == "accuracy":
            return (
                binary_accuracy(predictions, targets)
                if loss_type == "bce"
                else classification_accuracy(predictions, targets)
            )
        if metric == "binary_accuracy":
            return binary_accuracy(predictions, targets)
        if metric == "classification_accuracy":
            return classification_accuracy(predictions, targets)
        raise ValueError(f"Unknown metric: {metric}")

    def train(self, inputs, targets, epochs=1000, print_every=100,
              loss_type="mse", batch_size=None, shuffle=True,
              validation_data=None, validation_split=0.0,
              metrics=None, early_stopping=False, patience=20,
              min_delta=0.0, restore_best_weights=True,
              lr_schedule=None, schedule_params=None, callbacks=None,
              seed=42):
        """Train the model.

        This is the general training method. It supports full-batch, stochastic,
        and mini-batch learning depending on ``batch_size``.

        Useful options:
        - ``optimizer`` is selected when the model is created: sgd/momentum/adam.
        - ``validation_split`` or ``validation_data`` tracks unseen examples.
        - ``early_stopping=True`` stops when validation no longer improves.
        - ``lr_schedule`` supports step/exponential/time/cosine/plateau.
        - ``callbacks`` receives ``(model, logs)`` after every epoch.

        The returned value remains a list of training losses for compatibility.
        Richer information is available in ``model.history``.
        """
        if len(inputs) != len(targets) or not inputs:
            raise ValueError("inputs and targets must be non-empty and have equal length.")
        if loss_type not in _LOSSES:
            raise ValueError(f"Unknown loss_type: {loss_type}. Choose from {tuple(_LOSSES)}")
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if patience <= 0:
            raise ValueError("patience must be positive.")
        if min_delta < 0:
            raise ValueError("min_delta cannot be negative.")
        if validation_data is not None and validation_split:
            raise ValueError("Use validation_data or validation_split, not both.")
        if not 0.0 <= validation_split < 1.0:
            raise ValueError("validation_split must be in [0, 1).")

        schedule_params = dict(schedule_params or {})
        callbacks = callbacks or []
        metrics = [] if metrics is None else ([metrics] if isinstance(metrics, str) else list(metrics))

        train_inputs = list(inputs)
        train_targets = list(targets)
        validation_inputs = validation_targets = None

        if validation_split:
            train_inputs, validation_inputs, train_targets, validation_targets = train_test_split(
                train_inputs,
                train_targets,
                test_size=validation_split,
                shuffle=True,
                seed=seed,
            )
        elif validation_data is not None:
            validation_inputs, validation_targets = validation_data
            if not validation_inputs or len(validation_inputs) != len(validation_targets):
                raise ValueError("validation inputs and targets must be non-empty and equally sized.")

        loss_fn, gradient_fn = _LOSSES[loss_type]
        n = len(train_inputs)
        batch_size = n if batch_size is None else max(1, min(batch_size, n))
        rng = random.Random(seed)

        self.history = {
            "loss": [],
            "val_loss": [],
            "learning_rate": [],
            "metrics": {name if isinstance(name, str) else getattr(name, "__name__", "metric"): [] for name in metrics},
            "val_metrics": {name if isinstance(name, str) else getattr(name, "__name__", "metric"): [] for name in metrics},
            "stopped_epoch": None,
        }

        best_loss = float("inf")
        best_weights = None
        epochs_without_improvement = 0
        plateau_bad_epochs = 0

        if lr_schedule == "cosine":
            schedule_params.setdefault("total_epochs", epochs)

        for epoch in range(1, epochs + 1):
            self.learning_rate = self._learning_rate_for_epoch(
                lr_schedule, epoch, schedule_params
            )

            indices = list(range(n))
            if shuffle:
                rng.shuffle(indices)

            epoch_loss = 0.0
            batches = 0
            for start in range(0, n, batch_size):
                batch_ids = indices[start:start + batch_size]
                batch_inputs = [train_inputs[i] for i in batch_ids]
                batch_targets = [train_targets[i] for i in batch_ids]
                batch_loss = self.train_on_batch(
                    batch_inputs,
                    batch_targets,
                    loss_type=loss_type,
                )
                epoch_loss += batch_loss
                batches += 1

            epoch_loss /= batches
            self.history["loss"].append(epoch_loss)
            self.history["learning_rate"].append(self.learning_rate)

            # Metrics are calculated after the epoch with the latest weights.
            train_predictions = self.predict(train_inputs) if metrics else None
            for metric in metrics:
                name = metric if isinstance(metric, str) else getattr(metric, "__name__", "metric")
                self.history["metrics"][name].append(
                    self._metric_value(metric, train_predictions, train_targets, loss_type)
                )

            validation_loss = None
            if validation_inputs is not None:
                validation_predictions = self.predict(validation_inputs)
                validation_loss = loss_fn(validation_predictions, validation_targets)
                self.history["val_loss"].append(validation_loss)
                for metric in metrics:
                    name = metric if isinstance(metric, str) else getattr(metric, "__name__", "metric")
                    self.history["val_metrics"][name].append(
                        self._metric_value(metric, validation_predictions, validation_targets, loss_type)
                    )

            monitored_loss = validation_loss if validation_loss is not None else epoch_loss
            improved = monitored_loss < best_loss - min_delta
            if improved:
                best_loss = monitored_loss
                epochs_without_improvement = 0
                plateau_bad_epochs = 0
                if restore_best_weights:
                    best_weights = self.get_weights()
            else:
                epochs_without_improvement += 1
                plateau_bad_epochs += 1

            if lr_schedule == "plateau":
                lr_patience = schedule_params.get("patience", 5)
                factor = schedule_params.get("factor", 0.5)
                minimum = schedule_params.get("min_learning_rate", 1e-6)
                if not 0.0 < factor < 1.0:
                    raise ValueError("plateau factor must be between 0 and 1.")
                if plateau_bad_epochs >= lr_patience:
                    self.learning_rate = max(minimum, self.learning_rate * factor)
                    plateau_bad_epochs = 0

            logs = {
                "epoch": epoch,
                "loss": epoch_loss,
                "val_loss": validation_loss,
                "learning_rate": self.learning_rate,
            }
            for metric in metrics:
                name = metric if isinstance(metric, str) else getattr(metric, "__name__", "metric")
                logs[name] = self.history["metrics"][name][-1]
                if validation_inputs is not None:
                    logs[f"val_{name}"] = self.history["val_metrics"][name][-1]

            if print_every and (epoch == 1 or epoch % print_every == 0 or epoch == epochs):
                message = f"Epoch {epoch:6d} | Loss: {epoch_loss:.10f}"
                if validation_loss is not None:
                    message += f" | Val loss: {validation_loss:.10f}"
                for metric in metrics:
                    name = metric if isinstance(metric, str) else getattr(metric, "__name__", "metric")
                    message += f" | {name}: {logs[name]:.4f}"
                    if validation_inputs is not None:
                        message += f" | val_{name}: {logs[f'val_{name}']:.4f}"
                message += f" | LR: {self.learning_rate:.6g}"
                print(message)

            for callback in callbacks:
                callback(self, logs)

            if early_stopping and epochs_without_improvement >= patience:
                self.history["stopped_epoch"] = epoch
                if print_every:
                    print(f"Early stopping at epoch {epoch}; best monitored loss: {best_loss:.10f}")
                break

        if restore_best_weights and best_weights is not None and early_stopping:
            self.set_weights(best_weights)

        return self.history["loss"]

    # Convenience methods: all use the same backpropagation engine.
    def fit(self, inputs, targets, **kwargs):
        return self.train(inputs, targets, **kwargs)

    def partial_fit(self, inputs, targets, epochs=1, batch_size=None,
                    loss_type="mse", **kwargs):
        """Continue learning from new data without rebuilding the model."""
        kwargs.setdefault("print_every", 0)
        return self.train(
            inputs,
            targets,
            epochs=epochs,
            batch_size=batch_size,
            loss_type=loss_type,
            **kwargs,
        )

    def train_full_batch(self, inputs, targets, **kwargs):
        kwargs["batch_size"] = len(inputs)
        return self.train(inputs, targets, **kwargs)

    def train_stochastic(self, inputs, targets, **kwargs):
        kwargs["batch_size"] = 1
        return self.train(inputs, targets, **kwargs)

    def train_mini_batch(self, inputs, targets, batch_size=32, **kwargs):
        kwargs["batch_size"] = batch_size
        return self.train(inputs, targets, **kwargs)

    def predict(self, inputs):
        return self.forward(inputs, training=False)

    def predict_classes(self, inputs):
        return [argmax(row) for row in self.predict(inputs)]

    def evaluate(self, inputs, targets, loss_type="mse"):
        if loss_type not in _LOSSES:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        return _LOSSES[loss_type][0](self.predict(inputs), targets)

    def evaluate_metrics(self, inputs, targets, loss_type="mse", metrics=None):
        metrics = metrics or []
        metrics = [metrics] if isinstance(metrics, str) else list(metrics)
        predictions = self.predict(inputs)
        result = {"loss": _LOSSES[loss_type][0](predictions, targets)}
        for metric in metrics:
            name = metric if isinstance(metric, str) else getattr(metric, "__name__", "metric")
            result[name] = self._metric_value(metric, predictions, targets, loss_type)
        return result

    def get_weights(self):
        return [
            {
                "weights": copy.deepcopy(layer.weights),
                "biases": copy.deepcopy(layer.biases),
            }
            for layer in self.layers
            if isinstance(layer, DenseLayer)
        ]

    def set_weights(self, saved_weights):
        current = [layer for layer in self.layers if isinstance(layer, DenseLayer)]
        if len(saved_weights) != len(current):
            raise ValueError("Weight collection does not match this network.")

        for layer, layer_data in zip(current, saved_weights):
            weights = layer_data["weights"]
            biases = layer_data["biases"]
            if (
                len(layer.weights) != len(weights)
                or len(layer.weights[0]) != len(weights[0])
                or len(biases) != 1
                or len(layer.biases[0]) != len(biases[0])
            ):
                raise ValueError("Saved parameter shapes do not match this network.")
            layer.weights = copy.deepcopy(weights)
            layer.biases = copy.deepcopy(biases)
            layer.reset_optimizer_state()
        self._optimizer_step = 0

    def parameter_count(self):
        return sum(
            layer.parameter_count()
            for layer in self.layers
            if isinstance(layer, DenseLayer)
        )

    def summary(self):
        print("Model summary")
        print("-" * 56)
        total = 0
        for index, layer in enumerate(self.layers):
            if isinstance(layer, DenseLayer):
                count = layer.parameter_count()
                shape = f"{len(layer.weights)} -> {len(layer.weights[0])}"
                total += count
            else:
                count = 0
                shape = "activation"
            print(f"{index:>3}  {layer.__class__.__name__:<22} {shape:<14} params={count}")
        print("-" * 56)
        print(f"Total trainable parameters: {total}")
        print(f"Optimizer: {self.optimizer} | Learning rate: {self.learning_rate}")

    def to_dict(self):
        """Return a JSON-serializable description of the network.

        This makes the framework reusable inside larger saved artifacts such as
        language models without first writing a temporary model file.
        """
        return {
            "version": 5,
            "learning_rate": self.learning_rate,
            "initial_learning_rate": self.initial_learning_rate,
            "optimizer": self.optimizer,
            "optimizer_params": self.optimizer_params,
            "architecture": [
                layer.get_config() for layer in self.layers
                if hasattr(layer, "get_config")
            ],
            "dense_layers": self.get_weights(),
        }

    def save(self, filename):
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2)

    @classmethod
    def from_dict(cls, data):
        if "architecture" not in data:
            raise ValueError("Data does not contain architecture information.")

        from .layers import (
            DenseLayer, SigmoidLayer, ReLULayer, LeakyReLULayer,
            TanhLayer, SoftmaxLayer, DropoutLayer
        )

        layer_classes = {
            "DenseLayer": DenseLayer,
            "SigmoidLayer": SigmoidLayer,
            "ReLULayer": ReLULayer,
            "LeakyReLULayer": LeakyReLULayer,
            "TanhLayer": TanhLayer,
            "SoftmaxLayer": SoftmaxLayer,
            "DropoutLayer": DropoutLayer
        }

        layers = []
        for saved_config in data["architecture"]:
            config = dict(saved_config)
            layer_type = config.pop("type", None)
            if layer_type not in layer_classes:
                raise ValueError(f"Unknown saved layer type: {layer_type}")
            layers.append(layer_classes[layer_type](**config))

        learning_rate = data.get(
            "initial_learning_rate",
            data.get("learning_rate", 0.05),
        )
        model = cls(
            layers=layers,
            learning_rate=learning_rate,
            optimizer=data.get("optimizer", "sgd"),
            optimizer_params=data.get("optimizer_params"),
        )
        model.set_weights(data["dense_layers"])
        model.learning_rate = data.get("learning_rate", learning_rate)
        return model

    @classmethod
    def from_file(cls, filename):
        with open(filename, "r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))

    def load(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)

        saved = data.get("dense_layers")
        if saved is None:  # backward compatibility with version 1
            old = data.get("layers", {})
            saved = [old[key] for key in sorted(old)]

        self.set_weights(saved)
        self.learning_rate = data.get("learning_rate", self.learning_rate)
        self.initial_learning_rate = data.get("initial_learning_rate", self.learning_rate)
        saved_optimizer = data.get("optimizer", self.optimizer)
        if saved_optimizer in _OPTIMIZERS:
            self.optimizer = saved_optimizer
        self.optimizer_params = data.get("optimizer_params", self.optimizer_params)
