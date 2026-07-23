import json
from .layers import DenseLayer
from .losses import (
    mean_squared_error, 
    mean_squared_error_derivative,
    binary_cross_entropy,
    binary_cross_entropy_derivative,
    categorical_cross_entropy,
    categorical_cross_entropy_derivative
)

class Sequential:
    def __init__(self, layers, learning_rate=0.05):
        self.layers = layers
        self.learning_rate = learning_rate

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, loss_gradient):
        output_gradient = loss_gradient
        for layer in reversed(self.layers):
            if isinstance(layer, DenseLayer):
                output_gradient = layer.backward(output_gradient, self.learning_rate)
            else:
                output_gradient = layer.backward(output_gradient)

    def train(self, inputs, targets, epochs=10_000, print_every=500, loss_type="mse"):
        for epoch in range(1, epochs + 1):
            predictions = self.forward(inputs)
            
            if loss_type == "mse":
                loss = mean_squared_error(predictions, targets)
                loss_gradient = mean_squared_error_derivative(predictions, targets)
            elif loss_type == "bce":
                loss = binary_cross_entropy(predictions, targets)
                loss_gradient = binary_cross_entropy_derivative(predictions, targets)
            elif loss_type == "cce":
                loss = categorical_cross_entropy(predictions, targets)
                loss_gradient = categorical_cross_entropy_derivative(predictions, targets)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")
                
            self.backward(loss_gradient)
            
            if epoch == 1 or epoch % print_every == 0:
                print(f"Epoch {epoch:6d} | Loss: {loss:.10f}")

    def predict(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        network_data = {"learning_rate": self.learning_rate, "layers": {}}
        dense_index = 0
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                network_data["layers"][f"dense_{dense_index}"] = {
                    "weights": layer.weights,
                    "biases": layer.biases,
                }
                dense_index += 1
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(network_data, file, indent=4)

    def load(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            network_data = json.load(file)
        self.learning_rate = network_data["learning_rate"]
        dense_index = 0
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer_data = network_data["layers"][f"dense_{dense_index}"]
                layer.weights = layer_data["weights"]
                layer.biases = layer_data["biases"]
                dense_index += 1
