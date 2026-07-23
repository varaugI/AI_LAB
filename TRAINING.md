# How to Train Your Neural Network

Welcome to the AI LAB Training Documentation! This guide will walk you through how to use the custom, pure-Python framework to build, train, evaluate, and save powerful Neural Networks from scratch.

---

## 1. Importing the Framework
Everything you need is located in `builder.framework`.

```python
from builder.framework import (
    Sequential,
    DenseLayer,
    ReLULayer,
    SoftmaxLayer,
    DropoutLayer
)
```

## 2. Building a Model
The `Sequential` class is the container for your network. You define a list of `layers`, the `optimizer` (e.g., `"sgd"`, `"momentum"`, `"adam"`), and the `learning_rate`.

```python
model = Sequential(
    layers=[
        DenseLayer(input_size=10, output_size=32),
        ReLULayer(),
        DropoutLayer(rate=0.2), # 20% dropout to prevent overfitting
        DenseLayer(input_size=32, output_size=4),
        SoftmaxLayer() # Turns outputs into probabilities
    ],
    learning_rate=0.01,
    optimizer="adam",
    optimizer_params={"beta1": 0.9, "beta2": 0.999} # Optional
)
```

## 3. Training the Model
For large datasets, use **Mini-Batch Gradient Descent**. This divides your dataset into small chunks (`batch_size`), processes them, and updates the weights much faster than full-batch training.

The `train_mini_batch` function handles everything for you, including:
- **Loss Calculation:** Set `loss_type` to `"mse"` (regression), `"bce"` (binary classification), or `"cce"` (multi-class classification).
- **Validation:** Pass `validation_data=(val_x, val_y)` to see how the model performs on unseen data.
- **Early Stopping:** Stops training if the validation loss doesn't improve for a certain number of epochs (`patience`), saving immense time!
- **Learning Rate Schedules:** Dynamically reduce the learning rate to converge smoothly (`lr_schedule="plateau"`).

```python
history = model.train_mini_batch(
    train_x,
    train_y,
    batch_size=32,
    epochs=500,
    print_every=10,
    loss_type="cce",
    validation_data=(val_x, val_y),
    metrics=["accuracy"],
    early_stopping=True,
    patience=20,
    lr_schedule="plateau",
    schedule_params={
        "patience": 5,
        "factor": 0.5, # Reduce LR by 50% if plateaued
        "min_learning_rate": 0.0001
    }
)
```

## 4. Evaluating the Model
You can predict new data and evaluate its accuracy:

```python
metrics = model.evaluate_metrics(
    test_x,
    test_y,
    loss_type="cce",
    metrics=["accuracy"]
)
print(f"Test Accuracy: {metrics['accuracy']:.2%}")
```

## 5. Saving and Loading
The framework fully supports architecture serialization. This means you can save the entire model (structure AND weights) into a single JSON file and load it back later with one line of code!

**Saving:**
```python
model.save("my_awesome_network.json")
```

**Loading:**
```python
loaded_model = Sequential.from_file("my_awesome_network.json")
```
