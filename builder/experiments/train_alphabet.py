"""Train a neural network to classify capital letters A-Z.

This is intentionally a small bitmap-learning step before real handwritten-image
recognition. It uses only Python lists and the standard library.
"""

import random

from builder.datasets.alphabet_data import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    LABELS,
    create_alphabet_dataset,
    create_clean_alphabet_data,
)
from builder.framework import (
    DenseLayer,
    DropoutLayer,
    LeakyReLULayer,
    Sequential,
    SoftmaxLayer,
    TanhLayer,
)


def build_alphabet_model():
    input_size = CANVAS_WIDTH * CANVAS_HEIGHT
    return Sequential(
        layers=[
            DenseLayer(input_size, 128),
            LeakyReLULayer(alpha=0.05),
            DropoutLayer(rate=0.2),
            DenseLayer(128, 64),
            TanhLayer(),
            DropoutLayer(rate=0.1),
            DenseLayer(64, len(LABELS)),
            SoftmaxLayer(),
        ],
        learning_rate=0.008,
        optimizer="adam",
        optimizer_params={
            "gradient_clip": 1.0,
            "weight_decay": 0.00001,
        },
    )


def main():
    random.seed(42)

    train_x, train_y, _ = create_alphabet_dataset(
        copies_per_letter=15,
        noise_probability=0.010,
        max_shift=1,
        seed=42,
        include_clean=True,
    )
    validation_x, validation_y, _ = create_alphabet_dataset(
        copies_per_letter=5,
        noise_probability=0.010,
        max_shift=1,
        seed=9001,
        include_clean=False,
    )

    model = build_alphabet_model()
    model.summary()
    print("\nTraining A-Z classifier...\n")

    model.train_mini_batch(
        train_x,
        train_y,
        batch_size=26,
        epochs=350,
        print_every=10,
        loss_type="cce",
        validation_data=(validation_x, validation_y),
        metrics=["accuracy"],
        early_stopping=True,
        patience=35,
        min_delta=0.00005,
        restore_best_weights=False,
        lr_schedule="plateau",
        schedule_params={
            "patience": 10,
            "factor": 0.5,
            "min_learning_rate": 0.0002,
        },
        seed=42,
    )

    print("\nValidation:")
    print(model.evaluate_metrics(
        validation_x,
        validation_y,
        loss_type="cce",
        metrics=["accuracy"],
    ))

    clean_x, clean_y, clean_letters = create_clean_alphabet_data()
    predicted_classes = model.predict_classes(clean_x)
    correct = 0
    print("\nClean A-Z predictions:")
    for actual, predicted_index in zip(clean_letters, predicted_classes):
        predicted = LABELS[predicted_index]
        correct += predicted == actual
        print(f"{actual} -> {predicted}")
    print(f"Clean accuracy: {correct}/{len(clean_letters)}")

    model.save("alphabet_network.json")
    print("\nSaved alphabet_network.json")
    print("Run: python -m builder.experiments.recognize_alphabet")


if __name__ == "__main__":
    main()
