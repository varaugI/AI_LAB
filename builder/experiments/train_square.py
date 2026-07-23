import random
import os
import sys

# Add the parent directory of builder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from builder.framework import Sequential, DenseLayer, SigmoidLayer

MAX_INPUT = 10.0
MAX_OUTPUT = MAX_INPUT * MAX_INPUT

def normalize_input(number):
    return number / MAX_INPUT

def normalize_output(number):
    return number / MAX_OUTPUT

def denormalize_output(number):
    return number * MAX_OUTPUT

def create_square_training_data(number_of_examples):
    inputs = []
    targets = []
    for _ in range(number_of_examples):
        number = random.uniform(0, MAX_INPUT)
        answer = number * number
        inputs.append([normalize_input(number)])
        targets.append([normalize_output(answer)])
    return inputs, targets

def main():
    random.seed(42)
    training_inputs, training_targets = create_square_training_data(1_500)

    # Note: input_size is 1 instead of 2!
    network = Sequential(
        layers=[
            DenseLayer(input_size=1, output_size=16),
            SigmoidLayer(),
            DenseLayer(input_size=16, output_size=1),
        ],
        learning_rate=0.05,
    )

    print("Training the neural network on SQUARE (x^2)...\n")
    network.train(training_inputs, training_targets, epochs=15_000, print_every=1000, batch_size=32)
    network.save("square_network.json")
    print("\nTraining completed.")
    print("Type 'exit' to stop.")

    while True:
        text = input("\nEnter a number to square (0-10): ").strip()
        if text.lower() == "exit": break

        try:
            number = float(text)
        except ValueError:
            print("Please enter a valid number.")
            continue

        normalized_inputs = [[normalize_input(number)]]
        normalized_prediction = network.predict(normalized_inputs)[0][0]
        prediction = denormalize_output(normalized_prediction)

        print(f"Neural network prediction: {prediction:.4f}")
        print(f"Correct answer: {number * number:.4f}")

if __name__ == "__main__":
    main()
