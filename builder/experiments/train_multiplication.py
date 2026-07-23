import random
import os
import sys

# Add the parent directory of builder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from builder.framework import Sequential, DenseLayer, SigmoidLayer

MULT_MAX_INPUT = 10.0
MULT_MAX_OUTPUT = MULT_MAX_INPUT * MULT_MAX_INPUT

def normalize_mult_input(number):
    return number / MULT_MAX_INPUT

def normalize_mult_output(number):
    return number / MULT_MAX_OUTPUT

def denormalize_mult_output(number):
    return number * MULT_MAX_OUTPUT

def create_multiplication_training_data(number_of_examples):
    inputs = []
    targets = []
    for _ in range(number_of_examples):
        first_number = random.uniform(0, MULT_MAX_INPUT)
        second_number = random.uniform(0, MULT_MAX_INPUT)
        answer = first_number * second_number
        inputs.append([normalize_mult_input(first_number), normalize_mult_input(second_number)])
        targets.append([normalize_mult_output(answer)])
    return inputs, targets

def main():
    random.seed(42)
    training_inputs, training_targets = create_multiplication_training_data(1_500)

    network = Sequential(
        layers=[
            DenseLayer(input_size=2, output_size=16),
            SigmoidLayer(),
            DenseLayer(input_size=16, output_size=1),
        ],
        learning_rate=0.05,
    )

    print("Training the neural network on MULTIPLICATION...\n")
    network.train(training_inputs, training_targets, epochs=15_000, print_every=1000, batch_size=32)
    network.save("multiplication_network.json")
    print("\nTraining completed.")
    print("Type 'exit' to stop.")

    while True:
        first_text = input("\nFirst number (0-10): ").strip()
        if first_text.lower() == "exit": break
        second_text = input("Second number (0-10): ").strip()
        if second_text.lower() == "exit": break

        try:
            first_number = float(first_text)
            second_number = float(second_text)
        except ValueError:
            print("Please enter valid numbers.")
            continue

        normalized_inputs = [[normalize_mult_input(first_number), normalize_mult_input(second_number)]]
        normalized_prediction = network.predict(normalized_inputs)[0][0]
        prediction = denormalize_mult_output(normalized_prediction)

        print(f"Neural network prediction: {prediction:.4f}")
        print(f"Correct answer: {first_number * second_number:.4f}")

if __name__ == "__main__":
    main()
