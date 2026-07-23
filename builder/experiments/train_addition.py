import random
import os
import sys

# Add the parent directory of builder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from builder.framework import Sequential, DenseLayer, SigmoidLayer

MAXIMUM_INPUT_NUMBER = 100.0
MAXIMUM_OUTPUT_NUMBER = MAXIMUM_INPUT_NUMBER * 2.0

def normalize_input(number):
    return number / MAXIMUM_INPUT_NUMBER

def normalize_output(number):
    return number / MAXIMUM_OUTPUT_NUMBER

def denormalize_output(number):
    return number * MAXIMUM_OUTPUT_NUMBER

def create_addition_training_data(number_of_examples):
    inputs = []
    targets = []
    for _ in range(number_of_examples):
        first_number = random.uniform(0, MAXIMUM_INPUT_NUMBER)
        second_number = random.uniform(0, MAXIMUM_INPUT_NUMBER)
        answer = first_number + second_number
        inputs.append([normalize_input(first_number), normalize_input(second_number)])
        targets.append([normalize_output(answer)])
    return inputs, targets

def main():
    random.seed(42)
    training_inputs, training_targets = create_addition_training_data(1_000)

    network = Sequential(
        layers=[
            DenseLayer(input_size=2, output_size=12),
            SigmoidLayer(),
            DenseLayer(input_size=12, output_size=1),
        ],
        learning_rate=0.1,
    )

    print("Training the neural network on ADDITION...\n")
    network.train(training_inputs, training_targets, epochs=10_000, print_every=500, batch_size=32)
    network.save("addition_network.json")
    print("\nTraining completed.")
    print("Type 'exit' to stop.")

    while True:
        first_text = input("\nFirst number: ").strip()
        if first_text.lower() == "exit": break
        second_text = input("Second number: ").strip()
        if second_text.lower() == "exit": break

        try:
            first_number = float(first_text)
            second_number = float(second_text)
        except ValueError:
            print("Please enter valid numbers.")
            continue

        normalized_inputs = [[normalize_input(first_number), normalize_input(second_number)]]
        normalized_prediction = network.predict(normalized_inputs)[0][0]
        prediction = denormalize_output(normalized_prediction)

        print(f"Neural network prediction: {prediction:.4f}")
        print(f"Correct answer: {first_number + second_number:.4f}")

if __name__ == "__main__":
    main()
