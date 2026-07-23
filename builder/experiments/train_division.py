import random
import os
import sys

# Add the parent directory of builder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from builder.framework import Sequential, DenseLayer, SigmoidLayer

MAX_VAL = 100.0

def normalize_input(number):
    return number / MAX_VAL

def normalize_output(number):
    # Output can be from 0 to MAX_VAL
    return number / MAX_VAL

def denormalize_output(number):
    return number * MAX_VAL

def create_division_training_data(number_of_examples):
    inputs = []
    targets = []
    for _ in range(number_of_examples):
        first_number = random.uniform(0, MAX_VAL)
        second_number = random.uniform(1.0, MAX_VAL) # avoid zero div
        answer = first_number / second_number
        inputs.append([normalize_input(first_number), normalize_input(second_number)])
        targets.append([normalize_output(answer)])
    return inputs, targets

def main():
    random.seed(42)
    training_inputs, training_targets = create_division_training_data(2_000)

    network = Sequential(
        layers=[
            DenseLayer(input_size=2, output_size=16),
            SigmoidLayer(),
            DenseLayer(input_size=16, output_size=1),
        ],
        learning_rate=0.05,
    )

    print("Training the neural network on DIVISION...\n")
    network.train(training_inputs, training_targets, epochs=15_000, print_every=1000)
    network.save("division_network.json")
    print("\nTraining completed.")
    print("Type 'exit' to stop.")

    while True:
        first_text = input("\nFirst number (0-100): ").strip()
        if first_text.lower() == "exit": break
        second_text = input("Second number (1-100): ").strip()
        if second_text.lower() == "exit": break

        try:
            first_number = float(first_text)
            second_number = float(second_text)
            if second_number == 0:
                print("Cannot divide by zero!")
                continue
        except ValueError:
            print("Please enter valid numbers.")
            continue

        normalized_inputs = [[normalize_input(first_number), normalize_input(second_number)]]
        normalized_prediction = network.predict(normalized_inputs)[0][0]
        prediction = denormalize_output(normalized_prediction)

        print(f"Neural network prediction: {prediction:.4f}")
        print(f"Correct answer: {first_number / second_number:.4f}")

if __name__ == "__main__":
    main()
