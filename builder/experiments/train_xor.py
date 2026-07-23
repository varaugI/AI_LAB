import random
import os
import sys

# Add the parent directory of builder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from builder.framework import Sequential, DenseLayer, SigmoidLayer

def create_xor_training_data(number_of_examples):
    # XOR logic gate truth table
    possible_inputs = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ]
    
    inputs = []
    targets = []
    for _ in range(number_of_examples):
        inp, tgt = random.choice(possible_inputs)
        inputs.append(inp)
        targets.append(tgt)
        
    return inputs, targets

def main():
    random.seed(42)
    training_inputs, training_targets = create_xor_training_data(1000)

    # We need a hidden layer to learn XOR because it is not linearly separable.
    network = Sequential(
        layers=[
            DenseLayer(input_size=2, output_size=4),
            SigmoidLayer(),
            DenseLayer(input_size=4, output_size=1),
            SigmoidLayer(), # The final output must be 0-1 for BCE loss
        ],
        learning_rate=0.5, # Learning rate can be higher for classification
    )

    print("Training the neural network on XOR gate using BCE loss...\n")
    # Notice we pass loss_type="bce" here!
    network.train(training_inputs, training_targets, epochs=20_000, print_every=2000, loss_type="bce")
    network.save("xor_network.json")
    print("\nTraining completed.")
    print("Type 'exit' to stop.")

    while True:
        first_text = input("\nFirst input (0 or 1): ").strip()
        if first_text.lower() == "exit": break
        second_text = input("Second input (0 or 1): ").strip()
        if second_text.lower() == "exit": break

        try:
            first_number = float(first_text)
            second_number = float(second_text)
        except ValueError:
            print("Please enter valid numbers.")
            continue

        normalized_inputs = [[first_number, second_number]]
        prediction = network.predict(normalized_inputs)[0][0]

        print(f"Neural network prediction: {prediction:.4f}")
        # Threshold at 0.5 to classify
        if prediction > 0.5:
            print(f"Classification: 1")
        else:
            print(f"Classification: 0")

if __name__ == "__main__":
    main()
