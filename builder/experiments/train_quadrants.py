import random
import os
import sys

# Add the parent directory of builder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from builder.framework import Sequential, DenseLayer, ReLULayer, SoftmaxLayer

def create_quadrant_training_data(number_of_examples):
    inputs = []
    targets = []
    
    for _ in range(number_of_examples):
        # Generate random coordinates between -10 and 10
        x = random.uniform(-10.0, 10.0)
        y = random.uniform(-10.0, 10.0)
        
        inputs.append([x, y])
        
        # Classify the quadrant (one-hot encoding)
        if x >= 0 and y >= 0:
            targets.append([1.0, 0.0, 0.0, 0.0]) # Top Right
        elif x < 0 and y >= 0:
            targets.append([0.0, 1.0, 0.0, 0.0]) # Top Left
        elif x < 0 and y < 0:
            targets.append([0.0, 0.0, 1.0, 0.0]) # Bottom Left
        elif x >= 0 and y < 0:
            targets.append([0.0, 0.0, 0.0, 1.0]) # Bottom Right

    return inputs, targets

def main():
    random.seed(42)
    training_inputs, training_targets = create_quadrant_training_data(2000)

    # 4 classes means output_size = 4, followed by Softmax
    network = Sequential(
        layers=[
            DenseLayer(input_size=2, output_size=12),
            ReLULayer(), # ReLU is great for coordinate data
            DenseLayer(input_size=12, output_size=4),
            SoftmaxLayer(),
        ],
        learning_rate=0.05,
    )

    print("Training the neural network on Quadrant Classification using CCE loss...\n")
    network.train(training_inputs, training_targets, epochs=10_000, print_every=1000, loss_type="cce")
    network.save("quadrant_network.json")
    print("\nTraining completed.")
    print("Type 'exit' to stop.")

    classes = ["Top Right", "Top Left", "Bottom Left", "Bottom Right"]

    while True:
        x_text = input("\nEnter X coordinate (-10 to 10): ").strip()
        if x_text.lower() == "exit": break
        y_text = input("Enter Y coordinate (-10 to 10): ").strip()
        if y_text.lower() == "exit": break

        try:
            x_coord = float(x_text)
            y_coord = float(y_text)
        except ValueError:
            print("Please enter valid numbers.")
            continue

        normalized_inputs = [[x_coord, y_coord]]
        prediction = network.predict(normalized_inputs)[0] # List of 4 probabilities

        print(f"Neural network prediction probabilities: {['%.4f' % p for p in prediction]}")
        
        # Find the index of the highest probability
        predicted_class_idx = prediction.index(max(prediction))
        
        print(f"Classification: {classes[predicted_class_idx]}")

if __name__ == "__main__":
    main()
