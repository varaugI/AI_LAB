import random
import os
import sys

# Add the parent directory of builder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from builder.framework import Sequential, DenseLayer, ReLULayer, DropoutLayer

def main():
    # Simple network to test serialization and dropout
    network = Sequential(
        layers=[
            DenseLayer(input_size=10, output_size=20),
            ReLULayer(),
            DropoutLayer(rate=0.2),
            DenseLayer(input_size=20, output_size=2)
        ],
        learning_rate=0.01,
        optimizer="adam"
    )

    print("Created network with Dropout.")
    network.summary()
    
    print("\nSaving network architecture to 'test_dropout.json'...")
    network.save("test_dropout.json")
    
    print("Loading network from 'test_dropout.json' using Sequential.from_file()...")
    loaded_network = Sequential.from_file("test_dropout.json")
    
    print("\nLoaded network summary:")
    loaded_network.summary()
    
    print("\nSuccess! Architecture serialization and DropoutLayer are working.")

if __name__ == "__main__":
    main()
