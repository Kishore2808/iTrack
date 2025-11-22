import os
import numpy as np
import datetime

# --- Configuration ---
DATA_DIR = 'sessions_data'
LABELS = ['distracted', 'focused']
SEQUENCE_LENGTH = 15

def analyze_data(folder_path):
    """Analyzes all sequences in a folder to find the mean and std dev for each timestep."""
    all_sequences = []
    if not os.path.exists(folder_path):
        return None, None
        
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as f:
                sequence = [float(x) for x in f.read().strip().split(',')]
                # Ensure sequence is the correct length for analysis
                if len(sequence) == SEQUENCE_LENGTH:
                    all_sequences.append(sequence)
    
    if not all_sequences:
        return None, None

    # Convert to a NumPy array for statistical analysis
    sequences_array = np.array(all_sequences)
    
    # Calculate the mean and standard deviation for each timestep (column)
    means = np.mean(sequences_array, axis=0)
    std_devs = np.std(sequences_array, axis=0)
    
    return means, std_devs

def generate_sample(means, std_devs):
    """Generates a new synthetic sample based on the given statistics."""
    new_sequence = []
    for i in range(SEQUENCE_LENGTH):
        # Generate a random value from a normal distribution for each timestep
        value = np.random.normal(loc=means[i], scale=std_devs[i])
        # Clip the value to ensure it's between 0.0 and 1.0
        clipped_value = np.clip(value, 0.0, 1.0)
        new_sequence.append(clipped_value)
    return new_sequence

def save_sample(sequence, folder_path):
    """Saves a generated sequence to a new timestamped file."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Added microseconds for rapid creation
    session_filename = os.path.join(folder_path, f"synthetic_session_{timestamp}.txt")
    
    with open(session_filename, 'w') as f:
        f.write(",".join(["{:.2f}".format(s) for s in sequence]))

if __name__ == "__main__":
    print("--- Synthetic Data Generator ---")
    
    for label in LABELS:
        folder = os.path.join(DATA_DIR, label)
        print(f"\nAnalyzing data for label: '{label}'...")
        
        means, std_devs = analyze_data(folder)
        
        if means is None:
            print(f"No data found or folder missing for '{label}'. Skipping.")
            continue
            
        try:
            num_to_generate = int(input(f"How many new '{label}' samples do you want to generate? "))
        except ValueError:
            print("Invalid number. Skipping.")
            continue

        print(f"Generating {num_to_generate} new samples for '{label}'...")
        for i in range(num_to_generate):
            new_sample = generate_sample(means, std_devs)
            save_sample(new_sample, folder)
        print("Done.")

    print("\nSynthetic data generation complete!")