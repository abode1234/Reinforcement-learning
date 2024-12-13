import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ctypes
import os

# Disable CUDA to avoid GPU-related errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load the dynamic library
lib_path = os.path.join(os.path.dirname(__file__), "libqvalue.so")
lib = ctypes.CDLL(lib_path)

# Define input and output types for the library functions
lib.UpdateQValue.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.UpdateQValue.restype = ctypes.c_double

def update_q_value(current_q, reward, max_next_q, learning_rate=0.1, discount=0.9):
    """Update Q-value using the dynamic library"""
    return lib.UpdateQValue(current_q, reward, max_next_q, learning_rate, discount)

class RLModel:
    def __init__(self, num_states=100, num_actions=2):
        """Initialize the reinforcement learning model"""
        self.states = np.random.random((num_states, 4))
        self.actions = np.random.randint(0, num_actions, num_states)
        self.rewards = np.random.random(num_states)

        # Create the neural network model
        self.model = self._create_neural_network()

        # Lists for tracking results
        self.q_values_list = []
        self.loss_list = []

    def _create_neural_network(self):
        """Create the neural network"""
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(4,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(2, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def _process_state(self, state_info):
        """Process a single state"""
        state, action, reward = state_info

        # Predict Q-values
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)

        # Get the current Q-value for the action
        current_q = q_values[0][action]

        # Find the maximum Q-value for the next state
        max_next_q = max(q_values[0])

        # Update Q-value
        updated_q = update_q_value(current_q, reward, max_next_q)

        # Update the Q-values for the action
        q_values[0][action] = updated_q

        # Train the model
        history = self.model.fit(state.reshape(1, -1), q_values, verbose=0)

        return q_values[0], history.history['loss'][0]

    def train(self, num_episodes=40):
        """Train the model"""
        for episode in range(num_episodes):
            print(f"\n=== Episode {episode + 1} ===")

            episode_q_values = []
            episode_losses = []

            # Process states sequentially
            for state_info in zip(self.states, self.actions, self.rewards):
                q_values, loss = self._process_state(state_info)
                print(f"State: {state_info[0]}, Action: {state_info[1]}, Reward: {state_info[2]}, Q-Values: {q_values}, Loss: {loss}")
                episode_q_values.append(q_values)
                episode_losses.append(loss)

            # Aggregate Q-values and losses
            self.q_values_list.extend(episode_q_values)
            self.loss_list.append(sum(episode_losses))

    def visualize_results(self):
        """Visualize training results"""
        q_values_array = np.array(self.q_values_list)

        plt.figure(figsize=(12, 6))

        # Plot Q-values
        plt.subplot(1, 2, 1)
        for i in range(q_values_array.shape[1]):
            plt.plot(q_values_array[:, i], label=f"Action {i}")
        plt.title("Q-Values during Training")
        plt.xlabel("Step")
        plt.ylabel("Q-Value")
        plt.legend()

        # Plot losses
        plt.subplot(1, 2, 2)
        plt.plot(self.loss_list, label="Loss")
        plt.title("Loss during Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.show()

def main():
    # Initialize and train the model
    rl_model = RLModel()
    rl_model.train(num_episodes=40)
    rl_model.visualize_results()

if __name__ == "__main__":
    main()

