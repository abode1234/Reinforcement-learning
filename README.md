
# 🤖 Cross-Language Reinforcement Learning Simulator

## Project Overview

This project demonstrates an innovative approach to reinforcement learning by creating a unique system that bridges Go and Python, showcasing advanced machine learning techniques through a custom Q-learning implementation.

## Technical Architecture

### Core Technologies
- **Languages**: Go and Python
- **Machine Learning**: TensorFlow
- **Visualization**: Matplotlib
- **Interoperability**: Dynamic Library Linking

## Detailed Code Breakdown

### Go Component (`main.go`): Q-Value Computation Engine

The Go file contains the core Q-learning update logic:

```go
func UpdateQValue(currentQ, reward, maxNextQ, learningRate, discount C.double) C.double {
    // Q-learning update formula implementation
    var updatedQ float64
    if maxNextQFloat > 0 {
        updatedQ = currentQFloat + learningRateFloat * (rewardFloat + discountFloat * maxNextQFloat - currentQFloat)
    } else {
        updatedQ = currentQFloat + learningRateFloat * (rewardFloat - currentQFloat)
    }
    return C.double(updatedQ)
}
```

#### Q-Learning Update Formula Explained
- `Q(s,a) = Q(s,a) + α * (R + γ * max(Q(s')) - Q(s,a))`
  - `Q(s,a)`: Current Q-value
  - `α`: Learning rate
  - `R`: Reward
  - `γ`: Discount factor

### Python Component (`main.py`): Learning Ecosystem

#### Neural Network Architecture
```python
def _create_neural_network(self):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(4,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
```

#### Key Neural Network Characteristics
- Input Layer: 4 features
- Hidden Layer: 32 ReLU neurons
- Output Layer: 2 actions with linear activation
- Optimizer: Adam
- Loss Function: Mean Squared Error

#### Dynamic Library Integration
```python
lib = ctypes.CDLL(lib_path)
lib.UpdateQValue.argtypes = [
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double
]
```

## Learning Process Mechanics

### Training Workflow
1. Generate random states
2. Predict initial Q-values
3. Update Q-values using Go library function
4. Retrain neural network
5. Track performance metrics

### State Processing Method
```python
def _process_state(self, state_info):
    # Predict Q-values
    q_values = self.model.predict(state.reshape(1, -1), verbose=0)
    
    # Update Q-value using Go library
    updated_q = update_q_value(current_q, reward, max_next_q)
    
    # Train the model
    history = self.model.fit(state.reshape(1, -1), q_values, verbose=0)
```

## Visualization Strategy

The project creates a dual-panel matplotlib visualization:
- Left Panel: Q-Values Progression
- Right Panel: Loss Trajectory

## System Requirements

### Prerequisites
- Go 1.16+
- Python 3.8+
- TensorFlow
- NumPy
- Matplotlib
- C compiler

### Installation & Running

```bash
# Compile Go dynamic library
go build -buildmode=c-shared -o libqvalue.so main.go

# Run Python script
python main.py
```

## Computational Considerations

### Performance Optimizations
- CUDA disabled for consistent results
- Minimal overhead cross-language calls
- Efficient NumPy/TensorFlow computations

## Limitations & Future Improvements

### Current Constraints
- Synthetic training data
- Simplified learning environment
- Basic neural network architecture

### Potential Development Vectors
- Integrate real-world state spaces
- Implement more complex reward structures
- Develop advanced neural network designs
- Create more sophisticated environment interactions

## Experimental Insights

- Demonstrates language interoperability
- Showcases machine learning fundamentals
- Highlights dynamic library usage
- Illustrates neural network adaptability

