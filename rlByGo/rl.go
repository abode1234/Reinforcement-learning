package main

import (
	"fmt"
	"math/rand"
	"time"
)

const (
	numStates    = 5 // Number of states
	numActions   = 3 // Number of actions
	learningRate = 0.1
	discount     = 0.9
	episodes     = 1000
)

// QTable structure: state -> action -> reward
type QTable map[int]map[int]float64

// Initialize Q-Table
func initQTable() QTable {
	qTable := make(QTable)
	for state := 0; state < numStates; state++ {
		qTable[state] = make(map[int]float64)
		for action := 0; action < numActions; action++ {
			qTable[state][action] = 0.0
		}
	}
	return qTable
}

// Choose action using epsilon-greedy
func chooseAction(qTable QTable, state int, epsilon float64) int {
	if rand.Float64() < epsilon {
		// Explore: Random action
		return rand.Intn(numActions)
	}
	// Exploit: Choose best action
	bestAction := 0
	bestValue := -1.0
	for action, value := range qTable[state] {
		if value > bestValue {
			bestValue = value
			bestAction = action
		}
	}
	return bestAction
}

// Simulate environment: reward and next state
func takeAction(state, action int) (int, float64) {
	// Dummy logic: Define rewards and state transitions
	rewards := [][]float64{
		{1, -1, 0},
		{0, 2, -1},
		{-1, 1, 2},
		{2, 0, -1},
		{-1, -1, 1},
	}
	nextState := (state + action) % numStates
	reward := rewards[state][action]
	return nextState, reward
}

// Update Q-Value
func updateQValue(qTable QTable, state, action int, reward float64, nextState int) {
	maxNextQ := -1.0
	for _, value := range qTable[nextState] {
		if value > maxNextQ {
			maxNextQ = value
		}
	}
	qTable[state][action] += learningRate * (reward + discount*maxNextQ - qTable[state][action])
}

func main() {
	rand.Seed(time.Now().UnixNano())
	qTable := initQTable()

	// Training Loop
	epsilon := 1.0
	epsilonDecay := 0.995
	minEpsilon := 0.01

	for e := 0; e < episodes; e++ {
		state := rand.Intn(numStates)
		for step := 0; step < 100; step++ {
			// Choose action
			action := chooseAction(qTable, state, epsilon)

			// Take action
			nextState, reward := takeAction(state, action)

			// Update Q-Table
			updateQValue(qTable, state, action, reward, nextState)

			state = nextState
		}
		// Decay epsilon
		if epsilon > minEpsilon {
			epsilon *= epsilonDecay
		}
	}

	// Print the learned Q-Table
	fmt.Println("Learned Q-Table:")
	for state, actions := range qTable {
		fmt.Printf("State %d: %v\n", state, actions)
	}
}

