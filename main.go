package main

/*
#include <stdlib.h>
*/
import "C"

//export UpdateQValue
func UpdateQValue(currentQ, reward, maxNextQ, learningRate, discount C.double) C.double {
    currentQFloat := float64(currentQ)
    rewardFloat := float64(reward)
    maxNextQFloat := float64(maxNextQ)
    learningRateFloat := float64(learningRate)
    discountFloat := float64(discount)

    var updatedQ float64
    if maxNextQFloat > 0 {
        updatedQ = currentQFloat + learningRateFloat * (rewardFloat + discountFloat * maxNextQFloat - currentQFloat)
    } else {
        updatedQ = currentQFloat + learningRateFloat * (rewardFloat - currentQFloat)
    }

    return C.double(updatedQ)
}

func main() {}
