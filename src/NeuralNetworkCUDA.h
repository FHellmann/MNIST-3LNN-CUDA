/*
 * NeuralNetworkCUDA.h
 *
 *  Created on: 11.07.2017
 *      Author: Stefan
 */

#ifndef NEURALNETWORKCUDA_H_
#define NEURALNETWORKCUDA_H_

#include <src/NeuralNetwork.h>

class NeuralNetworkCUDA: public NeuralNetwork {
public:
	NeuralNetworkCUDA();
	virtual ~NeuralNetworkCUDA();
};

#endif /* NEURALNETWORKCUDA_H_ */
