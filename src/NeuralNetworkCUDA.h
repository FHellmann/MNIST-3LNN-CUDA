/*
 * NeuralNetworkCUDA.h
 *
 *  Created on: 11.07.2017
 *      Author: Stefan
 */

#ifndef NEURALNETWORKCUDA_H_
#define NEURALNETWORKCUDA_H_

#include "NeuralNetwork.h"

//#define PRINTF(...) {printf(__VA_ARGS__);}
#define PRINTF(...)

class NeuralNetworkCUDA: public NeuralNetwork {
public:
	NeuralNetworkCUDA(const int inpCount, const int hidCount,
			const int outCount, const double learningRate);
	virtual ~NeuralNetworkCUDA();

	virtual void train(MNISTImageDataset const& images,
			MNISTLableDataset const& labels,
			double const training_error_threshold, double const max_derivation);

	virtual void feedForward();
};

#endif /* NEURALNETWORKCUDA_H_ */
