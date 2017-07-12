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

	virtual void train(MNISTImageDataset const& images,
			MNISTLableDataset const& labels,
			double const training_error_threshold, double const max_derivation);
};

#endif /* NEURALNETWORKCUDA_H_ */
