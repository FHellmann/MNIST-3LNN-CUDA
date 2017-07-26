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

struct GPUTrainingParameters;

class NeuralNetworkCUDA: public NeuralNetwork {
public:
	NeuralNetworkCUDA(const int inpCount, const int hidCount,
			const int outCount, const double learningRate);
	virtual ~NeuralNetworkCUDA();

	virtual void train(MNISTImageDataset const& images,
			MNISTLableDataset const& labels,
			double const training_error_threshold, double const max_derivation);

	virtual void feedForward();

private:
	float* feedForwardImage = nullptr;
	float* d_feedForwardImage = nullptr;
	GPUTrainingParameters* feedForwardParams = nullptr;

	/**
	 * Allocates memory for the public interface feedForward call on the GPU and the host.
	 * If memory is already allocated, does nothing.
	 */
	void initializeFeedForwardCUDAMemory();
	void releaseFeedForwardCUDAMemory();
};

#endif /* NEURALNETWORKCUDA_H_ */
