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
			const int outCount, const double learningRate, size_t const numIterations = 1);
	virtual ~NeuralNetworkCUDA();

	virtual double train(MNISTImageDataset const& images,
			MNISTLableDataset const& labels,
			double const training_error_threshold, double const max_derivation);

	virtual void feedForward();

private:
	float* feedForwardImage = nullptr;
	float* d_feedForwardImage = nullptr;
	GPUTrainingParameters* feedForwardParams = nullptr;

	/**
	 * Allocates memory for the public interface feedForward call on the GPU and the host.
	 * If memory is already allocated, nothing is done.
	 */
	void initializeFeedForwardCUDAMemory();
	void releaseFeedForwardCUDAMemory();

	/** Number of iterations over the dataset. */
	size_t numIterations;
};

#endif /* NEURALNETWORKCUDA_H_ */
