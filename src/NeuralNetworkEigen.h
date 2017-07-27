/*
 * NeuralNetworkEigen.h
 *
 *  Created on: Jul 26, 2017
 *      Author: xadrin
 */

#ifndef NEURALNETWORKEIGEN_H_
#define NEURALNETWORKEIGEN_H_

#include <src/NeuralNetwork.h>
#include <eigen3/Eigen/Eigen>

class NeuralNetworkEigen: public NeuralNetwork {
public:
	NeuralNetworkEigen(const int inpCount, const int hidCount, const int outCount,
			const double learningRate);
	virtual ~NeuralNetworkEigen();

	virtual void train(MNISTImageDataset const& images,
			MNISTLableDataset const& labels,
			double const training_error_threshold,
			double const max_derivation);

private:

	Eigen::MatrixXf W12;
	Eigen::MatrixXf bias2;
	Eigen::MatrixXf output2;
	Eigen::MatrixXf error2;

	Eigen::MatrixXf W23;
	Eigen::MatrixXf bias3;
	Eigen::MatrixXf output3;
	Eigen::MatrixXf error3;

	void initMatrices(size_t const batchSize);
};

#endif /* NEURALNETWORKEIGEN_H_ */
