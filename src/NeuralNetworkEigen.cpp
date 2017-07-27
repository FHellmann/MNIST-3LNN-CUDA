/*
 * NeuralNetworkEigen.cpp
 *
 *  Created on: Jul 26, 2017
 *      Author: xadrin
 */

#include <src/NeuralNetworkEigen.h>

#include <iostream>
using namespace std;

NeuralNetworkEigen::NeuralNetworkEigen(const int inpCount, const int hidCount,
		const int outCount, const double learningRate, size_t iterations) :
		NeuralNetwork(inpCount, hidCount, outCount, learningRate),
		numIterations(iterations) {

}

NeuralNetworkEigen::~NeuralNetworkEigen() {
}

double NeuralNetworkEigen::train(MNISTImageDataset const& images,
		MNISTLableDataset const& labels, double const training_error_threshold,
		double const max_derivation) {

	size_t const batchSize = BATCH_SIZE;

	resizeMatrices(batchSize);

	//
	// Initialize the weights
	//
	Layer* const inputLayer  = getLayer(INPUT);
	Layer* const hiddenLayer = getLayer(HIDDEN);
	Layer* const outputLayer = getLayer(OUTPUT);
	for (size_t j = 0; j < hiddenLayer->nodes.size(); ++j) {
		Layer::Node* node = hiddenLayer->nodes[j];
		bias2(j) = node->bias;
		for (size_t i = 0; i < node->weights.size(); ++i) {
			W12(j, i) = node->weights[i];
		}
	}

	for (size_t j = 0; j < outputLayer->nodes.size(); ++j) {
		Layer::Node* node = outputLayer->nodes[j];
		bias3(j) = node->bias;
		for (size_t i = 0; i < node->weights.size(); ++i) {
			W23(j, i) = node->weights[i];
		}
	}

	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixColMajor;
	MatrixColMajor imageBatch(images.front().total(), batchSize);
	MatrixColMajor labelBatch(NUM_DIGITS, batchSize);
	Eigen::MatrixXf const ones2 = Eigen::MatrixXf::Ones(output2.rows(), output2.cols());
	Eigen::MatrixXf const ones3 = Eigen::MatrixXf::Ones(output3.rows(), output3.cols());

	for (size_t it = 0; it < numIterations; ++it)
	{
		cout << "Iteration " << it << endl;

		MNISTImageDataset::const_iterator img = images.begin();
		MNISTLableDataset::const_iterator lbl = labels.begin();

		size_t i = 0;
		for (; i < images.size() / batchSize; ++i)
		{
			for (size_t b = 0; b < batchSize; ++b) {
				uint8_t* src = img->datastart;
				for (size_t k = 0; k < img->total(); ++k, ++src) {
					imageBatch(k, b) = (*src > 128) ? 1.0f : 0.0f;
				}

				for (size_t k = 0; k < NUM_DIGITS; ++k) {
					labelBatch(k, b) = 0;
					if (k == *lbl) {
						labelBatch(k, b) = 1.0f;
					}
				}
				++img;
				++lbl;
			}

			// feed forward
			output2 = bias2 * Eigen::MatrixXf::Ones(1, batchSize);
			output2 += W12 * imageBatch;
			output2 = ones2.cwiseQuotient(ones2 + Eigen::exp((-output2).array()).matrix());

			output3 = bias3 * Eigen::MatrixXf::Ones(1, batchSize);
			output3 += W23 * output2;
			output3 = ones3.cwiseQuotient(ones3 + Eigen::exp((-output3).array()).matrix());

			// backpropagate
			error3 = labelBatch - output3;
			error3 = error3.cwiseProduct(output3.cwiseProduct(ones3 - output3));

			error2 = W23.transpose() * error3;
			error2 = error2.cwiseProduct(output2.cwiseProduct(ones2 - output2));

			W12 += learningRate * error2 * imageBatch.transpose() / batchSize;
			bias2 += learningRate * error2 * Eigen::MatrixXf::Ones(error2.cols(), 1) / batchSize;

			W23 += learningRate * error3 * output2.transpose() / batchSize;
			bias3 += learningRate * error3 * Eigen::MatrixXf::Ones(error3.cols(), 1) / batchSize;
		}
	}

	//
	// Copy the weight data into the c++ data structure.
	//
	for (size_t j = 0; j < hiddenLayer->nodes.size(); ++j) {
		Layer::Node* node = hiddenLayer->nodes[j];
		node->bias = bias2(j);
		for (size_t i = 0; i < node->weights.size(); ++i) {
			node->weights[i] = W12(j, i);
		}
	}

	for (size_t j = 0; j < outputLayer->nodes.size(); ++j) {
		Layer::Node* node = outputLayer->nodes[j];
		node->bias = bias3(j);
		for (size_t i = 0; i < node->weights.size(); ++i) {
			node->weights[i] = W23(j, i);
		}
	}

	return 0.0;
}

void NeuralNetworkEigen::resizeMatrices(size_t const batchSize) {

	size_t const inputSize = getLayer(INPUT)->nodes.size();
	size_t const hiddenSize = getLayer(HIDDEN)->nodes.size();
	size_t const outputSize = getLayer(OUTPUT)->nodes.size();

	W12.resize(hiddenSize, inputSize);
	bias2.resize(hiddenSize, 1);
	output2.resize(hiddenSize, batchSize);
	error2.resize(hiddenSize, batchSize);

	W23.resize(outputSize, hiddenSize);
	bias3.resize(outputSize, 1);
	output3.resize(outputSize, batchSize);
	error3.resize(outputSize, batchSize);
}
