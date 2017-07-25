/*
 * cudaUtility.h
 *
 *  Created on: 25.07.2017
 *      Author: buettnst
 */

#ifndef CUDAUTILITY_H_
#define CUDAUTILITY_H_

#include "NeuralNetworkCUDA.h"

#define MATRIX_SIZE_DIVISOR 2

struct Matrix {
	enum Layout {
		ROW_MAJOR,
		COLUMN_MAJOR
	};

	size_t rows;
	size_t cols;
	Layout layout = ROW_MAJOR;
	float* data;
};

struct GPUTrainingParameters {
	/* Training data. */
	/** Only a reference to the global image data on the GPU. */
	Matrix images;
	/** Only a reference to the global label data on the GPU. */
	Matrix labels;

	/* Training data parameters. */
	size_t numHiddenNodes;
	/** Number of images per training. */
	size_t batchSize;
	float learningRate;

	/* Weight matrices. */
	Matrix W12;
	Matrix W23;

	/* Biases */
	Matrix bias2;
	Matrix bias3;

	/* Layer data */
	Matrix output2;
	Matrix output3;

	NeuralNetwork::ActFctType activationFunction2;
	NeuralNetwork::ActFctType activationFunction3;

	/* Training parameters. */
//	float errorThreshold;
//	float maxDerivation;

	/* Temporary buffers, e.g. for back propagation. */
	Matrix tmp3;
	Matrix tmp2;
};

/* Utility functions */
__global__ void mul(Matrix const C, Matrix const A, Matrix const B);
__global__ void fill(Matrix const, float const);
__device__ void d_fill(Matrix const&, float const);

/* Neural network stuff */
__global__ void calculateOutputError(GPUTrainingParameters const params);
__global__ void updateWeightsAndBias(Matrix const weights, Matrix const bias, Matrix const errors, Matrix const transposedLayerInput);
__global__ void calculateHiddenError(Matrix const transposedPreviousWeights, Matrix const previousErrors, Matrix const hiddenOutput, Matrix const outError, NeuralNetwork::ActFctType actFct);
__global__ void feedForwardLayer(Matrix const input, Matrix const weights, Matrix const bias, NeuralNetwork::ActFctType actFct, Matrix const output);

__device__ void d_apply_activation(Matrix const&, NeuralNetwork::ActFctType);
__device__ void d_apply_activation_derivative(Matrix const&, NeuralNetwork::ActFctType);
__device__ void d_set_bias(Matrix const& output, Matrix const& bias);
__device__ void d_update_bias(Matrix const& bias, Matrix const& error);

/* Matrix manipulation operations. */
__device__ void d_mul_base(Matrix const& C, Matrix const& A, Matrix const& B, void(*op)(float*, float const, float const));
__device__ void d_mul(Matrix const& C, Matrix const& A, Matrix const& B);
__device__ void d_mul_add(Matrix const& C, Matrix const& A, Matrix const& B);
__device__ void d_cwise_op(Matrix const& C, Matrix const& A, Matrix const& B, void(*op)(float*, float const, float const));
__device__ void d_cwise_op(Matrix const& C, Matrix const& A, float const v, void(*op)(float*, float const, float const));
__device__ void d_cwise_mul(Matrix const& C, Matrix const& A, Matrix const& B);
__device__ void d_cwise_mul(Matrix const& C, Matrix const& A, float const v);
__device__ void d_cwise_sub(Matrix const& C, Matrix const& A, Matrix const& B);

/* Matrix access */

__host__ size_t matrix_size(Matrix const& A);
__host__ Matrix matrix_transpose(Matrix const& A);
__device__ float* d_matrix_pget(Matrix const& M, size_t const y, size_t const x);
__device__ float  d_matrix_get(Matrix const& M, size_t const y, size_t const x);
__device__ void   d_matrix_set(Matrix const& M, size_t const y, size_t const x, float const value);
__device__ size_t d_matrix_size(Matrix const& A);

#endif /* CUDAUTILITY_H_ */
