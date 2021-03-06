/*
 * cudaUtility.h
 *
 *  Created on: 25.07.2017
 *      Author: buettnst
 */

#ifndef CUDAUTILITY_H_
#define CUDAUTILITY_H_

#include "NeuralNetworkCUDA.h"
#include <eigen3/Eigen/Eigen>

#define MATRIX_SIZE_DIVISOR 7

struct Matrix {
	enum Layout {
		ROW_MAJOR,
		COLUMN_MAJOR
	};

	size_t rows = 0;
	size_t cols = 0;
	Layout layout = ROW_MAJOR;
	float* data = nullptr;
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
	Matrix error3;
	Matrix error2;
};

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixRowMajor;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> EigenMatrixColumnMajor;

struct TrainingParameters {
	EigenMatrixColumnMajor images;
	EigenMatrixColumnMajor labels;

	size_t numHiddenNodes;
	size_t batchSize;

	EigenMatrixRowMajor W12;
	EigenMatrixRowMajor W23;

	EigenMatrixRowMajor bias2;
	EigenMatrixRowMajor bias3;

	EigenMatrixRowMajor output2;
	EigenMatrixRowMajor output3;

	EigenMatrixRowMajor error2;
	EigenMatrixRowMajor error3;
};

std::ostream& operator<<(std::ostream&, TrainingParameters const&);

/* Utility functions */
__global__ void mul(Matrix const C, Matrix const A, Matrix const B);
__global__ void mul_add(Matrix const C, Matrix const A, Matrix const B);
__global__ void cwise_mul_act_deriv(Matrix const C, Matrix const A, Matrix const B, NeuralNetwork::ActFctType const actFct);
__global__ void cwise_sub(Matrix const C, Matrix const A, Matrix const B);
__global__ void fill(Matrix const, float const);
__global__ void fill_pattern(Matrix const);

/* THESE FUNCTIONS ARE EVIL!!
 * ONLY USE THEM IF YOU KNOW THAT YOU DON'T HAVE TO SYNCHRONIZE YOUR GRID
 * FOR THE FOLLOWING OPERATIONS! */
__device__ void d_fill(Matrix const&, float const);
__device__ void d_fill_pattern(Matrix const&);

/* Neural network stuff */
__global__ void calculateOutputError(Matrix const error, Matrix const output, Matrix const labels, NeuralNetwork::ActFctType);
__global__ void calculateHiddenError(Matrix const error, Matrix const transposedPreviousWeights, Matrix const previousError, Matrix const hiddenOutput, NeuralNetwork::ActFctType);
__global__ void updateBias(Matrix const bias, Matrix const error);
__global__ void updateWeightsAndBias(Matrix const weights, Matrix const bias, Matrix const error, Matrix const transposedOutput, float const learningRate);
__global__ void feedForwardLayer(Matrix const input, Matrix const weights, Matrix const bias, NeuralNetwork::ActFctType actFct, Matrix const output);

__device__ void d_apply_activation(Matrix const&, NeuralNetwork::ActFctType);
__device__ void d_apply_activation_derivative(Matrix const&, NeuralNetwork::ActFctType);
__device__ void d_set_bias(Matrix const& output, Matrix const& bias);
__device__ void d_update_bias(Matrix const& bias, Matrix const& error, float const learningRate = 1.0f);

/* Matrix manipulation operations. */
__device__ void d_mul_base(Matrix const& C, Matrix const& A, Matrix const& B, void(*op)(float*, float const, float const), float const c = 1.0f);
__device__ void d_mul(Matrix const& C, Matrix const& A, Matrix const& B, float const c = 1.0f);
__device__ void d_mul_add(Matrix const& C, Matrix const& A, Matrix const& B, float const c = 1.0f);
__device__ void d_cwise_op(Matrix const& C, Matrix const& A, Matrix const& B, void(*op)(float*, float const, float const), NeuralNetwork::ActFctType const actFct = NeuralNetwork::NONE);
__device__ void d_cwise_op(Matrix const& C, Matrix const& A, float v, void(*op)(float*, float const, float const), NeuralNetwork::ActFctType const actFct = NeuralNetwork::NONE);
__device__ void d_cwise_mul(Matrix const& C, Matrix const& A, Matrix const& B);
__device__ void d_cwise_mul(Matrix const& C, Matrix const& A, float const v);
__device__ void d_cwise_mul_act_deriv(Matrix const& C, Matrix const& A, Matrix const& B, NeuralNetwork::ActFctType const actFct);
__device__ void d_cwise_sub(Matrix const& C, Matrix const& A, Matrix const& B);

/* Matrix access */
__host__ size_t matrix_size(Matrix const& A);
__host__ Matrix matrix_transpose(Matrix const& A);
__device__ float* d_matrix_pget(Matrix const& M, size_t const y, size_t const x);
__device__ float  d_matrix_get(Matrix const& M, size_t const y, size_t const x);
__device__ void   d_matrix_set(Matrix const& M, size_t const y, size_t const x, float const value);
__device__ size_t d_matrix_size(Matrix const& A);

#endif /* CUDAUTILITY_H_ */
