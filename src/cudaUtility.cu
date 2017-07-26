/*
 * cudaUtility.cu
 *
 *  Created on: 25.07.2017
 *      Author: buettnst
 */
#include "cudaUtility.h"

#include <ostream>
using namespace std;

#undef PRINTF
//#define PRINTF(...) {if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) { printf( __VA_ARGS__ ); }}
#define PRINTF(...)

__global__ void feedForwardLayer(Matrix const input, Matrix const weights,
		Matrix const bias, NeuralNetwork::ActFctType actFct,
		Matrix const output) {
	if (blockIdx.x == 0 && blockIdx.y == 0) {
		if (threadIdx.x < input.cols && threadIdx.y < input.rows) {
			printf("input_%p(%u, %u): %f\n", input.data, threadIdx.y, threadIdx.x, d_matrix_get(input, threadIdx.y, threadIdx.x));
		}
	}

	if (blockIdx.x == 0 && blockIdx.y == 0) {
		if (threadIdx.x < bias.cols && threadIdx.y < bias.rows) {
			printf("bias_%p(%u, %u): %f\n", bias.data, threadIdx.y, threadIdx.x, d_matrix_get(bias, threadIdx.y, threadIdx.x));
		}
	}

	if (blockIdx.x == 0 && blockIdx.y == 0) {
		if (threadIdx.x < weights.cols && threadIdx.y < weights.rows) {
			printf("weights_%p(%u, %u): %f\n", weights.data, threadIdx.y, threadIdx.x, d_matrix_get(weights, threadIdx.y, threadIdx.x));
		}
	}

	d_set_bias(output, bias);
	__syncthreads();
	d_mul_add(output, weights, input);
	__syncthreads();
	if (blockIdx.x == 0 && blockIdx.y == 0) {
		if (threadIdx.x < output.cols && threadIdx.y < output.rows) {
			printf("output_%p(%u, %u): %f\n", output.data, threadIdx.y, threadIdx.x, d_matrix_get(output, threadIdx.y, threadIdx.x));
		}
	}
	d_apply_activation(output, actFct);

}

/** Saves the error in output3! */
__global__ void calculateOutputError(Matrix const error, Matrix const output, Matrix const labels,
		NeuralNetwork::ActFctType actFct) {

	/* No thread synchronization should be required in this method
	 * because it only uses component wise operations. */
	if (blockIdx.x == 0 && blockIdx.y == 0) {
		if (threadIdx.x < output.cols && threadIdx.y < output.rows) {
			printf("output(%u, %u): %f\n", threadIdx.y, threadIdx.x, d_matrix_get(output, threadIdx.y, threadIdx.x));
		}
	}

	if (blockIdx.x == 0 && blockIdx.y == 0) {
		if (threadIdx.x < labels.cols && threadIdx.y < labels.rows) {
			printf("labels(%u, %u): %f\n", threadIdx.y, threadIdx.x, d_matrix_get(labels, threadIdx.y, threadIdx.x));
		}
	}

	// Save the difference into tmp buffer
	// l - o
	d_cwise_sub(error, labels, output);

	if (blockIdx.x == 0 && blockIdx.y == 0) {
		if (threadIdx.x < error.cols && threadIdx.y < error.rows) {
			printf("error(%u, %u): %f\n", threadIdx.y, threadIdx.x, d_matrix_get(error, threadIdx.y, threadIdx.x));
		}
	}

	// Reuse the output buffer for saving the error.
	// s'(o)
	//d_apply_activation_derivative(output, actFct);
	// s'(o)(l - o)
	d_cwise_mul_act_deriv(error, error, output, actFct);
	// n s'(o)(l - o)
	//d_cwise_mul(outError, outError, learningRate);

	if (blockIdx.x == 0 && blockIdx.y == 0) {
		if (threadIdx.x < error.cols && threadIdx.y < error.rows) {
			printf("outError(%u, %u): %f\n", threadIdx.y, threadIdx.x, d_matrix_get(error, threadIdx.y, threadIdx.x));
		}
	}
}

__global__ void calculateHiddenError(Matrix const error, Matrix const transposedPreviousWeight,
		Matrix const previousError, Matrix const hiddenOutput, NeuralNetwork::ActFctType actFct) {

	// Backpropagate the error.
	d_mul(error, transposedPreviousWeight, previousError);
	//d_cwise_mul(outError, outError, learningRate);

	//d_apply_activation_derivative(hiddenOutput, actFct);
	d_cwise_mul_act_deriv(error, error, hiddenOutput, actFct);
}

__global__ void updateBias(Matrix const bias, Matrix const error) {

	d_update_bias(bias, error);
}

__global__ void updateWeightsAndBias(Matrix const weights, Matrix const bias,
		Matrix const error, Matrix const transposedLayerInput, float const learningRate) {

	/* No thread synchronization should be required in this method
	 * because the target matrices are different and the
	 * operands are constant. */
	d_mul_add(weights, error, transposedLayerInput, learningRate);
	d_update_bias(bias, error, learningRate);
}

__device__ void d_apply_activation(Matrix const& A, NeuralNetwork::ActFctType functionType) {

	PRINTF("d_activate_layer\n");

	// Target index for this thread.
	size_t const x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t const y = blockIdx.y * blockDim.y + threadIdx.y;

	// If the target index would handle an element outside of the data buffer, terminate.
	if (x >= A.cols || y >= A.rows) {
		return;
	}

	float* const dst = d_matrix_pget(A, y, x);
	switch (functionType) {
	case NeuralNetwork::SIGMOID:
		*dst = 1.0f / (1.0f + exp(-(*dst)));
		break;
	case NeuralNetwork::TANH:
		*dst = tanh(*dst);
		break;
	}
}

__device__ void d_apply_activation_derivative(Matrix const& A, NeuralNetwork::ActFctType functionType) {

	PRINTF("d_apply_activation_derivative\n");

	// Target index for this thread.
	size_t const x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t const y = blockIdx.y * blockDim.y + threadIdx.y;

	// If the target index would handle an element outside of the data buffer, terminate.
	if (x >= A.cols || y >= A.rows) {
		return;
	}

	float* const dst = d_matrix_pget(A, y, x);
	switch (functionType) {
	case NeuralNetwork::SIGMOID:
		*dst = *dst * (1.0f - *dst);
		break;
	case NeuralNetwork::TANH:
		float t = tanh(*dst);
		*dst = 1.0f - t * t;
		break;
	}
}

__device__ void d_set_bias(Matrix const& output, Matrix const& bias) {

	if (bias.rows != output.rows) {
		PRINTF("d_set_bias: Bias and output dimensions mismatch. Expected same height but bias was %lu and output was %lu\n", bias.rows, output.rows);
		return;
	}

	if (bias.cols > 1) {
		PRINTF("d_set_bias: Bias column dimension is %lu > 1. Not handled.\n", bias.cols);
		return;
	}

	size_t const targetX = threadIdx.x + blockIdx.x * blockDim.x;
	size_t const targetY = threadIdx.y + blockIdx.y * blockDim.y;

	if (targetX >= output.cols || targetY >= output.rows) {
		return;
	}

	d_matrix_set(output, targetY, targetX, d_matrix_get(bias, targetY, 0));
}

__device__ void d_assign(float* c, float const a, float const b) {
	*c = b;
}

__device__ void d_add(float* c, float const a, float const b) {
	*c = a + b;
}

__device__ void d_sub(float* c, float const a, float const b) {
	*c = a - b;
}

__device__ void d_mul(float* c, float const a, float const b) {
	*c = a * b;
}

__device__ void d_mul(Matrix const& C, Matrix const& A, Matrix const& B, float const c) {
	PRINTF("d_mul\n");
	d_mul_base(C, A, B, &d_assign, c);
}

__device__ void d_mul_add(Matrix const& C, Matrix const& A, Matrix const& B, float const c) {
	PRINTF("d_mul_add\n");
	d_mul_base(C, A, B, &d_add, c);
}

/**
 * Computes C = AB where the dimensions of A and be have to be a multiple of MATRIX_SIZE_DIVISOR.
 *
 * @param[in] A first factor of the matrix multiplication.
 * @param[in] B second factor of the multiplication.
 * @param[out] C Matrix holding the result. Must provide enough storage space.
 */
__device__ void d_mul_base(Matrix const& C, Matrix const& A, Matrix const& B, void(*op)(float*, float const, float const), float const c) {

	if (A.cols != B.rows) {

		PRINTF("d_mul_base: Incompatible matrices: (%lu, %lu) x (%lu, %lu)\n", A.rows, A.cols, B.rows, B.cols);
		return;
	}

	// The block caches are row major.
	__shared__ float blockCacheA[MATRIX_SIZE_DIVISOR][MATRIX_SIZE_DIVISOR];
	__shared__ float blockCacheB[MATRIX_SIZE_DIVISOR][MATRIX_SIZE_DIVISOR];

	// Compute the target coordinates.
	size_t const blockX = blockIdx.x * MATRIX_SIZE_DIVISOR;
	size_t const blockY = blockIdx.y * MATRIX_SIZE_DIVISOR;

	// If this block does not completely lie within the destination matrix,
	// exit. If it parly lies within, we need some threads for loading data.
	if (blockX >= C.cols || blockY >= C.rows) {
		return;
	}

	size_t const x = blockX + threadIdx.x;
	size_t const y = blockY + threadIdx.y;

//	if (A.cols % MATRIX_SIZE_DIVISOR != 0 || B.rows % MATRIX_SIZE_DIVISOR != 0) {
//		printf("d_mul_base: A's cols is not a multiple of %u: (%lu, %lu) x (%lu, %lu)\n", MATRIX_SIZE_DIVISOR, A.rows, A.cols, B.rows, B.cols);
//		return;
//	}

	float threadValue = 0.0f;
	unsigned int const numSubBlocks = (A.cols - 1) / MATRIX_SIZE_DIVISOR + 1;
	for (size_t k = 0; k < numSubBlocks; ++k)
	{
		size_t const xA = k * MATRIX_SIZE_DIVISOR + threadIdx.x;
		if (xA < A.cols && y < A.rows) {
			blockCacheA[threadIdx.y][threadIdx.x] = d_matrix_get(A, y, xA);
		} else {
			blockCacheA[threadIdx.y][threadIdx.x] = 0.0f;
		}

		size_t const yB = k * MATRIX_SIZE_DIVISOR + threadIdx.y;
		if (yB < B.rows && x < B.cols) {
			blockCacheB[threadIdx.y][threadIdx.x] = d_matrix_get(B, yB, x);
		} else {
			blockCacheB[threadIdx.y][threadIdx.x] = 0.0f;
		}

		__syncthreads();

		#pragma unroll
		for (size_t i = 0; i < MATRIX_SIZE_DIVISOR; ++i)
		{
			threadValue += blockCacheA[threadIdx.y][i] * blockCacheB[i][threadIdx.x];
		}

		__syncthreads();
	}

	// If this thread has nothing to do, because it would access invalid memory, exit
	if (x >= C.cols || y >= C.rows) {
		return;
	}

	float* const pValue = d_matrix_pget(C, y, x);
	op(pValue, *pValue, threadValue * c);
}

__device__ void d_cwise_sub(Matrix const& C, Matrix const& A, Matrix const& B) {
	d_cwise_op(C, A, B, &d_sub);
}

__device__ void d_cwise_mul(Matrix const& C, Matrix const& A, Matrix const& B) {
	d_cwise_op(C, A, B, &d_mul);
}

__device__ void d_cwise_mul_act_deriv(Matrix const& C, Matrix const& A, Matrix const& B, NeuralNetwork::ActFctType const actFct) {
	d_cwise_op(C, A, B, &d_mul, actFct);
}

__device__ void d_cwise_op(Matrix const& C, Matrix const& A, Matrix const& B, void(*op)(float*, float const, float const), NeuralNetwork::ActFctType const actFct) {

	if (A.cols != B.cols || A.rows != B.rows || B.cols != C.cols || B.rows != C.rows) {

		PRINTF("d_cwise_op: Incompatible matrices: (%lu, %lu) + (%lu, %lu) = (%lu, %lu)\n", A.rows, A.cols, B.rows, B.cols, C.rows, C.cols);
		return;
	}

	size_t const x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t const y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= A.cols || y >= A.rows) {
		return;
	}

	float b =  d_matrix_get(B, y, x);
	switch (actFct) {
	case NeuralNetwork::SIGMOID:
		b = b * (1.0f - b);
		break;
	case NeuralNetwork::TANH:
		b = 1.0f - b * b;
		break;
	}

	op(d_matrix_pget(C, y, x), d_matrix_get(A, y, x), b);
}

__device__ void d_cwise_mul(Matrix const& C, Matrix const& A, float const v) {
	d_cwise_op(C, A, v, &d_mul);
}

__device__ void d_cwise_op(Matrix const& C, Matrix const& A, float v, void(*op)(float*, float const, float const), NeuralNetwork::ActFctType const actFct) {

	if (A.cols != C.cols || A.rows != C.rows) {

		PRINTF("d_cwise_op: Incompatible matrices: v * (%lu, %lu) = (%lu, %lu)\n", A.rows, A.cols, C.rows, C.cols);
		return;
	}

	size_t const x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t const y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= A.cols || y >= A.rows) {
		return;
	}

	switch (actFct) {
	case NeuralNetwork::SIGMOID:
		v = v * (1.0f - v);
		break;
	case NeuralNetwork::TANH:
		v = 1.0f - v * v;
		break;
	}

	op(d_matrix_pget(C, y, x), d_matrix_get(A, y, x), v);
}

__global__ void fill(Matrix const A, float const v) {
	d_fill(A, v);
}

__global__ void fill_pattern(Matrix const A) {

	d_fill_pattern(A);
}

__device__ void d_fill(Matrix const& A, float const v) {

	size_t const targetX = threadIdx.x + blockIdx.x * blockDim.x;
	size_t const targetY = threadIdx.y + blockIdx.y * blockDim.y;

	if (targetX >= A.cols || targetY >= A.rows) {
		return;
	}

	d_matrix_set(A, targetY, targetX, v);
}

__device__ void d_fill_pattern(Matrix const& A) {

	d_fill(A, threadIdx.y + blockIdx.y * blockDim.y);
}

__device__ void d_update_bias(Matrix const& bias, Matrix const& error, float const c) {

	if (bias.rows != error.rows|| bias.cols != 1) {

		PRINTF("d_update_bias: Invalid matrices: bias(%lu, %lu) error(%lu, %lu)\n", bias.rows, bias.cols, error.rows, error.cols);
		return;
	}

	// The block caches are row major.
	__shared__ float blockCacheError[MATRIX_SIZE_DIVISOR][MATRIX_SIZE_DIVISOR];

	size_t const blockX = blockIdx.x * MATRIX_SIZE_DIVISOR;
	size_t const blockY = blockIdx.y * MATRIX_SIZE_DIVISOR;

	if (blockX >= bias.cols || blockY >= bias.rows) {
		return;
	}

	// Compute the target coordinates.
	size_t const x = blockX + threadIdx.x;
	size_t const y = blockY + threadIdx.y;

	float threadValue = 0.0f;
	unsigned int const numSubBlocks = (error.cols - 1) / MATRIX_SIZE_DIVISOR + 1;
	for (int k = 0; k < numSubBlocks; ++k)
	{
		size_t const xA = k * MATRIX_SIZE_DIVISOR + threadIdx.x;
		if (xA < error.cols && y < error.rows) {
			blockCacheError[threadIdx.y][threadIdx.x] = d_matrix_get(error, y, xA);
		} else {
			blockCacheError[threadIdx.y][threadIdx.x] = 0.0f;
		}

		__syncthreads();

		#pragma unroll
		for (size_t i = 0; i < MATRIX_SIZE_DIVISOR; ++i)
		{
			threadValue += blockCacheError[threadIdx.y][i];
		}

		__syncthreads();
	}

	// If this thread has nothing to do, because it would access invalid memory, exit
	if (y >= bias.rows || x >= bias.cols) {
		return;
	}

	*d_matrix_pget(bias, y, x) += threadValue * c;
}

__device__ float* d_matrix_pget(Matrix const& M, size_t const y, size_t const x) {
	if (M.layout == Matrix::ROW_MAJOR) {
		return M.data + (x + y * M.cols);
	} else {
		return M.data + (x * M.rows + y);
	}
}

__device__ float d_matrix_get(Matrix const& M, size_t const y, size_t const x) {
	return *d_matrix_pget(M, y, x);
}

__device__ void d_matrix_set(Matrix const& M, size_t const y, size_t const x, float const value) {
	*d_matrix_pget(M, y, x) = value;
}

__device__ size_t d_matrix_size(Matrix const& A) {
	return A.rows * A.cols;
}

__global__ void mul(Matrix const C, Matrix const A, Matrix const B) {
	d_mul(C, A, B);
}

__global__ void mul_add(Matrix const C, Matrix const A, Matrix const B) {
	d_mul_add(C, A, B);
}

size_t matrix_size(Matrix const& A) {
	return A.rows * A.cols;
}

Matrix matrix_transpose(Matrix const& A) {
	Matrix T;
	T.rows = A.cols;
	T.cols = A.rows;
	T.layout = Matrix::ROW_MAJOR;
	T.data = A.data;
	if (A.layout == Matrix::ROW_MAJOR) {
		T.layout = Matrix::COLUMN_MAJOR;
	}
	return T;
}

ostream& operator<<(ostream& out, TrainingParameters const& params) {

//	out << "images: " << endl;
//	out << params.images << endl << endl;
//	out << "labels: " << endl;
//	out << params.labels << endl << endl;

	out << "bias2: " << endl;
	out << params.bias2 << endl << endl;
	out << "W12: " << endl;
	out << params.W12 << endl << endl;
	out << "output2: " << endl;
	out << params.output2 << endl << endl;
	out << "error2: " << endl;
	out << params.error2 << endl << endl;

	out << "bias3: " << endl;
	out << params.bias3 << endl << endl;
	out << "W23: " << endl;
	out << params.W23 << endl << endl;
	out << "output3: " << endl;
	out << params.output3 << endl << endl;
	out << "error3: " << endl;
	out << params.error3 << endl << endl;

	return out;


}
