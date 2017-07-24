#include "NeuralNetworkCUDA.h"

#include <iostream>
#include <cmath>

using namespace std;

#define PRINTF(...) {if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) { printf( __VA_ARGS__ ); }}
//#define PRINTF(...)

__host__ NeuralNetworkCUDA::NeuralNetworkCUDA(const int inpCount,
		const int hidCount, const int outCount, const double learningRate) :
		NeuralNetwork(inpCount, hidCount, outCount, learningRate) {
}

__host__ NeuralNetworkCUDA::~NeuralNetworkCUDA() {
}

#define MATRIX_SIZE_DIVISOR 28
#define NUM_DIGITS 10

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
	float* images;
	float* labels;

	/* Training data parameters. */
	size_t numExamples;
	size_t numHiddenNodes;
	size_t width;
	size_t height;
	// Number of images per training
	size_t batchSize;

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
	float errorThreshold;
	float maxDerivation;
	float learningRate;

	/* Temporary buffers, e.g. for back propagation. */
	Matrix tmp3;
	Matrix tmp2;
};

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

size_t matrix_size(Matrix const& A) {
	return A.rows * A.cols;
}

__device__ Matrix d_matrix_transpose(Matrix const& A) {
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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void d_feed_forward(GPUTrainingParameters const);
__global__ void d_back_propagate(GPUTrainingParameters const);

__host__ void NeuralNetworkCUDA::train(MNISTImageDataset const& images,
		MNISTLableDataset const& labels, double const training_error_threshold,
		double const max_derivation) {

	if (images.size() <= 0)
		return;
	if (labels.size() <= 0)
		return;

	Layer* const inputLayer  = getLayer(INPUT);
	Layer* const hiddenLayer = getLayer(HIDDEN);
	Layer* const outputLayer = getLayer(OUTPUT);

	GPUTrainingParameters trainingParams;
	trainingParams.numExamples = images.size();
	trainingParams.width = images.front().cols;
	trainingParams.height = images.front().rows;
	trainingParams.numHiddenNodes = hiddenLayer->nodes.size();
	trainingParams.learningRate = learningRate;
	trainingParams.errorThreshold = training_error_threshold;
	trainingParams.maxDerivation = max_derivation;
	trainingParams.batchSize = MATRIX_SIZE_DIVISOR;

	cudaError_t err;

	//
	// Allocate cuda memory
	//
	size_t const singleImgPixCount = images.front().total();
	size_t const allImgBufElements = singleImgPixCount * images.size();

	// Images
	err = cudaMalloc((void**) &trainingParams.images, allImgBufElements * sizeof(float));
	assert(err == cudaSuccess);

	// Labels
	err = cudaMalloc((void**) &trainingParams.labels, labels.size() * sizeof(float));
	assert(err == cudaSuccess);

	// Storage for the first weight matrix
	trainingParams.W12.rows = hiddenLayer->nodes.size();
	trainingParams.W12.cols = inputLayer->nodes.size();
	trainingParams.W12.layout = Matrix::ROW_MAJOR;
	err = cudaMalloc((void**) &trainingParams.W12.data, matrix_size(trainingParams.W12) * sizeof(float));
	assert(err == cudaSuccess);

	// Storage for the hidden layer bias vector
	trainingParams.bias2.rows = hiddenLayer->nodes.size();
	trainingParams.bias2.cols = 1;
	trainingParams.bias2.layout = Matrix::ROW_MAJOR;
	err = cudaMalloc((void**) &trainingParams.bias2.data, matrix_size(trainingParams.bias2) * sizeof(float));
	assert(err == cudaSuccess);

	// Storage for the second weight matrix
	trainingParams.W23.rows = outputLayer->nodes.size();
	trainingParams.W23.cols = hiddenLayer->nodes.size();
	trainingParams.W23.layout = Matrix::ROW_MAJOR;
	err = cudaMalloc((void**) &trainingParams.W23.data, matrix_size(trainingParams.W23) * sizeof(float));
	assert(err == cudaSuccess);

	// Storage for the output layer bias vector
	trainingParams.bias3.rows = outputLayer->nodes.size();
	trainingParams.bias3.cols = 1;
	trainingParams.bias3.layout = Matrix::ROW_MAJOR;
	err = cudaMalloc((void**) &trainingParams.bias3.data, matrix_size(trainingParams.bias3) * sizeof(float));
	assert(err == cudaSuccess);

	// Storage for the output layer output vectors
	trainingParams.output2.rows = trainingParams.numHiddenNodes;
	trainingParams.output2.cols = trainingParams.batchSize;
	trainingParams.output2.layout = Matrix::ROW_MAJOR;
	err = cudaMalloc((void**) &trainingParams.output2.data, matrix_size(trainingParams.output2) * sizeof(float));
	assert(err == cudaSuccess);

	// Storage for the output layer output vectors
	trainingParams.output3.rows = outputLayer->nodes.size();
	trainingParams.output3.cols = trainingParams.batchSize;
	trainingParams.output3.layout = Matrix::ROW_MAJOR;
	err = cudaMalloc((void**) &trainingParams.output3.data, matrix_size(trainingParams.output3) * sizeof(float));
	assert(err == cudaSuccess);

	// Temporary storage of the size of the output layer output vectors
	trainingParams.tmp3.rows = outputLayer->nodes.size();
	trainingParams.tmp3.cols = trainingParams.batchSize;
	trainingParams.tmp3.layout = Matrix::ROW_MAJOR;
	err = cudaMalloc((void**) &trainingParams.tmp3.data, matrix_size(trainingParams.tmp3) * sizeof(float));
	assert(err == cudaSuccess);

	// Temporary storage of the size of the hidden layer output vectors
	trainingParams.tmp2.rows = hiddenLayer->nodes.size();
	trainingParams.tmp2.cols = trainingParams.batchSize;
	trainingParams.tmp2.layout = Matrix::ROW_MAJOR;
	err = cudaMalloc((void**) &trainingParams.tmp2.data, matrix_size(trainingParams.tmp2) * sizeof(float));
	assert(err == cudaSuccess);

	//
	// Collect memory in RAM
	//
	float* imgData = new float[allImgBufElements];
	float* dst = imgData;
	for (cv::Mat const& img : images) {
		for (uint8_t* src = img.datastart; src != img.dataend;) {
			*(dst++) = static_cast<float>(*(src++));
		}
	}

	float* flabels = new float[labels.size()];
	dst = flabels;
	for (uint8_t const& l : labels) {
		*(dst++) = static_cast<float>(l);
	}

	float* W12 = new float[matrix_size(trainingParams.W12)];
	float* W23 = new float[matrix_size(trainingParams.W23)];
	float* bias2 = new float[matrix_size(trainingParams.bias2)];
	float* bias3 = new float[matrix_size(trainingParams.bias3)];

	//
	// Collect the initial weights and biases in buffers for submission to the GPU.
	//
	trainingParams.activationFunction2 = hiddenLayer->actFctType;
	{
		size_t k = 0;
		for (size_t j = 0; j < hiddenLayer->nodes.size(); ++j) {
			Layer::Node* node = hiddenLayer->nodes[j];
			bias2[j] = node->bias;
			for (size_t i = 0; i < node->weights.size(); ++i, ++k) {
				W12[k] = node->weights[i];
			}
		}
	}

	trainingParams.activationFunction3 = outputLayer->actFctType;
	{
		size_t k = 0;
		for (size_t j = 0; j < outputLayer->nodes.size(); ++j) {
			Layer::Node* node = outputLayer->nodes[j];
			bias3[j] = node->bias;
			for (size_t i = 0; i < node->weights.size(); ++i, ++k) {
				W23[k] = node->weights[i];
			}
		}
	}

	//
	// Copy data to graphics card
	//
	err = cudaMemcpy(trainingParams.images, imgData, allImgBufElements * sizeof(float), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);
	err = cudaMemcpy(trainingParams.labels, flabels, labels.size() * sizeof(float), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);
	err = cudaMemcpy(trainingParams.W12.data, W12, matrix_size(trainingParams.W12) * sizeof(float), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);
	err = cudaMemcpy(trainingParams.bias2.data, bias2, matrix_size(trainingParams.bias2) * sizeof(float), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);
	err = cudaMemcpy(trainingParams.W23.data, W23, matrix_size(trainingParams.W23) * sizeof(float), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);
	err = cudaMemcpy(trainingParams.bias3.data, bias3, matrix_size(trainingParams.bias3) * sizeof(float), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);

	delete[] imgData;
	imgData = nullptr;
	delete[] flabels;
	flabels = nullptr;

	// Configure Grid, i.e. setup Blocks and Threads
	dim3 numBlocks(MATRIX_SIZE_DIVISOR, MATRIX_SIZE_DIVISOR);
	dim3 threadsPerBlock(MATRIX_SIZE_DIVISOR, MATRIX_SIZE_DIVISOR);
	cout << "Blocks:            (" << numBlocks.x << ", " << numBlocks.y << ")"
			<< endl;
	cout << "Threads per block: (" << threadsPerBlock.x << ", "
			<< threadsPerBlock.y << ")" << endl;

	// Call graphics card functions
	d_feed_forward<<<numBlocks, threadsPerBlock>>>(trainingParams);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	d_back_propagate<<<numBlocks, threadsPerBlock>>>(trainingParams);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	//
	// Retreive the data
	//

	// Copy it back to neural network data structure
	err = cudaMemcpy(W12, trainingParams.W12.data, matrix_size(trainingParams.W12) * sizeof(float), cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);
	err = cudaMemcpy(W23, trainingParams.W23.data, matrix_size(trainingParams.W23) * sizeof(float), cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);
	err = cudaMemcpy(bias2, trainingParams.bias2.data, matrix_size(trainingParams.bias2) * sizeof(float), cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);
	err = cudaMemcpy(bias3, trainingParams.bias3.data, matrix_size(trainingParams.bias3) * sizeof(float), cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);

	// Free the cuda buffers
	cudaFree (trainingParams.images);
	trainingParams.images = nullptr;
	cudaFree (trainingParams.labels);
	trainingParams.labels = nullptr;
	cudaFree (trainingParams.W12.data);
	trainingParams.W12.data = nullptr;
	cudaFree (trainingParams.W23.data);
	trainingParams.W23.data = nullptr;
	cudaFree (trainingParams.bias2.data);
	trainingParams.bias2.data = nullptr;
	cudaFree (trainingParams.bias3.data);
	trainingParams.bias3.data = nullptr;
	cudaFree (trainingParams.output2.data);
	trainingParams.output2.data = nullptr;
	cudaFree (trainingParams.output3.data);
	trainingParams.output3.data = nullptr;
	cudaFree (trainingParams.tmp3.data);
	trainingParams.tmp3.data = nullptr;
	cudaFree (trainingParams.tmp2.data);
	trainingParams.tmp2.data = nullptr;

	//
	// Copy the weight data into the c++ data structure.
	//
	trainingParams.activationFunction2 = hiddenLayer->actFctType;
	{
		size_t k = 0;
		for (size_t j = 0; j < hiddenLayer->nodes.size(); ++j) {
			Layer::Node* node = hiddenLayer->nodes[j];
			node->bias = bias2[j];
			for (size_t i = 0; i < node->weights.size(); ++i) {
				node->weights[i] = W12[k];
				++k;
			}
		}
	}

	trainingParams.activationFunction3 = outputLayer->actFctType;
	{
		size_t k = 0;
		for (size_t j = 0; j < outputLayer->nodes.size(); ++j) {
			Layer::Node* node = outputLayer->nodes[j];
			node->bias = bias3[j];
			for (size_t i = 0; i < node->weights.size(); ++i) {
				node->weights[i] = W23[k];
				++k;
			}
		}
	}

	// Delete the host buffers
	delete[] W12;
	W12 = nullptr;
	delete[] W23;
	W23 = nullptr;
	delete[] bias2;
	bias2 = nullptr;
	delete[] bias3;
	bias3 = nullptr;
}

/* Matrix manipulation operations. */
__device__ void d_mul_base(Matrix const& C, Matrix const& A, Matrix const& B, void(*op)(float*, float const, float const));
__device__ void d_mul(Matrix const& C, Matrix const& A, Matrix const& B);
__device__ void d_mul_add(Matrix const& C, Matrix const& A, Matrix const& B);
__device__ void d_cwise_op(Matrix const& C, Matrix const& A, Matrix const& B, void(*op)(float*, float const, float const));
__device__ void d_cwise_op(Matrix const& C, Matrix const& A, float const v, void(*op)(float*, float const, float const));
__device__ void d_cwise_mul(Matrix const& C, Matrix const& A, Matrix const& B);
__device__ void d_cwise_mul(Matrix const& C, Matrix const& A, float const v);
__device__ void d_cwise_sub(Matrix const& C, Matrix const& A, Matrix const& B);

/* Neural network operations. */
__device__ void d_apply_activation(Matrix const&, NeuralNetwork::ActFctType);
__device__ void d_apply_activation_derivative(Matrix const&, NeuralNetwork::ActFctType);
__device__ void d_back_propagate_output(GPUTrainingParameters const&);
__device__ void d_back_propagate_hidden(GPUTrainingParameters const&);
__device__ void d_fill_target_output(GPUTrainingParameters const&, Matrix const&);
__device__ void d_set_bias(Matrix const& output, Matrix const& bias);
__device__ void d_fill_random(Matrix const&);
__device__ void d_update_bias(Matrix const& bias, Matrix const& error);

__global__ void d_feed_forward(GPUTrainingParameters const params) {

	PRINTF("d_feed_forward\n");

	Matrix imgs;
	imgs.rows = params.width * params.height;
	imgs.cols = params.batchSize;
	imgs.layout = Matrix::COLUMN_MAJOR;
	imgs.data = params.images; // Global data pointer, column major, yields one image in each column vector.

	d_set_bias(params.output2, params.bias2);
	d_mul_add(params.output2, params.W12, imgs);
	d_apply_activation(params.output2, params.activationFunction2);

	d_set_bias(params.output3, params.bias3);
	d_mul_add(params.output3, params.W23, params.output2);
	d_apply_activation(params.output3, params.activationFunction3);
}

__global__ void d_back_propagate(GPUTrainingParameters const params) {

	PRINTF("d_back_propagate\n");

	d_back_propagate_output(params);
	d_back_propagate_hidden(params);
}

__device__ void d_back_propagate_output(GPUTrainingParameters const& params) {

	Matrix const& targetOutput = params.tmp3;

	// Compute the target output based on the labels
	d_fill_target_output(params, targetOutput);

	// Save the difference into the target output buffer
	Matrix const& difference = targetOutput;
	d_cwise_sub(difference, targetOutput, params.output3);

	// Reuse the output buffer for saving the error, for now. Perhaps this is a problem later on.
	Matrix const& error = params.output3;
	d_apply_activation_derivative(params.output3, params.activationFunction3);
	d_cwise_mul(error, params.output3, difference);
	d_cwise_mul(error, error, params.learningRate);

	// Important to make a local copy.
	// Otherwise every thread would transpose the matrix which
	// would lead to undefined behavior.
	Matrix output2 = d_matrix_transpose(params.output2);
	d_mul_add(params.W23, error, output2);

	d_update_bias(params.bias3, error);
}

__device__ void d_back_propagate_hidden(GPUTrainingParameters const& params) {

	PRINTF("d_back_propagate_hidden\n");

	// The weight updates are computed by
	// W23^T * e3 * ∇σ * input^T

	// Important to make a local copy.
	// Otherwise every thread would transpose the matrix which
	// would lead to undefined behavior.
	Matrix W23 = d_matrix_transpose(params.W23);

	// See d_back_propagation_output
	// Already contains the learningRate.
	Matrix const& error = params.output3;

	// Backpropagate the error.
	d_mul(params.tmp2, W23, error);

	d_update_bias(params.bias2, params.tmp2);

	// And then compute the weight update
	d_apply_activation_derivative(params.output2, params.activationFunction2);
	d_cwise_mul(params.tmp2, params.output2, params.tmp2);

	Matrix images;
	images.rows = params.width * params.height;
	images.cols = params.batchSize;
	images.layout = Matrix::COLUMN_MAJOR;
	images.data = params.images;
	images = d_matrix_transpose(images);

	d_mul_add(params.W12, params.tmp2, images);
}

__device__ void d_apply_activation(Matrix const& A, NeuralNetwork::ActFctType functionType) {

	PRINTF("d_activate_layer\n");

	// Target index for this thread.
	size_t const idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y + blockIdx.y * blockDim.x * blockDim.y * gridDim.x;

	// If the target index would handle an element outside of the data buffer, terminate.
	if (idx >= A.cols * A.rows) {
		return;
	}

	switch (functionType) {
	case NeuralNetwork::SIGMOID:
		A.data[idx] = 1.0f / (1.0f + exp(-A.data[idx]));
		break;
	case NeuralNetwork::TANH:
		A.data[idx] = tanh(A.data[idx]);
		break;
	}
}

__device__ void d_apply_activation_derivative(Matrix const& A, NeuralNetwork::ActFctType functionType) {

	PRINTF("d_apply_activation_derivative\n");

	// Target index for this thread.
	size_t const idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y + blockIdx.y * blockDim.x * blockDim.y * gridDim.x;

	// If the target index would handle an element outside of the data buffer, terminate.
	if (idx >= A.rows * A.cols) {
		return;
	}

	switch (functionType) {
	case NeuralNetwork::SIGMOID:
		A.data[idx] = A.data[idx] * (1.0f - A.data[idx]);
		break;
	case NeuralNetwork::TANH:
		float t = tanh(A.data[idx]);
		A.data[idx] = 1.0f - t * t;
		break;
	}
}

__device__ void d_fill_target_output(GPUTrainingParameters const& params, Matrix const& targetOutput) {

	if (targetOutput.rows != NUM_DIGITS) {
		printf("d_fill_target_output: wrong number of rows. Given %lu, expected %u\n", targetOutput.rows, NUM_DIGITS);
		return;
	}

	size_t const srcIdx = threadIdx.x + blockIdx.x * blockDim.x;
	size_t const targetX = threadIdx.x + blockIdx.x * blockDim.x;
	size_t const targetY = threadIdx.y + blockIdx.y * blockDim.y;

	if (targetX >= targetOutput.cols || targetY >= targetOutput.rows) {
		return;
	}

	float const v = (threadIdx.y == params.labels[srcIdx]) ? 1.0f : 0.0f;
	d_matrix_set(targetOutput, targetY, targetX, v);
}

__device__ void d_set_bias(Matrix const& output, Matrix const& bias) {

	if (bias.rows != output.rows) {
		printf("d_set_bias: Bias and output dimensions mismatch. Expected same height but bias was %lu and output was %lu\n", bias.rows, output.rows);
		return;
	}

	if (bias.cols > 1) {
		printf("d_set_bias: Bias column dimension is %lu > 1. Not handled.\n", bias.cols);
		return;
	}

	size_t const targetX = threadIdx.x + blockIdx.x * blockDim.x;
	size_t const targetY = threadIdx.y + blockIdx.y * blockDim.y;

	if (targetX >= output.cols || targetY >= output.rows) {
		return;
	}

	d_matrix_set(output, targetY, targetX, d_matrix_get(bias, targetY, 1));
}

__device__ void d_assign(float* c, float const a, float const b) {
	*c = b;
}

__device__ void d_add(float* c, float const a, float const b) {
	*c = a + b;
	//printf("d_add(%f, %f, %f\n)", *a, b, c);
}

__device__ void d_sub(float* c, float const a, float const b) {
	*c = a - b;
	//printf("d_add(%f, %f, %f)\n", *c, a, b);
}

__device__ void d_mul(float* c, float const a, float const b) {
	*c = a * b;
}

__device__ void d_mul(Matrix const& C, Matrix const& A, Matrix const& B) {
	PRINTF("d_mul\n");
	d_mul_base(C, A, B, &d_assign);
}

__device__ void d_mul_add(Matrix const& C, Matrix const& A, Matrix const& B) {
	PRINTF("d_mul_add\n");
	d_mul_base(C, A, B, &d_add);
}

/**
 * Computes C = AB where the dimensions of A and be have to be a multiple of MATRIX_SIZE_DIVISOR.
 *
 * @param[in] A first factor of the matrix multiplication.
 * @param[in] B second factor of the multiplication.
 * @param[out] C Matrix holding the result. Must provide enough storage space.
 */
__device__ void d_mul_base(Matrix const& C, Matrix const& A, Matrix const& B, void(*op)(float*, float const, float const)) {

	if (A.cols != B.rows) {

		printf("d_mul_base: Incompatible matrices: (%lu, %lu) x (%lu, %lu)\n", A.rows, A.cols, B.rows, B.cols);
		return;
	}

	// The block caches are row major.
	__shared__ float blockCacheA[MATRIX_SIZE_DIVISOR][MATRIX_SIZE_DIVISOR];
	__shared__ float blockCacheB[MATRIX_SIZE_DIVISOR][MATRIX_SIZE_DIVISOR];

	// Compute the target coordinates.
	size_t const x = blockIdx.x * MATRIX_SIZE_DIVISOR + threadIdx.x;
	size_t const y = blockIdx.y * MATRIX_SIZE_DIVISOR + threadIdx.y;

	// If this thread has nothing to do, because it would access invalid memory, exit
	if (x >= C.cols || y >= C.rows) {
		return;
	}

//	if (A.cols % MATRIX_SIZE_DIVISOR != 0 || B.rows % MATRIX_SIZE_DIVISOR != 0) {
//		printf("d_mul_base: A's cols is not a multiple of %u: (%lu, %lu) x (%lu, %lu)\n", MATRIX_SIZE_DIVISOR, A.rows, A.cols, B.rows, B.cols);
//		return;
//	}

	float threadValue = 0.0f;
	unsigned int const numSubBlocks = A.cols / MATRIX_SIZE_DIVISOR;
	for (int k = 0; k < numSubBlocks; ++k)
	{
		size_t const xA = k * MATRIX_SIZE_DIVISOR + threadIdx.x;
		if (xA < A.cols) {
			blockCacheA[threadIdx.y][threadIdx.x] = d_matrix_get(A, y, xA);
		} else {
			blockCacheA[threadIdx.y][threadIdx.x] = 0.0f;
		}

		size_t const yB = k * MATRIX_SIZE_DIVISOR + threadIdx.y;
		if (yB < B.rows) {
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

	float* const pValue = d_matrix_pget(C, y, x);
	op(pValue, *pValue, threadValue);
}

__device__ void d_cwise_sub(Matrix const& C, Matrix const& A, Matrix const& B) {
	d_cwise_op(C, A, B, &d_sub);
}

__device__ void d_cwise_mul(Matrix const& C, Matrix const& A, Matrix const& B) {
	d_cwise_op(C, A, B, &d_mul);
}

__device__ void d_cwise_op(Matrix const& C, Matrix const& A, Matrix const& B, void(*op)(float*, float const, float const)) {

	if (A.cols != B.cols || A.rows != B.rows || B.cols != C.cols || B.rows != C.rows) {

		printf("d_cwise_op: Incompatible matrices: (%lu, %lu) + (%lu, %lu) = (%lu, %lu)\n", A.rows, A.cols, B.rows, B.cols, C.rows, C.cols);
		return;
	}

	size_t const x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t const y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= A.cols || y >= A.rows) {
		return;
	}

	op(d_matrix_pget(C, y, x), d_matrix_get(A, y, x), d_matrix_get(B, y, x));
}

__device__ void d_cwise_mul(Matrix const& C, Matrix const& A, float const v) {
	d_cwise_op(C, A, v, &d_mul);
}

__device__ void d_cwise_op(Matrix const& C, Matrix const& A, float const v, void(*op)(float*, float const, float const)) {

	if (A.cols != C.cols || A.rows != C.rows) {

		printf("d_cwise_op: Incompatible matrices: v * (%lu, %lu) = (%lu, %lu)\n", A.rows, A.cols, C.rows, C.cols);
		return;
	}

	size_t const x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t const y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= A.cols || y >= A.rows) {
		return;
	}

	op(d_matrix_pget(C, y, x), d_matrix_get(A, y, x), v);
}

__device__ void d_fill_random(Matrix const& A) {

	size_t const targetX = threadIdx.x + blockIdx.x * blockDim.x;
	size_t const targetY = threadIdx.y + blockIdx.y * blockDim.y;

	if (targetX >= A.cols || targetY >= A.rows) {
		return;
	}

	d_matrix_set(A, targetY, targetX, static_cast<float>(targetX));
}

__device__ void d_update_bias(Matrix const& bias, Matrix const& error) {

	if (bias.rows != error.rows|| bias.cols != 1) {

		printf("d_update_bias: Invalid matrices: bias(%lu, %lu) error(%lu, %lu)\n", bias.rows, bias.cols, error.rows, error.cols);
		return;
	}

	// The block caches are row major.
	__shared__ float blockCacheError[MATRIX_SIZE_DIVISOR][MATRIX_SIZE_DIVISOR];

	// Compute the target coordinates.
	size_t const y = blockIdx.y * MATRIX_SIZE_DIVISOR + threadIdx.y;

	// If this thread has nothing to do, because it would access invalid memory, exit
	if (y >= bias.rows) {
		return;
	}

	float threadValue = 0.0f;
	unsigned int const numSubBlocks = error.cols / MATRIX_SIZE_DIVISOR;
	for (int k = 0; k < numSubBlocks; ++k)
	{
		size_t const xA = k * MATRIX_SIZE_DIVISOR + threadIdx.x;
		if (xA < error.cols) {
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

	d_matrix_set(bias, y, 1, threadValue);
}
