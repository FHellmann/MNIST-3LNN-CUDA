#include "NeuralNetworkCUDA.h"

#include <iostream>
#include <cmath>

using namespace std;

__host__ NeuralNetworkCUDA::NeuralNetworkCUDA(const int inpCount,
		const int hidCount, const int outCount, const double learningRate) :
		NeuralNetwork(inpCount, hidCount, outCount, learningRate) {
}

__host__ NeuralNetworkCUDA::~NeuralNetworkCUDA() {
}

#define MATRIX_SIZE_DIVISOR 28
#define NUM_DIGITS 10

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
	float* W12;
	size_t W12_len;
	float* W23;
	size_t W23_len;

	/* Biases */
	float* bias2;
	size_t bias2_len;
	float* bias3;
	size_t bias3_len;

	/* Layer data */
	float* output2;
	size_t output2_len;
	float* output3;
	size_t output3_len;

	NeuralNetwork::ActFctType activationFunction2;
	NeuralNetwork::ActFctType activationFunction3;

	/* Training parameters. */
	float errorThreshold;
	float maxDerivation;
};

struct GPUSharedMemoryLayout {
	size_t W1_pos = 0;
	size_t W1_size = 0;
	size_t W2_pos = 0;
	size_t W2_size = 0;
	size_t inputBias_pos = 0;
	size_t inputBias_size = 0;
	size_t hiddenOutput_pos = 0;
	size_t hiddenOutput_size = 0;
	size_t hiddenBias_pos = 0;
	size_t hiddenBias_size = 0;
	size_t outputOutput_pos = 0;
	size_t outputOutput_size = 0;
	size_t outputBias_pos = 0;
	size_t outputBias_size = 0;
	size_t image_pos = 0;
	size_t image_size = 0;
} gpuSharedMemoryLayout;

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

	// Collect memory in RAM
	size_t const singleImgPixCount = images.front().total();
	size_t const allImgBufElements = singleImgPixCount * images.size();
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

	cudaError_t err;

	Layer* inputLayer  = getLayer(INPUT);
	Layer* hiddenLayer = getLayer(HIDDEN);
	Layer* outputLayer = getLayer(OUTPUT);

	GPUTrainingParameters trainingParams;
	trainingParams.numExamples = images.size();
	trainingParams.width = images.front().cols;
	trainingParams.height = images.front().rows;
	trainingParams.numHiddenNodes = hiddenLayer->nodes.size();
	trainingParams.errorThreshold = training_error_threshold;
	trainingParams.maxDerivation = max_derivation;
	trainingParams.batchSize = MATRIX_SIZE_DIVISOR;

	//
	// Allocate cuda memory
	//

	// Images
	err = cudaMalloc((void**) &trainingParams.images, allImgBufElements * sizeof(float));
	assert(err == cudaSuccess);

	// Labels
	err = cudaMalloc((void**) &trainingParams.labels, labels.size() * sizeof(float));
	assert(err == cudaSuccess);

	// Storage for the first weight matrix
	trainingParams.W12_len = inputLayer->nodes.size() * hiddenLayer->nodes.size();
	err = cudaMalloc((void**) &trainingParams.W12, trainingParams.W12_len * sizeof(float));
	assert(err == cudaSuccess);

	// Storage for the hidden layer bias vector
	trainingParams.bias2_len = hiddenLayer->nodes.size();
	err = cudaMalloc((void**) &trainingParams.bias2, trainingParams.bias2_len * sizeof(float));
	assert(err == cudaSuccess);

	// Storage for the second weight matrix
	trainingParams.W23_len = hiddenLayer->nodes.size() * outputLayer->nodes.size();
	err = cudaMalloc((void**) &trainingParams.W23, trainingParams.W23_len * sizeof(float));
	assert(err == cudaSuccess);

	// Storage for the output layer bias vector
	trainingParams.bias3_len = outputLayer->nodes.size();
	err = cudaMalloc((void**) &trainingParams.bias3, trainingParams.bias3_len * sizeof(float));
	assert(err == cudaSuccess);

	// Storage for the output layer output vectors
	trainingParams.output2_len = hiddenLayer->nodes.size() * trainingParams.batchSize;
	err = cudaMalloc((void**) &trainingParams.output2, trainingParams.output2_len * sizeof(float));
	assert(err == cudaSuccess);

	// Storage for the output layer output vectors
	trainingParams.output3_len = outputLayer->nodes.size() * trainingParams.batchSize;
	err = cudaMalloc((void**) &trainingParams.output3, trainingParams.output3_len * sizeof(float));
	assert(err == cudaSuccess);

	//
	// Copy data to graphics card
	//
	err = cudaMemcpy(trainingParams.images, imgData, allImgBufElements * sizeof(float), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);
	err = cudaMemcpy(trainingParams.labels, flabels, labels.size() * sizeof(float), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);

	delete[] imgData;
	imgData = nullptr;
	delete[] flabels;
	flabels = nullptr;

//	size_t sharedMemorySize = 0;

	// Size of the first weight matrix
//	gpuSharedMemoryLayout.W1_pos = 0;
//	gpuSharedMemoryLayout.W1_size = trainingParams.W1_len * sizeof(float);
//	sharedMemorySize += gpuSharedMemoryLayout.W1_size;
//
//	// Size of the second weight matrix
//	gpuSharedMemoryLayout.W2_pos = gpuSharedMemoryLayout.W1_pos + gpuSharedMemoryLayout.W1_size;
//	gpuSharedMemoryLayout.W2_size = trainingParams.W2_len * sizeof(float);
//	sharedMemorySize += gpuSharedMemoryLayout.W2_size;
//
//	// Size of the hidden layer output nodes
//	gpuSharedMemoryLayout.hiddenOutput_pos = gpuSharedMemoryLayout.W2_pos + gpuSharedMemoryLayout.W2_size;
//	gpuSharedMemoryLayout.hiddenOutput_size = hiddenLayer->nodes.size() * sizeof(float);
//	sharedMemorySize += gpuSharedMemoryLayout.hiddenOutput_size;
//
//	// Size of the output layer output values
//	gpuSharedMemoryLayout.outputOutput_pos = gpuSharedMemoryLayout.hiddenOutput_pos + gpuSharedMemoryLayout.hiddenOutput_size;
//	gpuSharedMemoryLayout.outputOutput_size = outputLayer->nodes.size() * sizeof(float);
//	sharedMemorySize += gpuSharedMemoryLayout.outputOutput_size;
//
//	// Size of the hidden bias vector
//	gpuSharedMemoryLayout.hiddenBias_pos   = gpuSharedMemoryLayout.outputOutput_pos + gpuSharedMemoryLayout.outputOutput_size;
//	gpuSharedMemoryLayout.hiddenBias_size  = hiddenLayer->nodes.size() * sizeof(float);
//	sharedMemorySize += gpuSharedMemoryLayout.hiddenBias_size;
//
//	// Size of the input bias vector
//	gpuSharedMemoryLayout.inputBias_pos    = gpuSharedMemoryLayout.hiddenOutput_pos + gpuSharedMemoryLayout.hiddenOutput_size;
//	gpuSharedMemoryLayout.inputBias_size   = inputLayer->nodes.size() * sizeof(float);
//	sharedMemorySize += gpuSharedMemoryLayout.inputBias_size;
//
//	// Size of the input vector
//	gpuSharedMemoryLayout.image_pos        = gpuSharedMemoryLayout.inputBias_pos + gpuSharedMemoryLayout.inputBias_size;
//	gpuSharedMemoryLayout.image_size       = inputLayer->nodes.size() * sizeof(uint8_t);
//	sharedMemorySize += gpuSharedMemoryLayout.image_size;

	cudaMemset(trainingParams.W12, 0.0, trainingParams.W12_len * sizeof(float));
	cudaMemset(trainingParams.W23, 0.0, trainingParams.W23_len * sizeof(float));
	cudaMemset(trainingParams.bias2, 0.0, trainingParams.bias2_len * sizeof(float));
	cudaMemset(trainingParams.bias3, 0.0, trainingParams.bias3_len * sizeof(float));

	// Configure Grid, i.e. setup Blocks and Threads
	dim3 numBlocks(2,2);
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
	float* W12 = new float[trainingParams.W12_len];
	float* W23 = new float[trainingParams.W23_len];
	float* bias2 = new float[trainingParams.bias2_len];
	float* bias3 = new float[trainingParams.bias3_len];

	// Copy it back to neural network data structure
	err = cudaMemcpy(W12, trainingParams.W12, trainingParams.W12_len * sizeof(float), cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);
	err = cudaMemcpy(W23, trainingParams.W23, trainingParams.W23_len * sizeof(float), cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);
	err = cudaMemcpy(bias2, trainingParams.bias2, trainingParams.bias2_len * sizeof(float), cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);
	err = cudaMemcpy(bias3, trainingParams.bias3, trainingParams.bias3_len * sizeof(float), cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);

	// Free the cuda buffers
	cudaFree (trainingParams.images);
	trainingParams.images = nullptr;
	cudaFree (trainingParams.labels);
	trainingParams.labels = nullptr;
	cudaFree (trainingParams.W12);
	trainingParams.W12 = nullptr;
	cudaFree (trainingParams.W23);
	trainingParams.W23 = nullptr;
	cudaFree (trainingParams.bias2);
	trainingParams.bias2 = nullptr;
	cudaFree (trainingParams.bias3);
	trainingParams.bias3 = nullptr;
	cudaFree (trainingParams.output2);
	trainingParams.output2 = nullptr;
	cudaFree (trainingParams.output3);
	trainingParams.output3 = nullptr;

	//
	// Copy the weight data into the c++ data structure.
	//
	Layer* hidden = getLayer(HIDDEN);
	trainingParams.activationFunction2 = hidden->actFctType;
	for (size_t j = 0; j < hidden->nodes.size(); ++j) {
		Layer::Node* node = hidden->nodes[j];
		node->bias = bias2[j];
		for (size_t i = 0; i < node->weights.size(); ++i) {
			node->weights[i] = W12[j * hidden->nodes.size() + i];
		}
	}

	Layer* output = getLayer(OUTPUT);
	trainingParams.activationFunction3 = output->actFctType;
	for (size_t j = 0; j < output->nodes.size(); ++j) {
		Layer::Node* node = output->nodes[j];
		node->bias = bias3[j];
		for (size_t i = 0; i < node->weights.size(); ++i) {
			node->weights[i] = W23[j * output->nodes.size() + i];
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

__device__ void d_print(GPUTrainingParameters const params) {
	printf("TrainingParams:\n"
			"  W12: %p\n"
		    "  W1_len: %lu\n"
			"  W2: %p\n"
			"  W2_len: %lu\n"
			"  errorThreshold: %f\n"
			"  width: %lu\n"
			"  height: %lu\n"
			"  numExamples: %lu\n"
			"  numHiddenNodes: %lu\n",
			params.W12,
			params.W12_len,
			params.W23,
			params.W23_len,
			params.errorThreshold,
			params.width,
			params.height,
			params.numExamples,
			params.numHiddenNodes);
}

__device__ void d_mul_shared(Matrix A, Matrix B, Matrix C);
__device__ void d_activate_layer(float* const, size_t const, NeuralNetwork::ActFctType);

__global__ void d_feed_forward(GPUTrainingParameters const params) {

	size_t const numImages = params.numHiddenNodes;

	Matrix W12;
	W12.rows = params.numHiddenNodes;
	W12.cols = params.width * params.height;
	W12.layout = Matrix::ROW_MAJOR;
	W12.data = params.W12; // Global data pointer
	if (W12.rows * W12.cols != params.W12_len) {
		printf("ERROR: W12 matrix has wrong dimensions: %lu x %lu != %lu\n", W12.rows, W12.cols, params.W12_len);
	}

	Matrix imgs;
	imgs.rows = params.width * params.height;
	imgs.cols = numImages;
	imgs.layout = Matrix::COLUMN_MAJOR;
	imgs.data = params.images; // Global data pointer, column major, yields one image in each column vector.

	Matrix hiddenOutput;
	hiddenOutput.rows = params.numHiddenNodes;
	hiddenOutput.cols = numImages;
	hiddenOutput.layout = Matrix::ROW_MAJOR;
	hiddenOutput.data = params.output2;
	if (hiddenOutput.rows * hiddenOutput.cols != params.output2_len) {
		printf("ERROR: HiddenOutput matrix has wrong dimensions: %lu x %lu != %lu\n", hiddenOutput.rows, hiddenOutput.cols, params.output2_len);
	}

	d_mul_shared(W12, imgs, hiddenOutput);
	d_activate_layer(params.output2, params.output2_len, params.activationFunction2);

	Matrix W23;
	W23.rows = NUM_DIGITS;
	W23.cols = params.numHiddenNodes;
	W23.layout = Matrix::ROW_MAJOR;
	W23.data = params.W23;
	if (W23.rows * W23.cols != params.W23_len) {
		printf("ERROR: W23 matrix has wrong dimensions: %lu x %lu != %lu\n", W23.rows, W23.cols, params.W23_len);
	}

	Matrix output;
	output.rows = W23.rows;
	output.cols = hiddenOutput.cols;
	output.layout = Matrix::ROW_MAJOR;
	output.data = params.output3;
	if (output.rows * output.cols != params.output3_len) {
		printf("ERROR: Output matrix has wrong dimensions: %lu x %lu != %lu\n", output.rows, output.cols, params.output3_len);
	}

	d_mul_shared(W23, hiddenOutput, output);
	d_activate_layer(params.output3, params.output3_len, params.activationFunction3);
}

__global__ void d_back_propagate(GPUTrainingParameters const params) {

}

__device__ void d_activate_layer(float* const data, size_t const len, NeuralNetwork::ActFctType functionType) {

	// Target index for this thread.
	size_t const idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y + blockIdx.y * blockDim.x * blockDim.y * gridDim.x;

	// If the target index would handle an element outside of the data buffer, terminate.
	if (idx >= len) {
		return;
	}

	switch (functionType) {
	case NeuralNetwork::SIGMOID:
		data[idx] = 1.0f / (1.0f + exp(-data[idx]));
		break;
	case NeuralNetwork::TANH:
		data[idx] = tanh(data[idx]);
		break;
	}
}

/**
 * Computes C = AB where the dimensions of A and be have to be a multiple of MATRIX_SIZE_DIVISOR.
 *
 * @param[in] A first factor of the matrix multiplication.
 * @param[in] B second factor of the multiplication.
 * @param[out] C Matrix holding the result. Must provide enough storage space.
 */
__device__ void d_mul_shared(Matrix A, Matrix B, Matrix C) {

	if (A.cols != B.rows) {

		printf("Incompatible matrices: (%lu, %lu) x (%lu, %lu)\n", A.rows, A.cols, B.rows, B.cols);
		return;
	}

	// Not needed anymore.
//	if (A.cols % MATRIX_SIZE_DIVISOR != 0 ||
//	    A.rows % MATRIX_SIZE_DIVISOR != 0 ||
//	    B.cols % MATRIX_SIZE_DIVISOR != 0 ||
//	    B.rows % MATRIX_SIZE_DIVISOR != 0) {
//
//		printf("Matrix dimensions not a multiple of %hu: (%lu, %lu) x (%lu, %lu)\n", MATRIX_SIZE_DIVISOR, A.rows, A.cols, B.rows, B.cols);
//		return;
//	}

	// The block caches are row major.
	__shared__ float blockCacheA[MATRIX_SIZE_DIVISOR][MATRIX_SIZE_DIVISOR];
	__shared__ float blockCacheB[MATRIX_SIZE_DIVISOR][MATRIX_SIZE_DIVISOR];

	// If this thread has nothing to do, because it would access invalid memory, exit
	if (blockIdx.x * MATRIX_SIZE_DIVISOR + threadIdx.x > C.cols ||
		blockIdx.y * MATRIX_SIZE_DIVISOR + threadIdx.y > C.rows) {
		return;
	}

	float threadValue = 0.0f;
	unsigned int const numSubBlocks = A.cols / MATRIX_SIZE_DIVISOR;
	for (int k = 0; k < numSubBlocks; ++k)
	{
		if (A.layout == Matrix::COLUMN_MAJOR) {
			blockCacheA[threadIdx.y][threadIdx.x] = A.data[(blockIdx.y + k * A.cols) * MATRIX_SIZE_DIVISOR + threadIdx.y + threadIdx.x * A.cols];
		} else if (A.layout == Matrix::ROW_MAJOR) {
			blockCacheA[threadIdx.y][threadIdx.x] = A.data[(blockIdx.y * A.cols + k) * MATRIX_SIZE_DIVISOR + threadIdx.y * A.cols + threadIdx.x];
		}

		if (B.layout == Matrix::COLUMN_MAJOR) {
			blockCacheB[threadIdx.y][threadIdx.x] = B.data[(blockIdx.x * B.cols + k) * MATRIX_SIZE_DIVISOR + threadIdx.y + threadIdx.x * B.cols];
		} else if (B.layout == Matrix::ROW_MAJOR) {
			blockCacheB[threadIdx.y][threadIdx.x] = B.data[(blockIdx.x + k * B.cols) * MATRIX_SIZE_DIVISOR + threadIdx.y * B.cols + threadIdx.x];
		}

		__syncthreads();

		#pragma unroll
		for (int i = 0; i < MATRIX_SIZE_DIVISOR; ++i)
		{
			threadValue += blockCacheA[threadIdx.y][i] * blockCacheB[i][threadIdx.x];
		}

		__syncthreads();
	}

	if (C.layout == Matrix::COLUMN_MAJOR) {
		C.data[(blockIdx.y + blockIdx.x * C.cols) * MATRIX_SIZE_DIVISOR + threadIdx.y + threadIdx.x * C.cols] = threadValue;
	} else if (C.layout == Matrix::ROW_MAJOR) {
		C.data[(blockIdx.y * C.cols + blockIdx.x) * MATRIX_SIZE_DIVISOR + threadIdx.y * C.cols + threadIdx.x] = threadValue;
	}
}
