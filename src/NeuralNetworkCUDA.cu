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

	/* Temporary buffers, e.g. for back propagation. */
	float* tmp1;
	size_t tmp1_len;
	float* tmp2;
	size_t tmp2_len;
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

__device__ float* d_matrix_pget(Matrix const M, size_t const y, size_t const x) {
	if (M.layout == Matrix::ROW_MAJOR) {
		return M.data + (x + y * M.cols);
	} else {
		return M.data + (x * M.rows + y);
	}
}

__device__ float d_matrix_get(Matrix const M, size_t const y, size_t const x) {
	return *d_matrix_pget(M, y, x);
}

__device__ void d_matrix_set(Matrix const M, size_t const y, size_t const x, float const value) {
	if (M.layout == Matrix::ROW_MAJOR) {
		M.data[x + y * M.cols] = value;
	} else {
		M.data[x * M.rows + y] = value;
	}
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

	// Temporary storage of the size of the output vectors
	trainingParams.tmp1_len = outputLayer->nodes.size() * trainingParams.batchSize;
	err = cudaMalloc((void**) &trainingParams.tmp1, trainingParams.tmp1_len * sizeof(float));
	assert(err == cudaSuccess);

	// Temporary storage of the size of the output vectors
	trainingParams.tmp2_len = outputLayer->nodes.size() * trainingParams.batchSize;
	err = cudaMalloc((void**) &trainingParams.tmp2, trainingParams.tmp2_len * sizeof(float));
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
	cudaFree (trainingParams.tmp1);
	trainingParams.tmp1 = nullptr;
	cudaFree (trainingParams.tmp2);
	trainingParams.tmp2 = nullptr;

	//
	// Copy the weight data into the c++ data structure.
	//
	Layer* hidden = getLayer(HIDDEN);
	trainingParams.activationFunction2 = hidden->actFctType;
	size_t k = 0;
	for (size_t j = 0; j < hidden->nodes.size(); ++j) {
		Layer::Node* node = hidden->nodes[j];
		node->bias = bias2[j];
		for (size_t i = 0; i < node->weights.size(); ++i) {
			node->weights[i] = W12[k];
			++k;
		}
	}

	Layer* output = getLayer(OUTPUT);
	trainingParams.activationFunction3 = output->actFctType;
	k = 0;
	for (size_t j = 0; j < output->nodes.size(); ++j) {
		Layer::Node* node = output->nodes[j];
		node->bias = bias3[j];
		for (size_t i = 0; i < node->weights.size(); ++i) {
			node->weights[i] = W23[k];
			++k;
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

/* Matrix manipulation operations. */
__device__ void d_mul_base(Matrix C, Matrix A, Matrix B, void(*op)(float*, float, float));
__device__ void d_mul(Matrix C, Matrix A, Matrix B);
__device__ void d_mul_add(Matrix C, Matrix A, Matrix B);
__device__ void d_cwise_op(Matrix C, Matrix A, Matrix B, void(*op)(float*, float, float));
__device__ void d_cwise_mul(Matrix C, Matrix A, Matrix B);
__device__ void d_cwise_sub(Matrix C, Matrix A, Matrix B);

/* Neural network operations. */
__device__ void d_apply_activation(Matrix, NeuralNetwork::ActFctType);
__device__ void d_apply_activation_derivative(Matrix, NeuralNetwork::ActFctType);
__device__ void d_back_propagate_output(GPUTrainingParameters const);
__device__ void d_back_propagate_hidden(GPUTrainingParameters const);
__device__ void d_fill_target_output(GPUTrainingParameters const, Matrix);
__device__ void d_set_bias(Matrix output, Matrix const bias);
__device__ void d_fill_random(Matrix);

__global__ void d_feed_forward(GPUTrainingParameters const params) {

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		printf("d_feed_forward\n");
	}

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
	imgs.cols = params.batchSize;
	imgs.layout = Matrix::COLUMN_MAJOR;
	imgs.data = params.images; // Global data pointer, column major, yields one image in each column vector.

	Matrix hiddenOutput;
	hiddenOutput.rows = params.numHiddenNodes;
	hiddenOutput.cols = params.batchSize;
	hiddenOutput.layout = Matrix::ROW_MAJOR;
	hiddenOutput.data = params.output2;
	if (hiddenOutput.rows * hiddenOutput.cols != params.output2_len) {
		printf("ERROR: HiddenOutput matrix has wrong dimensions: %lu x %lu != %lu\n", hiddenOutput.rows, hiddenOutput.cols, params.output2_len);
	}

	Matrix bias2;
	bias2.rows = params.numHiddenNodes;
	bias2.cols = 1;
	bias2.data = params.bias2;
	if (bias2.rows * bias2.cols != params.bias2_len) {
		printf("ERROR: Bias2 has wrong dimensions: %lu x %lu != %lu\n", bias2.rows, bias2.cols, params.bias2_len);
	}

	d_set_bias(hiddenOutput, bias2);
	d_mul_add(hiddenOutput, W12, imgs);
	d_apply_activation(hiddenOutput, params.activationFunction2);

	Matrix W23;
	W23.rows = NUM_DIGITS;
	W23.cols = params.numHiddenNodes;
	W23.layout = Matrix::ROW_MAJOR;
	W23.data = params.W23;
	if (W23.rows * W23.cols != params.W23_len) {
		printf("ERROR: W23 matrix has wrong dimensions: %lu x %lu != %lu\n", W23.rows, W23.cols, params.W23_len);
	}

	Matrix output;
	output.rows = NUM_DIGITS;
	output.cols = params.batchSize;
	output.layout = Matrix::ROW_MAJOR;
	output.data = params.output3;
	if (output.rows * output.cols != params.output3_len) {
		printf("ERROR: Output matrix has wrong dimensions: %lu x %lu != %lu\n", output.rows, output.cols, params.output3_len);
	}

	Matrix bias3;
	bias3.rows = NUM_DIGITS;
	bias3.cols = 1;
	bias3.data = params.bias3;
	if (bias3.rows * bias3.cols != params.bias3_len) {
		printf("ERROR: Bias3 has wrong dimensions: %lu x %lu != %lu\n", bias3.rows, bias3.cols, params.bias3_len);
	}

	d_set_bias(output, bias3);
	d_mul_add(output, W23, hiddenOutput);
	d_apply_activation(output, params.activationFunction3);

//	d_fill_random(W12);
//	d_fill_random(W23);
//	d_fill_random(bias2);
//	d_fill_random(bias3);
}

__global__ void d_back_propagate(GPUTrainingParameters const params) {

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		printf("d_back_propagate\n");
	}

	d_back_propagate_output(params);
	d_back_propagate_hidden(params);
}

__device__ void d_back_propagate_output(GPUTrainingParameters const params) {

	Matrix targetOutput;
	targetOutput.rows = NUM_DIGITS;
	targetOutput.cols = params.batchSize;
	targetOutput.data = params.tmp1;
	if (targetOutput.rows * targetOutput.cols != params.tmp1_len) {
		printf("d_back_propagate_output: targetOutput matrix has wrong dimensions: %lu x %lu != %lu\n", targetOutput.rows, targetOutput.cols, params.tmp1_len);
	}

	// Compute the target output based on the labels
	d_fill_target_output(params, targetOutput);

	Matrix output;
	output.rows = NUM_DIGITS;
	output.cols = params.batchSize;
	output.data = params.output3;
	if (output.rows * output.cols != params.output3_len) {
		printf("d_back_propagate_output: Output matrix has wrong dimensions: %lu x %lu != %lu\n", output.rows, output.cols, params.output3_len);
	}

	// Save the difference into the target output buffer
	Matrix difference = targetOutput;
	// Reuse the output buffer for saving the error, for now. Perhaps this is a problem later on.
	Matrix error = output;

	d_cwise_sub(difference, targetOutput, output);
	d_apply_activation_derivative(output, params.activationFunction3);
	d_cwise_mul(error, output, difference);

	Matrix hiddenOutput;
	hiddenOutput.rows = params.batchSize;
	hiddenOutput.cols = params.numHiddenNodes;
	hiddenOutput.layout = Matrix::ROW_MAJOR;
	hiddenOutput.data = params.output2;
	if (hiddenOutput.rows * hiddenOutput.cols != params.output2_len) {
		printf("d_back_propagate_output: hidden output matrix has wrong dimensions: %lu x %lu != %lu\n", hiddenOutput.rows, hiddenOutput.cols, params.output2_len);
	}

	Matrix W23;
	W23.rows = NUM_DIGITS;
	W23.cols = params.numHiddenNodes;
	W23.data = params.W23;
	if (W23.rows * W23.cols != params.W23_len) {
		printf("d_back_propagate_output: W23 matrix has wrong dimensions: %lu x %lu != %lu\n", W23.rows, W23.cols, params.W23_len);
	}

	d_mul_add(W23, error, hiddenOutput);
}

__device__ void d_back_propagate_hidden(GPUTrainingParameters const) {
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		printf("d_back_propagate_hidden\n");
	}
}

__device__ void d_apply_activation(Matrix A, NeuralNetwork::ActFctType functionType) {

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		printf("d_activate_layer\n");
	}

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

__device__ void d_apply_activation_derivative(Matrix A, NeuralNetwork::ActFctType functionType) {
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		printf("d_apply_activation_derivative\n");
	}

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
	//printf("actFctDeriv(%lu) = %f\n", idx, data[idx]);
}

__device__ void d_fill_target_output(GPUTrainingParameters const params, Matrix targetOutput) {

	if (targetOutput.rows != NUM_DIGITS) {
		printf("d_fill_target_output: wrong number of rows. Given %lu, expected %u\n", targetOutput.rows, NUM_DIGITS);
		return;
	}

	size_t srcIdx = threadIdx.x + blockIdx.x * blockDim.x;
	size_t targetX = threadIdx.x + blockIdx.x * blockDim.x;
	size_t targetY = threadIdx.y + blockIdx.y * blockDim.y;

	if (targetX >= targetOutput.cols || targetY >= targetOutput.rows) {
		return;
	}

//	size_t targetIdx = 0;
//	if (targetOutput.layout == Matrix::ROW_MAJOR) {
//		targetIdx = targetX + targetY * targetOutput.cols;
//	} else if (targetOutput.layout == Matrix::COLUMN_MAJOR) {
//		targetIdx = targetX * targetOutput.rows + targetY;
//	}

	//targetOutput.data[targetIdx] = (threadIdx.y == params.labels[srcIdx]) ? 1.0f : 0.0f;
	float const v = (threadIdx.y == params.labels[srcIdx]) ? 1.0f : 0.0f;
	d_matrix_set(targetOutput, targetY, targetX, v);
//	if (threadIdx.x == 0) {
//		printf("d_fill_target_output: (%lu, %lu) = %f\n", targetX, targetY, targetOutput.data[targetIdx]);
//	}
}

__device__ void d_set_bias(Matrix output, Matrix const bias) {

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

	//d_matrix_set(output, targetY, targetX, d_matrix_get(bias, targetY, 1));
	d_matrix_set(output, targetY, targetX, static_cast<float>(targetY));
}

__device__ void d_assign(float* c, float a, float b) {
	*c = b;
}

__device__ void d_add(float* c, float a, float b) {
	*c = a + b;
	//printf("d_add(%f, %f, %f\n)", *a, b, c);
}

__device__ void d_sub(float* c, float a, float b) {
	*c = a - b;
	//printf("d_add(%f, %f, %f)\n", *c, a, b);
}

__device__ void d_mul(float* c, float a, float b) {
	*c = a * b;
}

__device__ void d_mul(Matrix C, Matrix A, Matrix B) {
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		printf("d_mul\n");
	}
	d_mul_base(C, A, B, &d_assign);
}

__device__ void d_mul_add(Matrix C, Matrix A, Matrix B) {
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		printf("d_mul_add\n");
	}
	d_mul_base(C, A, B, &d_add);
}

/**
 * Computes C = AB where the dimensions of A and be have to be a multiple of MATRIX_SIZE_DIVISOR.
 *
 * @param[in] A first factor of the matrix multiplication.
 * @param[in] B second factor of the multiplication.
 * @param[out] C Matrix holding the result. Must provide enough storage space.
 */
__device__ void d_mul_base(Matrix C, Matrix A, Matrix B, void(*op)(float*, float, float)) {

	if (A.cols != B.rows) {

		printf("Incompatible matrices: (%lu, %lu) x (%lu, %lu)\n", A.rows, A.cols, B.rows, B.cols);
		return;
	}

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

	size_t idx = 0;
	if (C.layout == Matrix::ROW_MAJOR) {
		idx = (blockIdx.y * C.cols + blockIdx.x) * MATRIX_SIZE_DIVISOR + threadIdx.y * C.cols + threadIdx.x;
	} else if (C.layout == Matrix::COLUMN_MAJOR) {
		idx = (blockIdx.y + blockIdx.x * C.cols) * MATRIX_SIZE_DIVISOR + threadIdx.y + threadIdx.x * C.cols;
	}
	float* pValue = &(C.data[idx]);
	op(pValue, *pValue, threadValue);
}

__device__ void d_cwise_sub(Matrix C, Matrix A, Matrix B) {
	d_cwise_op(C, A, B, &d_sub);
}

__device__ void d_cwise_mul(Matrix C, Matrix A, Matrix B) {
	d_cwise_op(C, A, B, &d_mul);
}

__device__ void d_cwise_op(Matrix C, Matrix A, Matrix B, void(*op)(float*, float, float)) {

	if (A.cols != B.cols || A.rows != B.rows || B.cols != C.cols || B.rows != C.rows) {

		printf("Incompatible matrices: (%lu, %lu) + (%lu, %lu) = (%lu, lu)\n", A.rows, A.cols, B.rows, B.cols, C.rows, C.cols);
		return;
	}

	size_t const x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t const y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= A.cols || y >= A.rows) {
		return;
	}

	//C.data[idxC] = A.data[idxA] - B.data[idxB];
	op(d_matrix_pget(C, y, x), d_matrix_get(A, y, x), d_matrix_get(B, y, x));
}

__device__ void d_fill_random(Matrix A) {

	size_t const targetX = threadIdx.x + blockIdx.x * blockDim.x;
	size_t const targetY = threadIdx.y + blockIdx.y * blockDim.y;

	if (targetX >= A.cols || targetY >= A.rows) {
		return;
	}

	d_matrix_set(A, targetY, targetX, static_cast<float>(targetX));
}
