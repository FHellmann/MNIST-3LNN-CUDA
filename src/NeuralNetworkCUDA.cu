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

	/* Temporary buffers, e.g. for back propagation. */
	Matrix tmp1;
	Matrix tmp2;
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

__device__ size_t d_matrix_size(Matrix const A) {
	return A.rows * A.cols;
}

size_t matrix_size(Matrix const A) {
	return A.rows * A.cols;
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
	err = cudaMalloc((void**) &trainingParams.W12.data, matrix_size(trainingParams.W12) * sizeof(float));
	assert(err == cudaSuccess);

	// Storage for the hidden layer bias vector
	trainingParams.bias2.rows = hiddenLayer->nodes.size();
	trainingParams.bias2.cols = 1;
	err = cudaMalloc((void**) &trainingParams.bias2.data, matrix_size(trainingParams.bias2) * sizeof(float));
	assert(err == cudaSuccess);

	// Storage for the second weight matrix
	trainingParams.W23.rows = outputLayer->nodes.size();
	trainingParams.W23.cols = hiddenLayer->nodes.size();
	err = cudaMalloc((void**) &trainingParams.W23.data, matrix_size(trainingParams.W23) * sizeof(float));
	assert(err == cudaSuccess);

	// Storage for the output layer bias vector
	trainingParams.bias3.rows = outputLayer->nodes.size();
	trainingParams.bias3.cols = 1;
	err = cudaMalloc((void**) &trainingParams.bias3.data, matrix_size(trainingParams.bias3) * sizeof(float));
	assert(err == cudaSuccess);

	// Storage for the output layer output vectors
	trainingParams.output2.rows = trainingParams.numHiddenNodes;
	trainingParams.output2.cols = trainingParams.batchSize;
	err = cudaMalloc((void**) &trainingParams.output2.data, matrix_size(trainingParams.output2) * sizeof(float));
	assert(err == cudaSuccess);

	// Storage for the output layer output vectors
	trainingParams.output3.rows = outputLayer->nodes.size();
	trainingParams.output3.cols = trainingParams.batchSize;
	err = cudaMalloc((void**) &trainingParams.output3.data, matrix_size(trainingParams.output3) * sizeof(float));
	assert(err == cudaSuccess);

	// Temporary storage of the size of the output vectors
	trainingParams.tmp1.rows = outputLayer->nodes.size();
	trainingParams.tmp1.cols = trainingParams.batchSize;
	err = cudaMalloc((void**) &trainingParams.tmp1.data, matrix_size(trainingParams.tmp1) * sizeof(float));
	assert(err == cudaSuccess);

	// Temporary storage of the size of the output vectors
	trainingParams.tmp2.rows = outputLayer->nodes.size();
	trainingParams.tmp2.cols = trainingParams.batchSize;
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
			for (size_t i = 0; i < node->weights.size(); ++i) {
				W12[k] = node->weights[i];
				++k;
			}
		}
	}

	trainingParams.activationFunction3 = outputLayer->actFctType;
	{
		size_t k = 0;
		for (size_t j = 0; j < outputLayer->nodes.size(); ++j) {
			Layer::Node* node = outputLayer->nodes[j];
			bias3[j] = node->bias;
			for (size_t i = 0; i < node->weights.size(); ++i) {
				W23[k] = node->weights[i];
				++k;
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
	cudaFree (trainingParams.tmp1.data);
	trainingParams.tmp1.data = nullptr;
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
			params.W12.data,
			d_matrix_size(params.W12),
			params.W23.data,
			d_matrix_size(params.W23),
			params.errorThreshold,
			params.width,
			params.height,
			params.numExamples,
			params.numHiddenNodes);
}

/* Matrix manipulation operations. */
__device__ void d_mul_base(Matrix C, Matrix const A, Matrix const B, void(*op)(float*, float const, float const));
__device__ void d_mul(Matrix C, Matrix const A, Matrix const B);
__device__ void d_mul_add(Matrix C, Matrix const A, Matrix const B);
__device__ void d_cwise_op(Matrix C, Matrix const A, Matrix const B, void(*op)(float*, float const, float const));
__device__ void d_cwise_mul(Matrix C, Matrix const A, Matrix const B);
__device__ void d_cwise_sub(Matrix C, Matrix const A, Matrix const B);

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

//	Matrix W12;
//	W12.rows = params.numHiddenNodes;
//	W12.cols = params.width * params.height;
//	W12.layout = Matrix::ROW_MAJOR;
//	W12.data = params.W12; // Global data pointer
//	if (W12.rows * W12.cols != params.W12_len) {
//		printf("ERROR: W12 matrix has wrong dimensions: %lu x %lu != %lu\n", W12.rows, W12.cols, params.W12_len);
//	}

	Matrix imgs;
	imgs.rows = params.width * params.height;
	imgs.cols = params.batchSize;
	imgs.layout = Matrix::COLUMN_MAJOR;
	imgs.data = params.images; // Global data pointer, column major, yields one image in each column vector.

//	Matrix hiddenOutput;
//	hiddenOutput.rows = params.numHiddenNodes;
//	hiddenOutput.cols = params.batchSize;
//	hiddenOutput.layout = Matrix::ROW_MAJOR;
//	hiddenOutput.data = params.output2;
//	if (hiddenOutput.rows * hiddenOutput.cols != params.output2_len) {
//		printf("ERROR: HiddenOutput matrix has wrong dimensions: %lu x %lu != %lu\n", hiddenOutput.rows, hiddenOutput.cols, params.output2_len);
//	}

//	Matrix bias2;
//	bias2.rows = params.numHiddenNodes;
//	bias2.cols = 1;
//	bias2.data = params.bias2;
//	if (bias2.rows * bias2.cols != params.bias2_len) {
//		printf("ERROR: Bias2 has wrong dimensions: %lu x %lu != %lu\n", bias2.rows, bias2.cols, params.bias2_len);
//	}

	d_set_bias(params.output2, params.bias2);
	d_mul_add(params.output2, params.W12, imgs);
	d_apply_activation(params.output2, params.activationFunction2);

//	Matrix W23;
//	W23.rows = NUM_DIGITS;
//	W23.cols = params.numHiddenNodes;
//	W23.layout = Matrix::ROW_MAJOR;
//	W23.data = params.W23;
//	if (W23.rows * W23.cols != params.W23_len) {
//		printf("ERROR: W23 matrix has wrong dimensions: %lu x %lu != %lu\n", W23.rows, W23.cols, params.W23_len);
//	}

//	Matrix output;
//	output.rows = NUM_DIGITS;
//	output.cols = params.batchSize;
//	output.layout = Matrix::ROW_MAJOR;
//	output.data = params.output3;
//	if (output.rows * output.cols != params.output3_len) {
//		printf("ERROR: Output matrix has wrong dimensions: %lu x %lu != %lu\n", output.rows, output.cols, params.output3_len);
//	}

//	Matrix bias3;
//	bias3.rows = NUM_DIGITS;
//	bias3.cols = 1;
//	bias3.data = params.bias3;
//	if (bias3.rows * bias3.cols != params.bias3_len) {
//		printf("ERROR: Bias3 has wrong dimensions: %lu x %lu != %lu\n", bias3.rows, bias3.cols, params.bias3_len);
//	}

	d_set_bias(params.output3, params.bias3);
	d_mul_add(params.output3, params.W23, params.output2);
	d_apply_activation(params.output3, params.activationFunction3);

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

//	Matrix targetOutput;
//	targetOutput.rows = NUM_DIGITS;
//	targetOutput.cols = params.batchSize;
//	targetOutput.data = params.tmp1;
//	if (targetOutput.rows * targetOutput.cols != params.tmp1_len) {
//		printf("d_back_propagate_output: targetOutput matrix has wrong dimensions: %lu x %lu != %lu\n", targetOutput.rows, targetOutput.cols, params.tmp1_len);
//	}
	Matrix targetOutput = params.tmp1;

	// Compute the target output based on the labels
	d_fill_target_output(params, targetOutput);

//	Matrix output;
//	output.rows = NUM_DIGITS;
//	output.cols = params.batchSize;
//	output.data = params.output3;
//	if (output.rows * output.cols != params.output3_len) {
//		printf("d_back_propagate_output: Output matrix has wrong dimensions: %lu x %lu != %lu\n", output.rows, output.cols, params.output3_len);
//	}

	// Save the difference into the target output buffer
	Matrix difference = targetOutput;
	// Reuse the output buffer for saving the error, for now. Perhaps this is a problem later on.
	Matrix error = params.output3;

	d_cwise_sub(difference, targetOutput, params.output3);
	d_apply_activation_derivative(params.output3, params.activationFunction3);
	d_cwise_mul(error, params.output3, difference);

//	Matrix hiddenOutput;
//	hiddenOutput.rows = params.batchSize;
//	hiddenOutput.cols = params.numHiddenNodes;
//	hiddenOutput.layout = Matrix::ROW_MAJOR;
//	hiddenOutput.data = params.output2;
//	if (hiddenOutput.rows * hiddenOutput.cols != params.output2_len) {
//		printf("d_back_propagate_output: hidden output matrix has wrong dimensions: %lu x %lu != %lu\n", hiddenOutput.rows, hiddenOutput.cols, params.output2_len);
//	}

//	Matrix W23;
//	W23.rows = NUM_DIGITS;
//	W23.cols = params.numHiddenNodes;
//	W23.data = params.W23;
//	if (W23.rows * W23.cols != params.W23_len) {
//		printf("d_back_propagate_output: W23 matrix has wrong dimensions: %lu x %lu != %lu\n", W23.rows, W23.cols, params.W23_len);
//	}

	d_mul_add(params.W23, error, params.output2);
}

__device__ void d_back_propagate_hidden(GPUTrainingParameters const params) {
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		printf("d_back_propagate_hidden\n");
	}

	// e3 == output, see "back propagation output"
	// The weight updates are computed by
	// W23^T * e3 * ∇σ * input^T

	Matrix W23 = params.W23;
//	W23.rows = NUM_DIGITS;
//	W23.cols = params.numHiddenNodes;
//	W23.data = params.W23;
//	if (W23.rows * W23.cols != params.W23_len) {
//		printf("d_back_propagate_output: W23 matrix has wrong dimensions: %lu x %lu != %lu\n", W23.rows, W23.cols, params.W23_len);
//	}

	Matrix error = params.output3;
//	error.rows = NUM_DIGITS;
//	error.cols = params.batchSize;
//	error.data = params.output3;
//	if (error.rows * error.cols != params.output3_len) {
//		printf("d_back_propagate_output: error matrix has wrong dimensions: %lu x %lu != %lu\n", error.rows, error.cols, params.output3_len);
//	}

//	Matrix output;
//	output.rows = params.numHiddenNodes;
//	output.cols = params.batchSize;
//	output.data = params.output2;
//	if (output.rows * output.cols != params.output2_len) {
//		printf("d_back_propagate_output: output vector has wrong dimensions: %lu x %lu != %lu\n", output.rows, output.cols, params.output2_len);
//	}

//	Matrix tmp;
//	tmp.rows = params.numHiddenNodes;
//	tmp.cols = params.batchSize;
//	tmp.data = params.tmp2;
//	if (tmp.rows * tmp.cols != params.tmp2_len) {
//		printf("d_back_propagate_output: tmp vector has wrong dimensions: %lu x %lu != %lu\n", tmp.rows, tmp.cols, params.tmp2_len);
//	}

//	Matrix W12;
//	W12.rows = params.width * params.height;
//	W12.cols = params.numHiddenNodes;
//	W12.data = params.W12;
//	if (W12.rows * W12.cols != params.W12_len) {
//		printf("d_back_propagate_output: W12 has wrong dimensions: %lu x %lu != %lu\n", W12.rows, W12.cols, params.W12_len);
//	}

	Matrix images;
	images.rows = params.width * params.height;
	images.cols = params.batchSize;
	images.layout = Matrix::COLUMN_MAJOR;
	images.data = params.images;

	d_apply_activation_derivative(params.output2, params.activationFunction2);
	d_mul(params.tmp2, W23, error);
	d_cwise_mul(params.tmp2, params.output2, params.tmp2);
	d_mul_add(params.W12, params.tmp2, images);
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

__device__ void d_mul(Matrix C, Matrix const A, Matrix const B) {
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		printf("d_mul\n");
	}
	d_mul_base(C, A, B, &d_assign);
}

__device__ void d_mul_add(Matrix C, Matrix const A, Matrix const B) {
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
__device__ void d_mul_base(Matrix C, Matrix const A, Matrix const B, void(*op)(float*, float const, float const)) {

	if (A.cols != B.rows) {

		printf("d_mul_base: Incompatible matrices: (%lu, %lu) x (%lu, %lu)\n", A.rows, A.cols, B.rows, B.cols);
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

__device__ void d_cwise_sub(Matrix C, Matrix const A, Matrix const B) {
	d_cwise_op(C, A, B, &d_sub);
}

__device__ void d_cwise_mul(Matrix C, Matrix const A, Matrix const B) {
	d_cwise_op(C, A, B, &d_mul);
}

__device__ void d_cwise_op(Matrix C, Matrix const A, Matrix const B, void(*op)(float*, float const, float const)) {

	if (A.cols != B.cols || A.rows != B.rows || B.cols != C.cols || B.rows != C.rows) {

		printf("d_cwise_op: Incompatible matrices: (%lu, %lu) + (%lu, %lu) = (%lu, %lu)\n", A.rows, A.cols, B.rows, B.cols, C.rows, C.cols);
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
