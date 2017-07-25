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

#define MATRIX_SIZE_DIVISOR 7
#define NUM_DIGITS 10
#define BATCH_SIZE 600

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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void feedForwardBatch(GPUTrainingParameters const&);
void backPropagateBatch(GPUTrainingParameters const&);

__host__ GPUTrainingParameters initTrainingParams(NeuralNetwork& net,
		size_t const batchSize, size_t const batchOffset, float* const d_images,
		size_t const imageSize, float* const d_labels, size_t const labelSize);
__host__ void freeTrainingParams(GPUTrainingParameters& params);

__host__ void NeuralNetworkCUDA::train(MNISTImageDataset const& images,
		MNISTLableDataset const& labels, double const training_error_threshold,
		double const max_derivation) {

	if (images.size() <= 0)
		return;
	if (labels.size() <= 0)
		return;

	cudaError_t err;

	size_t const singleImgPixCount = images.front().total();
	size_t const allImgBufElements = singleImgPixCount * images.size();

	//
	// Convert the image data and labels into a float array.
	//
	float* fImgData = new float[allImgBufElements];
	float* dst = fImgData;
	for (cv::Mat const& img : images) {
		for (uint8_t* src = img.datastart; src != img.dataend; ++src, ++dst) {
			*dst = static_cast<float>(*src);
		}
	}

	float* flabels = new float[labels.size() * NUM_DIGITS];
	dst = flabels;
	for (uint8_t const& l : labels) {
		for (uint8_t i = 0; i < NUM_DIGITS; ++i, ++dst) {
			*dst = (l == i) ? 1.0f : 0.0f;
		}
	}

	//
	// Copy all the images onto the GPU.
	//
	float* d_images = nullptr;
	float* d_labels = nullptr;

	err = cudaMalloc((void**) &d_images, allImgBufElements * sizeof(float));
	assert(err == cudaSuccess);
	err = cudaMalloc((void**) &d_labels, labels.size() * NUM_DIGITS * sizeof(float));
	assert(err == cudaSuccess);
	err = cudaMemcpy(d_images, fImgData, allImgBufElements * sizeof(float), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);
	err = cudaMemcpy(d_labels, flabels, labels.size() * NUM_DIGITS * sizeof(float), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);

	// Delete the image and label buffers on the host.
	delete[] fImgData;
	fImgData = nullptr;
	delete[] flabels;
	flabels = nullptr;


	GPUTrainingParameters trainingParams = initTrainingParams(*this, BATCH_SIZE, 0, d_images, singleImgPixCount, d_labels, NUM_DIGITS);
	cout << "Batch size: " << trainingParams.batchSize << endl;
	// Configure Grid, i.e. setup Blocks and Threads
	dim3 numBlocks(
			(singleImgPixCount - 1) / MATRIX_SIZE_DIVISOR + 1,
			(singleImgPixCount - 1) / MATRIX_SIZE_DIVISOR + 1);
	dim3 threadsPerBlock(MATRIX_SIZE_DIVISOR, MATRIX_SIZE_DIVISOR);
	cout << "Blocks:            (" << numBlocks.x << ", " << numBlocks.y << ")"
			<< endl;
	cout << "Threads per block: (" << threadsPerBlock.x << ", "
			<< threadsPerBlock.y << ")" << endl;

	//for (int batchId = 0; batchId < images.size() / trainingParams.batchSize; ++batchId)
	{
		//cout << "Processing batch " << batchId << endl;
		trainingParams.images.data = d_images + singleImgPixCount * trainingParams.batchSize;
		trainingParams.labels.data = d_labels + trainingParams.batchSize;
		// Call graphics card functions
		feedForwardBatch(trainingParams);
		backPropagateBatch(trainingParams);
	}

	//
	// Retreive the data
	//
	float* W12 = new float[matrix_size(trainingParams.W12)];
	float* W23 = new float[matrix_size(trainingParams.W23)];
	float* bias2 = new float[matrix_size(trainingParams.bias2)];
	float* bias3 = new float[matrix_size(trainingParams.bias3)];

	// Copy it back to neural network data structure
	err = cudaMemcpy(W12, trainingParams.W12.data, matrix_size(trainingParams.W12) * sizeof(float), cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);
	err = cudaMemcpy(W23, trainingParams.W23.data, matrix_size(trainingParams.W23) * sizeof(float), cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);
	err = cudaMemcpy(bias2, trainingParams.bias2.data, matrix_size(trainingParams.bias2) * sizeof(float), cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);
	err = cudaMemcpy(bias3, trainingParams.bias3.data, matrix_size(trainingParams.bias3) * sizeof(float), cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);

	//
	// Copy the weight data into the c++ data structure.
	//
	Layer* const inputLayer  = getLayer(INPUT);
	Layer* const hiddenLayer = getLayer(HIDDEN);
	Layer* const outputLayer = getLayer(OUTPUT);
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

	freeTrainingParams(trainingParams);

	// Free the cuda buffers
	cudaFree (d_images);
	d_images = nullptr;
	cudaFree (d_labels);
	d_labels = nullptr;
}

__host__ GPUTrainingParameters initTrainingParams(NeuralNetwork& net, size_t const batchSize,
		size_t const batchOffset, float* const d_images,
		size_t const imageSize, float* const d_labels,
		size_t const labelSize) {

	NeuralNetwork::Layer* const inputLayer  = net.getLayer(NeuralNetwork::INPUT);
	NeuralNetwork::Layer* const hiddenLayer = net.getLayer(NeuralNetwork::HIDDEN);
	NeuralNetwork::Layer* const outputLayer = net.getLayer(NeuralNetwork::OUTPUT);

	GPUTrainingParameters trainingParams;
	trainingParams.numHiddenNodes = hiddenLayer->nodes.size();
	//trainingParams.errorThreshold = training_error_threshold;
	//trainingParams.maxDerivation = max_derivation;
	trainingParams.batchSize = batchSize;
	trainingParams.learningRate = net.learningRate;

	// Set the image and labels matrices
	trainingParams.images.rows = imageSize;
	trainingParams.images.cols = batchSize;
	trainingParams.images.layout = Matrix::COLUMN_MAJOR;
	trainingParams.images.data = d_images + imageSize * batchOffset;

	trainingParams.labels.rows = NUM_DIGITS;
	trainingParams.labels.cols = batchSize;
	trainingParams.labels.layout = Matrix::COLUMN_MAJOR;
	trainingParams.labels.data = d_labels + labelSize * batchOffset;

	cudaError_t err;

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
			NeuralNetwork::Layer::Node* node = hiddenLayer->nodes[j];
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
			NeuralNetwork::Layer::Node* node = outputLayer->nodes[j];
			bias3[j] = node->bias;
			for (size_t i = 0; i < node->weights.size(); ++i, ++k) {
				W23[k] = node->weights[i];
			}
		}
	}

	//
	// Copy data to graphics card
	//
	err = cudaMemcpy(trainingParams.W12.data, W12, matrix_size(trainingParams.W12) * sizeof(float), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);
	err = cudaMemcpy(trainingParams.bias2.data, bias2, matrix_size(trainingParams.bias2) * sizeof(float), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);
	err = cudaMemcpy(trainingParams.W23.data, W23, matrix_size(trainingParams.W23) * sizeof(float), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);
	err = cudaMemcpy(trainingParams.bias3.data, bias3, matrix_size(trainingParams.bias3) * sizeof(float), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);

	delete[] W12;
	delete[] W23;
	delete[] bias2;
	delete[] bias3;

	return trainingParams;
}

void freeTrainingParams(GPUTrainingParameters& trainingParams) {
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
void backPropagateOutput(GPUTrainingParameters const&);
void backPropagateHidden(GPUTrainingParameters const&);
__device__ void d_apply_activation(Matrix const&, NeuralNetwork::ActFctType);
__device__ void d_apply_activation_derivative(Matrix const&, NeuralNetwork::ActFctType);
__device__ void d_set_bias(Matrix const& output, Matrix const& bias);
__device__ void d_update_bias(Matrix const& bias, Matrix const& error);

/* Utility functions */
__device__ void d_fill(Matrix const&, float const);

__global__ void feedForwardLayer(Matrix const input, Matrix const weights,
		Matrix const bias, NeuralNetwork::ActFctType actFct,
		Matrix const output) {

	d_set_bias(output, bias);
	d_mul_add(output, weights, input);
	d_apply_activation(output, actFct);
}

void feedForwardBatch(GPUTrainingParameters const& params) {

	cout << "feedForwardBatch" << endl;
	size_t const largestMatDim = params.images.rows;
	dim3 numBlocks(
			(largestMatDim - 1) / MATRIX_SIZE_DIVISOR + 1,
			(largestMatDim - 1) / MATRIX_SIZE_DIVISOR + 1);
	dim3 threadsPerBlock(MATRIX_SIZE_DIVISOR, MATRIX_SIZE_DIVISOR);

	feedForwardLayer<<<numBlocks, threadsPerBlock>>>(params.images, params.W12, params.bias2, params.activationFunction2, params.output2);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	feedForwardLayer<<<numBlocks, threadsPerBlock>>>(params.output2, params.W23, params.bias3, params.activationFunction3, params.output3);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

void backPropagateBatch(GPUTrainingParameters const& params) {

	cout << "backPropagateBatch" << endl;
	backPropagateOutput(params);
	backPropagateHidden(params);
}

/** Saves the error in output3! */
__global__ void calculateOutputError(GPUTrainingParameters const params) {

	// Save the difference into the target output buffer
	Matrix const& difference = params.tmp3;
	d_cwise_sub(difference, params.labels, params.output3);

	// Reuse the output buffer for saving the error, for now. Perhaps this is a problem later on.
	Matrix const& error = params.output3;
	d_apply_activation_derivative(params.output3, params.activationFunction3);
	d_cwise_mul(error, params.output3, difference);
	d_cwise_mul(error, error, params.learningRate);
}

__global__ void updateWeightsAndBias(Matrix const weights, Matrix const bias,
		Matrix const errors, Matrix const transposedLayerInput) {

	d_mul_add(weights, errors, transposedLayerInput);
	d_update_bias(bias, errors);
}

void backPropagateOutput(GPUTrainingParameters const& params) {

	cout << "backPropagateOutput" << endl;
	size_t const largestMatDim = params.images.rows;
	dim3 numBlocks(
			(largestMatDim - 1) / MATRIX_SIZE_DIVISOR + 1,
			(largestMatDim - 1) / MATRIX_SIZE_DIVISOR + 1);
	dim3 threadsPerBlock(MATRIX_SIZE_DIVISOR, MATRIX_SIZE_DIVISOR);

	calculateOutputError<<<numBlocks, threadsPerBlock>>>(params);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	Matrix output2 = matrix_transpose(params.output2);
	// output3 contains the errors.
	Matrix const& error = params.output3;
	updateWeightsAndBias<<<numBlocks, threadsPerBlock>>>(params.W23, params.bias3, error, output2);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

__global__ void calculateHiddenError(Matrix const transposedPreviousWeights,
		Matrix const previousErrors, Matrix const hiddenOutput,
		Matrix const outError, NeuralNetwork::ActFctType actFct) {

	// Backpropagate the error.
	d_mul(outError, transposedPreviousWeights, previousErrors);

	// And then compute the weight update
	d_apply_activation_derivative(hiddenOutput, actFct);
	d_cwise_mul(outError, outError, hiddenOutput);
}

void backPropagateHidden(GPUTrainingParameters const& params) {

	cout << "backPropagateHidden" << endl;
	size_t const largestMatDim = params.images.rows;
	dim3 numBlocks(
			(largestMatDim - 1) / MATRIX_SIZE_DIVISOR + 1,
			(largestMatDim - 1) / MATRIX_SIZE_DIVISOR + 1);
	dim3 threadsPerBlock(MATRIX_SIZE_DIVISOR, MATRIX_SIZE_DIVISOR);

	// The weight updates are computed by
	// W23^T * e3 * ∇σ * input^T

	// Important to make a local copy.
	// Otherwise every thread would transpose the matrix which
	// would lead to undefined behavior.
	Matrix W23 = matrix_transpose(params.W23);

	// See d_back_propagation_output
	// Already contains the learningRate.
//	Matrix const& error = params.output3;
	Matrix const& previousError = params.output3;
	Matrix const& error = params.tmp2;

//	// Backpropagate the error.
//	d_mul(params.tmp2, W23, error);
	calculateHiddenError<<<numBlocks, threadsPerBlock>>>(
			W23, previousError, params.output2, error,
			params.activationFunction3);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

//	d_update_bias(params.bias2, params.tmp2);

//	// And then compute the weight update
//	d_apply_activation_derivative(params.output2, params.activationFunction2);
//	d_cwise_mul(params.tmp2, params.output2, params.tmp2);


	Matrix images = matrix_transpose(params.images);
//	d_mul_add(params.W12, params.tmp2, images);
	updateWeightsAndBias<<<numBlocks, threadsPerBlock>>>(params.W12,
			params.bias2, error, images);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
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

	// If this thread has nothing to do, because it would access invalid memory, exit
	if (x >= C.cols || y >= C.rows) {
		return;
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

		PRINTF("d_cwise_op: Incompatible matrices: (%lu, %lu) + (%lu, %lu) = (%lu, %lu)\n", A.rows, A.cols, B.rows, B.cols, C.rows, C.cols);
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

		PRINTF("d_cwise_op: Incompatible matrices: v * (%lu, %lu) = (%lu, %lu)\n", A.rows, A.cols, C.rows, C.cols);
		return;
	}

	size_t const x = blockIdx.x * blockDim.x + threadIdx.x;
	size_t const y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= A.cols || y >= A.rows) {
		return;
	}

	op(d_matrix_pget(C, y, x), d_matrix_get(A, y, x), v);
}

__device__ void d_fill(Matrix const& A, float const v) {

	size_t const targetX = threadIdx.x + blockIdx.x * blockDim.x;
	size_t const targetY = threadIdx.y + blockIdx.y * blockDim.y;

	if (targetX >= A.cols || targetY >= A.rows) {
		return;
	}

	d_matrix_set(A, targetY, targetX, v);
}

__device__ void d_update_bias(Matrix const& bias, Matrix const& error) {

	if (bias.rows != error.rows|| bias.cols != 1) {

		PRINTF("d_update_bias: Invalid matrices: bias(%lu, %lu) error(%lu, %lu)\n", bias.rows, bias.cols, error.rows, error.cols);
		return;
	}

	// The block caches are row major.
	__shared__ float blockCacheError[MATRIX_SIZE_DIVISOR][MATRIX_SIZE_DIVISOR];

	// Compute the target coordinates.
	size_t const x = blockIdx.x * MATRIX_SIZE_DIVISOR + threadIdx.x;
	size_t const y = blockIdx.y * MATRIX_SIZE_DIVISOR + threadIdx.y;

	// If this thread has nothing to do, because it would access invalid memory, exit
	if (y >= bias.rows || x >= bias.cols) {
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

	*d_matrix_pget(bias, y, x) += threadValue;
}
