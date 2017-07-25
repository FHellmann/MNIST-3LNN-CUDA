#include "NeuralNetworkCUDA.h"

#include <iostream>
#include <cmath>
#include "cudaUtility.h"

using namespace std;

__host__ NeuralNetworkCUDA::NeuralNetworkCUDA(const int inpCount,
		const int hidCount, const int outCount, const double learningRate) :
		NeuralNetwork(inpCount, hidCount, outCount, learningRate) {
}

__host__ NeuralNetworkCUDA::~NeuralNetworkCUDA() {
}

#define NUM_DIGITS 10
#define BATCH_SIZE 1


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
			// Binarize the image to {0.0, 1.0}
			*dst = (*src > 128) ? 1.0f : 0.0f;
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

void feedForwardBatch(GPUTrainingParameters const& params) {

	PRINTF("feedForwardBatch\n");
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

/* Neural network operations. */
void backPropagateOutput(GPUTrainingParameters const&);
void backPropagateHidden(GPUTrainingParameters const&);

void backPropagateBatch(GPUTrainingParameters const& params) {

	PRINTF("backPropagateBatch\n");
	backPropagateOutput(params);
	backPropagateHidden(params);
}

void backPropagateOutput(GPUTrainingParameters const& params) {

	PRINTF("backPropagateOutput\n");
	size_t const largestMatDim = params.images.rows;
	dim3 blocks(
			(largestMatDim - 1) / MATRIX_SIZE_DIVISOR + 1,
			(largestMatDim - 1) / MATRIX_SIZE_DIVISOR + 1);
	dim3 threads(MATRIX_SIZE_DIVISOR, MATRIX_SIZE_DIVISOR);

	// output3 contains the errors.
	Matrix const& error = params.output3;

	calculateOutputError<<<blocks, threads>>>(params.output3,
			params.labels, error, params.tmp3, params.learningRate,
			params.activationFunction3);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	//fill<<<blocks, threads>>>(params.W23, 0.0f);
	//fill<<<blocks, threads>>>(params.output3, 2.0f);
	//fill<<<blocks, threads>>>(params.output2, 1.0f);

	updateBias<<<blocks, threads>>>(params.bias3, error);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	Matrix const output2Transposed = matrix_transpose(params.output2);
	updateWeights<<<blocks, threads>>>(params.W23, params.bias3, error, output2Transposed);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}

void backPropagateHidden(GPUTrainingParameters const& params) {

	PRINTF("backPropagateHidden");
	size_t const largestMatDim = params.images.rows;
	dim3 blocks(
			(largestMatDim - 1) / MATRIX_SIZE_DIVISOR + 1,
			(largestMatDim - 1) / MATRIX_SIZE_DIVISOR + 1);
	dim3 threads(MATRIX_SIZE_DIVISOR, MATRIX_SIZE_DIVISOR);

	// The weight updates are computed by
	// W23^T * e3 * ∇σ * input^T
	Matrix W23Transposed = matrix_transpose(params.W23);

	// See d_back_propagation_output
	// Already contains the learningRate.
	Matrix const& previousError = params.output3;
	Matrix const& error = params.tmp2;

	backpropagateOutputError<<<blocks, threads>>>(W23Transposed,
			previousError, error, params.learningRate);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	updateBias<<<blocks, threads>>>(params.bias2, error);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	// Backpropagate the error.
	finalizeHiddenError<<<blocks, threads>>>(
			params.output2, error, params.activationFunction2);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	//fill_pattern<<<blocks, threads>>>(params.W12);
	//fill<<<blocks, threads>>>(params.W12, 0.0f);
	//fill<<<blocks, threads>>>(error, 1.0f);
	//fill<<<blocks, threads>>>(params.images, 1.0f);

	Matrix const imagesTransposed = matrix_transpose(params.images);
	updateWeights<<<blocks, threads>>>(params.W12,
			params.bias2, error, imagesTransposed);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
}
