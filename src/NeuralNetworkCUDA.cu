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
	uint8_t* images;
	uint8_t* labels;

	/* Training data parameters. */
	size_t numExamples;
	size_t numHiddenNodes;
	size_t width;
	size_t height;

	/* Weight matrices. */
	float* W12;
	size_t W12_len;
	float* bias2;
	size_t bias2_len;
	float* W23;
	size_t W23_len;
	float* bias3;
	size_t bias3_len;

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
	size_t rows;
	size_t cols;
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

__global__ void trainCUDA(GPUTrainingParameters const);

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
	uint8_t* imgData = new uint8_t[allImgBufElements];
	uint8_t* it = imgData;
	for (cv::Mat const& img : images) {
		if (img.isContinuous()) {
			std::copy(img.datastart, img.dataend, it);
		} else {
			cerr << "cv::Mat is not continuous." << endl;
		}
		it += img.total() * img.elemSize();
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

	//
	// Allocate cuda memory
	//

	// Images
	err = cudaMalloc((void**) &trainingParams.images, allImgBufElements * sizeof(uint8_t));
	assert(err == cudaSuccess);

	// Labels
	err = cudaMalloc((void**) &trainingParams.labels, labels.size() * sizeof(uint8_t));
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

	//
	// Copy data to graphics card
	//
	err = cudaMemcpy(trainingParams.images, imgData, allImgBufElements * sizeof(uint8_t), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);
	err = cudaMemcpy(trainingParams.labels, labels.data(),labels.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);

	delete[] imgData;
	imgData = nullptr;

	// Configure Grid, i.e. setup Blocks and Threads
	dim3 numBlocks(1);
	dim3 threadsPerBlock(MATRIX_SIZE_DIVISOR, MATRIX_SIZE_DIVISOR);
	cout << "Blocks:            (" << numBlocks.x << ", " << numBlocks.y << ")"
			<< endl;
	cout << "Threads per block: (" << threadsPerBlock.x << ", "
			<< threadsPerBlock.y << ")" << endl;

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

	// Call graphics card functions
	trainCUDA<<<numBlocks, threadsPerBlock>>>(trainingParams);
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

	//
	// Copy the weight data into the c++ data structure.
	//
	Layer* hidden = getLayer(HIDDEN);
	for (size_t j = 0; j < hidden->nodes.size(); ++j) {
		Layer::Node* node = hidden->nodes[j];
		node->bias = bias2[j];
		for (size_t i = 0; i < node->weights.size(); ++i) {
			node->weights[i] = W12[j * hidden->nodes.size() + i];
		}
	}

	Layer* output = getLayer(OUTPUT);
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

__device__ void printCuda(GPUTrainingParameters const params) {
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

__device__ void feedForward(GPUTrainingParameters const);
__device__ void backPropagate(float sharedMem[]);

__global__ void trainCUDA(GPUTrainingParameters const params) {

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		printCuda(params);
	}
	//
	// Initialize the internal weight matrices for each network.
	//

	// Weight matrices will be column major for better bank access.
	//extern __shared__ float sharedMem[];


	// The layer outputs will be stored in shared vectors.

	//
	// Start training
	//
	bool needsFurtherTraining = true;
	float error = 100000000.0f;
	while (needsFurtherTraining) {

		float newError = 0;

		//NeuralNetworkParallel nnp_merge(*this);

//		{
//			//NeuralNetworkParallel nnp_local(*this);
//			size_t localErrCount = 0;
//
//			for (size_t imgCount = 0; imgCount < images.size(); imgCount++) {
//				// Convert the MNIST image to a standardized vector format and feed into the network
//				nnp_local.feedInput(images[imgCount]);
//
//				// Feed forward all layers (from input to hidden to output) calculating all nodes' output
				//feedForward(params);

				// Back propagate the error and adjust weights in all layers accordingly
				//backPropagate(nullptr);
//
//				// Classify image by choosing output cell with highest output
//				int classification = nnp_local.getNetworkClassification();
//				if (classification != labels[imgCount])
//					localErrCount++;
//
//				// Display progress during training
//				if ((imgCount % every_ten_percent) == 0) {
//					cout << "x";
//					cout.flush();
//				}
//			}
//
//			newError += static_cast<double>(localErrCount) / static_cast<double>(images.size());
//
//			// merge network weights together
//			mergeNeuralNetworks(nnp_local, nnp_merge, this);
//
//			//cout << "Thread-" << omp_get_thread_num() << ": Error=" << localErrCount << ", Images=" << localImageProcessed << endl;
//		}
//
//		mergeNeuralNetworks(nnp_merge, *this, this);

		if (newError < error) {
			error = newError;
		}

		needsFurtherTraining = !(error < params.errorThreshold || newError > error + params.maxDerivation);

//		cout << " Error: " << error * 100.0 << "%, NewError: " << newError * 100.0 << "%" << endl;
	}
}

__device__ void d_mul_shared(Matrix A, Matrix B, Matrix C);

__device__ void feedForward(GPUTrainingParameters const params) {

	__shared__ float* hiddenOutputs[MATRIX_SIZE_DIVISOR][MATRIX_SIZE_DIVISOR];
	__shared__ float* outputs[MATRIX_SIZE_DIVISOR][MATRIX_SIZE_DIVISOR];
	__shared__ float* imageData[MATRIX_SIZE_DIVISOR * MATRIX_SIZE_DIVISOR];
	__shared__ float* alignedW2[MATRIX_SIZE_DIVISOR][MATRIX_SIZE_DIVISOR];

	size_t const numImages = params.numHiddenNodes;

	Matrix W1;
	W1.rows = params.numHiddenNodes;
	W1.cols = params.width * params.height;
	W1.data = params.W12;

	Matrix imgs;
	imgs.rows = params.width * params.height;
	imgs.cols = numImages;
	imgs.data = (float*)imageData;
	imgs.data[threadIdx.x * threadIdx.y] = params.images[threadIdx.x * threadIdx.y];

	Matrix foobar;
	foobar.rows = params.numHiddenNodes;
	foobar.cols = numImages;
	foobar.data = (float*)hiddenOutputs;

	d_mul_shared(W1, imgs, foobar);

	Matrix W2;
	W2.rows = foobar.rows;
	W2.cols = params.numHiddenNodes;
	W2.data = (float*)alignedW2;
	//memcpy(W2.data, params.W2, params.W2_len * sizeof(float));
	//W23.data = params.W23;

	Matrix O;
	O.rows = W2.rows;
	O.cols = numImages;
	O.data = (float*)outputs;

	//d_mul_shared(W2, foobar, O);

	//delete[] imgs.data;
	//delete[] W23.data;
}

__device__ void backPropagate(float sharedMem[]) {

}

/**
 * Computes C = AB where the dimensions of A and be have to be a multiple of MATRIX_SIZE_DIVISOR.
 *
 * Matrices are expected to be row-major.
 *
 * @param[in] A first factor of the matrix multiplication.
 * @param[in] B second factor of the multiplication.
 * @param[out] C Matrix holding the result. Must provide enough storage space.
 */
__device__ void d_mul_shared(Matrix A, Matrix B, Matrix C) {

	if (A.cols != B.rows) {

		printf("Invalid matrix sizes: (%lu, %lu)x(%lu, %lu)\n", A.rows, A.cols, B.rows, B.cols);
		return;
	}

	if (A.cols % MATRIX_SIZE_DIVISOR != 0 ||
	    A.rows % MATRIX_SIZE_DIVISOR != 0 ||
	    B.cols % MATRIX_SIZE_DIVISOR != 0 ||
	    B.rows % MATRIX_SIZE_DIVISOR != 0) {

		printf("Matrix dimensions not a multiple of %hu: (%lu, %lu)x(%lu, %lu)\n", MATRIX_SIZE_DIVISOR, A.rows, A.cols, B.rows, B.cols);
		return;
	}

	__shared__ float blockCacheA[MATRIX_SIZE_DIVISOR][MATRIX_SIZE_DIVISOR];
	__shared__ float blockCacheB[MATRIX_SIZE_DIVISOR][MATRIX_SIZE_DIVISOR];

	// Column major!
	float threadValue = 0.0f;
	unsigned int const numSubBlocks = A.cols / MATRIX_SIZE_DIVISOR;
	for (int k = 0; k < numSubBlocks; ++k)
	{
		blockCacheA[threadIdx.x][threadIdx.y] = A.data[k * MATRIX_SIZE_DIVISOR + threadIdx.y * A.cols + threadIdx.x];
		printf("idx: %lu\n", k * B.cols * MATRIX_SIZE_DIVISOR + threadIdx.y * B.cols + threadIdx.x);
		blockCacheB[threadIdx.y][threadIdx.x] = B.data[k * B.cols * MATRIX_SIZE_DIVISOR + threadIdx.y * B.cols + threadIdx.x];

		__syncthreads();

	#pragma unroll
		for (int i = 0; i < MATRIX_SIZE_DIVISOR; ++i)
		{
			threadValue += blockCacheA[i][threadIdx.y] * blockCacheB[i][threadIdx.x];
		}

		__syncthreads();
	}

	C.data[threadIdx.y * C.cols + threadIdx.x] = threadValue;
}
