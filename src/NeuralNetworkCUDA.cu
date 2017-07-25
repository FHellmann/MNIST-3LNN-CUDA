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
	float* W1;
	size_t W1_len;
	float* W2;
	size_t W2_len;

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

__global__ void trainCUDA(GPUTrainingParameters const, GPUSharedMemoryLayout const);

__host__ double NeuralNetworkCUDA::train(MNISTImageDataset const& images,
		MNISTLableDataset const& labels, double const training_error_threshold,
		double const max_derivation) {

	if (images.size() <= 0)
		return 0;
	if (labels.size() <= 0)
		return 0;

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
	err = cudaMalloc((void**) &trainingParams.images,
			allImgBufElements * sizeof(uint8_t));
	assert(err == cudaSuccess);

	// Labels
	err = cudaMalloc((void**) &trainingParams.labels,
			labels.size() * sizeof(uint8_t));
	assert(err == cudaSuccess);

	// Storage for the first weight matrix
	trainingParams.W1_len = inputLayer->nodes.size() * hiddenLayer->nodes.size();
	err = cudaMalloc((void**) &trainingParams.W1, trainingParams.W1_len * sizeof(float));
	assert(err == cudaSuccess);

	// Storage for the first weight matrix
	trainingParams.W2_len = hiddenLayer->nodes.size() * outputLayer->nodes.size();
	err = cudaMalloc((void**) &trainingParams.W2, trainingParams.W2_len * sizeof(float));
	assert(err == cudaSuccess);

	//
	// Copy data to graphics card
	//
	err = cudaMemcpy(trainingParams.images, imgData,
			allImgBufElements * sizeof(uint8_t), cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);
	err = cudaMemcpy(trainingParams.labels, labels.data(),
			labels.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);

	delete[] imgData;
	imgData = nullptr;

	// Configure Grid, i.e. setup Blocks and Threads
	dim3 numBlocks(1);
	dim3 threadsPerBlock(MATRIX_SIZE_DIVISOR, MATRIX_SIZE_DIVISOR);
	cout << "Blocks:            (" << numBlocks.x << ", " << numBlocks.y << ")"
			<< endl;
	cout << "Threads per block: (" << threadsPerBlock.x << ", "
			<< threadsPerBlock.y << ")" << endl;

	size_t sharedMemorySize = 0;

	// Size of the first weight matrix
	gpuSharedMemoryLayout.W1_pos = 0;
	gpuSharedMemoryLayout.W1_size = trainingParams.W1_len * sizeof(float);
	sharedMemorySize += gpuSharedMemoryLayout.W1_size;

	// Size of the second weight matrix
	gpuSharedMemoryLayout.W2_pos = gpuSharedMemoryLayout.W1_pos + gpuSharedMemoryLayout.W1_size;
	gpuSharedMemoryLayout.W2_size = trainingParams.W2_len * sizeof(float);
	sharedMemorySize += gpuSharedMemoryLayout.W2_size;

	// Size of the hidden layer output nodes
	gpuSharedMemoryLayout.hiddenOutput_pos = gpuSharedMemoryLayout.W2_pos + gpuSharedMemoryLayout.W2_size;
	gpuSharedMemoryLayout.hiddenOutput_size = hiddenLayer->nodes.size() * sizeof(float);
	sharedMemorySize += gpuSharedMemoryLayout.hiddenOutput_size;

	// Size of the output layer output values
	gpuSharedMemoryLayout.outputOutput_pos = gpuSharedMemoryLayout.hiddenOutput_pos + gpuSharedMemoryLayout.hiddenOutput_size;
	gpuSharedMemoryLayout.outputOutput_size = outputLayer->nodes.size() * sizeof(float);
	sharedMemorySize += gpuSharedMemoryLayout.outputOutput_size;

	// Size of the hidden bias vector
	gpuSharedMemoryLayout.hiddenBias_pos   = gpuSharedMemoryLayout.outputOutput_pos + gpuSharedMemoryLayout.outputOutput_size;
	gpuSharedMemoryLayout.hiddenBias_size  = hiddenLayer->nodes.size() * sizeof(float);
	sharedMemorySize += gpuSharedMemoryLayout.hiddenBias_size;

	// Size of the input bias vector
	gpuSharedMemoryLayout.inputBias_pos    = gpuSharedMemoryLayout.hiddenOutput_pos + gpuSharedMemoryLayout.hiddenOutput_size;
	gpuSharedMemoryLayout.inputBias_size   = inputLayer->nodes.size() * sizeof(float);
	sharedMemorySize += gpuSharedMemoryLayout.inputBias_size;

	// Size of the input vector
	gpuSharedMemoryLayout.image_pos        = gpuSharedMemoryLayout.inputBias_pos + gpuSharedMemoryLayout.inputBias_size;
	gpuSharedMemoryLayout.image_size       = inputLayer->nodes.size() * sizeof(uint8_t);
	sharedMemorySize += gpuSharedMemoryLayout.image_size;

	// Call graphics card functions
	trainCUDA<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(trainingParams, gpuSharedMemoryLayout);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	//
	// Retreive the data
	//

	float* W1 = new float[trainingParams.W1_len];
	float* W2 = new float[trainingParams.W2_len];

	// Copy it back to neural network data structure
	err = cudaMemcpy(W1, trainingParams.W1, trainingParams.W1_len * sizeof(float), cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);
	err = cudaMemcpy(W2, trainingParams.W2, trainingParams.W2_len * sizeof(float), cudaMemcpyDeviceToHost);
	assert(err == cudaSuccess);

	// Free the cuda buffers
	cudaFree (trainingParams.images);
	trainingParams.images = nullptr;
	cudaFree (trainingParams.labels);
	trainingParams.labels = nullptr;
	cudaFree (trainingParams.W1);
	trainingParams.W1 = nullptr;
	cudaFree (trainingParams.W2);
	trainingParams.W2 = nullptr;

	//
	// Copy the weight data into the c++ data structure.
	//

	// Delete the host buffers
	delete[] W1;
	W1 = nullptr;
	delete[] W2;
	W2 = nullptr;

	return 0;
}

__device__ void feedForward(GPUTrainingParameters const/*, float sharedMem[]*/, GPUSharedMemoryLayout const sharedLayout);
__device__ void backPropagate(float sharedMem[], GPUSharedMemoryLayout const sharedLayout);

__global__ void trainCUDA(GPUTrainingParameters const params, GPUSharedMemoryLayout const sharedLayout) {


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
				feedForward(params /*, sharedMem*/, sharedLayout);

				// Back propagate the error and adjust weights in all layers accordingly
//				nnp_local.backPropagate(labels[imgCount]);
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

__device__ void feedForward(GPUTrainingParameters const params/*, float sharedMem[]*/, GPUSharedMemoryLayout const sharedLayout) {

	__shared__ float* hiddenOutputs[MATRIX_SIZE_DIVISOR][MATRIX_SIZE_DIVISOR];

	Matrix W1;
	W1.rows = MATRIX_SIZE_DIVISOR;
	W1.cols = params.width * params.height;
	W1.data = params.W1;

	Matrix imgs;
	imgs.rows = params.width * params.height;
	imgs.cols = MATRIX_SIZE_DIVISOR;
	imgs.data = new float[params.width * params.height];
	imgs.data[threadIdx.x * threadIdx.y] = params.images[threadIdx.x * threadIdx.y];

	Matrix foobar;
	foobar.rows = MATRIX_SIZE_DIVISOR;
	foobar.cols = MATRIX_SIZE_DIVISOR;
	foobar.data = (float*)hiddenOutputs;

	d_mul_shared(W1, imgs, foobar);

	delete[] imgs.data;
}

__device__ void backPropagate(float sharedMem[], GPUSharedMemoryLayout const sharedLayout) {

}

/**
 * Computes C = AB.
 *
 * @param[in] A first factor of the matrix multiplication.
 * @param[in] B second factor of the multiplication.
 * @param[out] C Matrix holding the result. Must provide enough storage space.
 */
__device__ void d_mul_shared(Matrix A, Matrix B, Matrix C)
{
	__shared__ float blockCacheA[MATRIX_SIZE_DIVISOR][MATRIX_SIZE_DIVISOR];
	__shared__ float blockCacheB[MATRIX_SIZE_DIVISOR][MATRIX_SIZE_DIVISOR];

	// Column major!
	float threadValue = 0.0f;
	unsigned int const numSubBlocks = A.cols / MATRIX_SIZE_DIVISOR;
	for (int k = 0; k < numSubBlocks; ++k)
	{
		blockCacheA[threadIdx.x][threadIdx.y] = A.data[(blockIdx.y + k * A.cols) * MATRIX_SIZE_DIVISOR + threadIdx.y + threadIdx.x * A.cols];
		blockCacheB[threadIdx.y][threadIdx.x] = B.data[(k + A.cols * blockIdx.x) * MATRIX_SIZE_DIVISOR + threadIdx.y + threadIdx.x * A.cols];

		__syncthreads();

	#pragma unroll
		for (int i = 0; i < MATRIX_SIZE_DIVISOR; ++i)
		{
			threadValue += blockCacheA[i][threadIdx.y] * blockCacheB[i][threadIdx.x];
		}

		__syncthreads();
	}

	C.data[(blockIdx.y + A.cols * blockIdx.x) * MATRIX_SIZE_DIVISOR + threadIdx.y + threadIdx.x * A.cols] = threadValue;
}
