#include "NeuralNetworkCUDA.h"

#include <iostream>

using namespace std;

__host__ NeuralNetworkCUDA::NeuralNetworkCUDA(const int inpCount,
		const int hidCount, const int outCount, const double learningRate) :
		NeuralNetwork(learningRate) {
}

__host__ NeuralNetworkCUDA::~NeuralNetworkCUDA() {
}

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
};

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

	GPUTrainingParameters trainingParams;
	trainingParams.numExamples = images.size();
	trainingParams.width = images.front().cols;
	trainingParams.height = images.front().rows;
	trainingParams.numHiddenNodes = getLayer(HIDDEN)->nodes.size();
	trainingParams.errorThreshold = training_error_threshold;

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
	err = cudaMalloc((void**) &trainingParams.W1, 0);
	assert(err == cudaSuccess);

	// Storage for the first weight matrix
	err = cudaMalloc((void**) &trainingParams.W2, 0);
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
	dim3 numBlocks(32, 32);
	dim3 threadsPerBlock(16, 16);
	cout << "Blocks:            (" << numBlocks.x << ", " << numBlocks.y << ")"
			<< endl;
	cout << "Threads per block: (" << threadsPerBlock.x << ", "
			<< threadsPerBlock.y << ")" << endl;

	// Call graphics card functions
	trainCUDA<<<numBlocks, threadsPerBlock>>>(trainingParams);

	// Retreive the data
//	err = cudaMemcpy(C.data(), d_C, C.size() * sizeof(float),
//			cudaMemcpyDeviceToHost);
//	assert(err == cudaSuccess);

	// Copy it back to neural network datastructure

	// Free the cuda buffers
	cudaFree (trainingParams.images);
	trainingParams.images = nullptr;
	cudaFree (trainingParams.labels);
	trainingParams.labels = nullptr;
	cudaFree (trainingParams.W1);
	trainingParams.W1 = nullptr;
	cudaFree (trainingParams.W2);
	trainingParams.W2 = nullptr;
}

__global__ void trainCUDA(GPUTrainingParameters const params) {

}
