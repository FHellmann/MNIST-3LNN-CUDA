#include "NeuralNetworkCUDA.h"

#include <iostream>

using namespace std;

__host__ NeuralNetworkCUDA::NeuralNetworkCUDA(const int inpCount, const int hidCount,
		const int outCount, const double learningRate) :
			NeuralNetwork(learningRate) {
}

__host__ NeuralNetworkCUDA::~NeuralNetworkCUDA() {
}

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

	// Allocate cuda memory
	uint8_t* d_ImgData = nullptr;
	err = cudaMalloc((void**) &d_ImgData, allImgBufElements * sizeof(uint8_t));
	assert(err == cudaSuccess);
	uint8_t* d_Labels = nullptr;
	err = cudaMalloc((void**) &d_Labels, labels.size() * sizeof(uint8_t));
	assert(err == cudaSuccess);

	// Copy data to graphics card
	err = cudaMemcpy(d_ImgData, imgData, allImgBufElements * sizeof(uint8_t),
			cudaMemcpyHostToDevice);
	assert(err == cudaSuccess);
	err = cudaMemcpy(d_Labels, labels.data(), labels.size() * sizeof(uint8_t),
			cudaMemcpyHostToDevice);

	delete[] imgData;
	imgData = nullptr;

	// Configure Grid, i.e. setup Blocks and Threads
//	dim3 threadsPerBlock(MATRIX_SIZE_DIVISOR, MATRIX_SIZE_DIVISOR);
//	dim3 numBlocks(C.cols() / MATRIX_SIZE_DIVISOR,
//			C.rows() / MATRIX_SIZE_DIVISOR);
//	cout << "Threads per block: (" << threadsPerBlock.x << ", "
//			<< threadsPerBlock.y << ")" << endl;
//	cout << "Blocks:            (" << numBlocks.x << ", " << numBlocks.y << ")"
//			<< endl;

	// Call graphics card functions
//	d_mul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, A.rows());

	// Retreive the data
//	err = cudaMemcpy(C.data(), d_C, C.size() * sizeof(float),
//			cudaMemcpyDeviceToHost);
//	assert(err == cudaSuccess);

	// Copy it back to neural network datastructure

	// Free the cuda buffers
	cudaFree(d_ImgData);
	d_ImgData = nullptr;
	cudaFree(d_Labels);
	d_Labels = nullptr;
}
