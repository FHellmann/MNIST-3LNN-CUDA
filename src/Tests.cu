/*
 * Tests.cpp
 *
 *  Created on: 14.07.2017
 *      Author: Stefan
 */
#include "NeuralNetworkParallel.h"
#include <iostream>
#include "cudaUtility.h"
#include <eigen3/Eigen/Eigen>

using namespace std;

bool ensureDeepCopy(NeuralNetworkParallel const& A, NeuralNetworkParallel const& B)
{
	if (A.learningRate != B.learningRate)
		return false;

	if (A.layers.size() != B.layers.size())
		return false;

	for (size_t i = 0; i < A.layers.size(); ++i)
	{
		NeuralNetworkParallel::Layer* layerA = A.layers[i];
		NeuralNetworkParallel::Layer* layerB = B.layers[i];
		if (layerA == layerB)
			return false;

		if (layerA->actFctType != layerB->actFctType)
			return false;
		if (layerA->layerType != layerB->layerType)
			return false;
		if (layerA->previousLayer == layerB->previousLayer)
			return false;

		for (size_t j = 0; j < A.layers.size(); ++j)
		{
			if (A.layers[j] == layerA->previousLayer)
			{
				if (B.layers[j] != layerB->previousLayer)
					return false;
			}
		}

		if (layerA->nodes.size() != layerB->nodes.size())
			return false;

		for (size_t j = 0; j < layerA->nodes.size(); ++j)
		{
			NeuralNetworkParallel::Layer::Node* nodeA = layerA->nodes[j];
			NeuralNetworkParallel::Layer::Node* nodeB = layerB->nodes[j];

			if (nodeA == nodeB)
				return false;
			if (nodeA->bias != nodeB->bias)
				return false;
			if (nodeA->output != nodeB->output)
				return false;
			if (nodeA->weights.size() != nodeB->weights.size())
				return false;

			for (size_t k = 0; k < nodeA->weights.size(); ++k)
			{
				if (nodeA->weights[k] != nodeB->weights[k])
					return false;
			}
		}
	}

	return true;
}

bool testCUDAMatrixMul() {

	Matrix d_A;
	d_A.rows = 3;
	d_A.cols = 28;
	cudaMalloc((void**)&d_A.data, matrix_size(d_A) * sizeof(float));

	Matrix d_B;
	d_B.rows = d_A.cols;
	d_B.cols = 60;
	cudaMalloc((void**)&d_B.data, matrix_size(d_B) * sizeof(float));

	Matrix d_C;
	d_C.rows = d_A.rows;
	d_C.cols = d_B.cols;
	cudaMalloc((void**)&d_C.data, matrix_size(d_C) * sizeof(float));

	size_t largestMatDim = max(d_A.rows, d_A.cols);
	largestMatDim = max(largestMatDim, d_B.cols);
	dim3 blocks((largestMatDim - 1) / MATRIX_SIZE_DIVISOR + 1, (largestMatDim - 1) / MATRIX_SIZE_DIVISOR + 1);
	dim3 threads(MATRIX_SIZE_DIVISOR, MATRIX_SIZE_DIVISOR);

	printf("blocks(%u, %u), threads(%u, %u)\n", blocks.x, blocks.y, threads.x, threads.y);

	fill_pattern<<<blocks, threads>>>(d_A);
	fill_pattern<<<blocks, threads>>>(d_B);
	mul<<<blocks, threads>>>(d_C, d_A, d_B);

	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajorBit> EigenMatrix;
	EigenMatrix A(d_A.rows, d_A.cols);
	EigenMatrix B(d_B.rows, d_B.cols);
	EigenMatrix C(d_C.rows, d_C.cols);

	cudaMemcpy((void**)A.data(), d_A.data, matrix_size(d_A) * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy((void**)B.data(), d_B.data, matrix_size(d_B) * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy((void**)C.data(), d_C.data, matrix_size(d_C) * sizeof(float), cudaMemcpyDeviceToHost);

	printf("A(%lu, %lu)\n", A.rows(), A.cols());
	cout << A << endl << endl;
	printf("B(%lu, %lu)\n", B.rows(), B.cols());
	cout << B << endl << endl;
	printf("C(%lu, %lu)\n", C.rows(), C.cols());
	cout << C << endl << endl;

	cudaFree(d_A.data);
	cudaFree(d_B.data);
	cudaFree(d_C.data);

	return (A*B).isApprox(C);
}

bool testCUDAMatrixMulTranspose() {

	Matrix d_A;
	d_A.rows = 10;
	d_A.cols = 60;
	cudaMalloc((void**)&d_A.data, matrix_size(d_A) * sizeof(float));

	Matrix d_B;
	d_B.rows = 28;
	d_B.cols = d_A.cols;
	cudaMalloc((void**)&d_B.data, matrix_size(d_B) * sizeof(float));


	size_t largestMatDim = 0;
	largestMatDim = max(largestMatDim, d_A.rows);
	largestMatDim = max(largestMatDim, d_A.cols);
	largestMatDim = max(largestMatDim, d_B.rows);
	largestMatDim = max(largestMatDim, d_B.cols);
	dim3 blocks((largestMatDim - 1) / MATRIX_SIZE_DIVISOR + 1, (largestMatDim - 1) / MATRIX_SIZE_DIVISOR + 1);
	dim3 threads(MATRIX_SIZE_DIVISOR, MATRIX_SIZE_DIVISOR);

	printf("blocks(%u, %u), threads(%u, %u)\n", blocks.x, blocks.y, threads.x, threads.y);

	fill_pattern<<<blocks, threads>>>(d_A);
	fill_pattern<<<blocks, threads>>>(d_B);
	d_B = matrix_transpose(d_B);

	Matrix d_C;
	d_C.rows = d_A.rows;
	d_C.cols = d_B.cols;
	cudaMalloc((void**)&d_C.data, matrix_size(d_C) * sizeof(float));
	fill<<<blocks, threads>>>(d_C, 0.0f);
	mul_add<<<blocks, threads>>>(d_C, d_A, d_B);

	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixRowMajor;
	typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> EigenMatrixColumnMajor;
	EigenMatrixRowMajor A(d_A.rows, d_A.cols);
	EigenMatrixColumnMajor B(d_B.rows, d_B.cols);
	//EigenMatrixRowMajor B(d_B.rows, d_B.cols);
	EigenMatrixRowMajor C(d_C.rows, d_C.cols);

	cudaMemcpy((void**)A.data(), d_A.data, matrix_size(d_A) * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy((void**)B.data(), d_B.data, matrix_size(d_B) * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy((void**)C.data(), d_C.data, matrix_size(d_C) * sizeof(float), cudaMemcpyDeviceToHost);

	printf("A(%lu, %lu)\n", A.rows(), A.cols());
	cout << A << endl << endl;
	printf("B(%lu, %lu)\n", B.rows(), B.cols());
	cout << B << endl << endl;
	printf("C(%lu, %lu)\n", C.rows(), C.cols());
	cout << C << endl << endl;

	cudaFree(d_A.data);
	cudaFree(d_B.data);
	cudaFree(d_C.data);

	return (A*B).isApprox(C);
}


int main(int argc, char* argv[])
{
	NeuralNetworkParallel A(4, 2, 17, 0.2);
	NeuralNetworkParallel B(A);

	if (ensureDeepCopy(A, B) == false)
	{
		cerr << "B is not a deep copy of A!" << endl;
		exit (EXIT_FAILURE);
	}

	if (!testCUDAMatrixMulTranspose()) {
		cerr << "Matrix multiplication errornous." << endl;
		exit (EXIT_FAILURE);
	}

	exit (EXIT_SUCCESS);
}
