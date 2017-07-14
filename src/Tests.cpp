/*
 * Tests.cpp
 *
 *  Created on: 14.07.2017
 *      Author: Stefan
 */
#include "NeuralNetworkParallel.h"
#include <iostream>

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

int main(int argc, char* argv[])
{
	NeuralNetworkParallel A(4, 2, 17, 0.2);
	NeuralNetworkParallel B(A);

	if (ensureDeepCopy(A, B) == false)
	{
		std::cerr << "B is not a deep copy of A!" << std::endl;
		exit (EXIT_FAILURE);
	}

	exit (EXIT_SUCCESS);
}
