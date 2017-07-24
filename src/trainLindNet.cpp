#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <tclap/CmdLine.h>
#include <algorithm>
#include <yaml-cpp/yaml.h>
#include "MNISTDataset.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkParallel.h"
#include "NeuralNetworkDistributed.h"
#include "NeuralNetworkCUDA.h"

using namespace std;
using namespace TCLAP;

unsigned char KEY_ESC = 27;

int main(int argc, char* argv[]) {

	CmdLine parser("Train the LindNet with the MNIST database.");

	ValueArg<string> mnistPath("", "mnist",
			"Folder containing the MNIST files.", false, "", "path", parser);

	ValueArg<string> netDefinitionPath("n", "lindNet",
			"yaml file for saving the resulting net.", false, "lindNet.yaml",
			"path", parser);

	ValueArg<string> networkType("t", "networkType",
			"The neural network type (sequentiell, parallel, cuda).", false,
			"sequentiell", "type", parser);

	SwitchArg distributed("d", "distributed",
			"The neural network will be executed distributed.", parser, false);

	ValueArg<int> inputLayerNodes("", "inputNodes",
			"The amount of input nodes.", false, 28 * 28, "inputLayerNodes",
			parser);

	ValueArg<int> hiddenLayerNodes("", "hiddenNodes",
			"The amount of hidden nodes.", false, 20, "hiddenLayerNodes",
			parser);

	ValueArg<int> outputLayerNodes("", "outputNodes",
			"The amount of output nodes.", false, 10, "outputLayerNodes",
			parser);

	ValueArg<double> learningRate("l", "learningRate",
			"The learning rate of the neural network.", false, 0.2,
			"learningRate", parser);

	ValueArg<double> trainingErrorThreshold("", "trainingError",
			"The training error when the neural network should quit work if the value is reached.",
			false, 0.04, "trainingError", parser);

	ValueArg<double> maxDerivation("", "derivation",
			"The max derivation between to erros after two train samples where proceed.",
			false, 0.005, "maxDerivation", parser);

	try {
		parser.parse(argc, argv);
	} catch (ArgParseException const& e) {
		cerr << e.what() << endl;
		exit(EXIT_FAILURE);
	}

	// Read the training data.
	string imagePath = mnistPath.getValue() + "/train-images-idx3-ubyte";
	MNISTImageDataset trainingImages(imagePath);
	trainingImages.load();

	string labelPath = mnistPath.getValue() + "/train-labels-idx1-ubyte";
	MNISTLableDataset trainingLabels(labelPath);
	trainingLabels.load();

	// Default is sequentiell
	NeuralNetwork* lindNet = nullptr;

	string networkTypeSelection = networkType.getValue();
	if (networkTypeSelection.compare("parallel") == 0) {
		lindNet = new NeuralNetworkParallel(inputLayerNodes.getValue(),
				hiddenLayerNodes.getValue(), outputLayerNodes.getValue(),
				learningRate.getValue());
		cout << "Neural Network - Parallel" << endl;
	} else if (networkTypeSelection.compare("cuda") == 0) {
		lindNet = new NeuralNetworkCUDA(inputLayerNodes.getValue(),
				hiddenLayerNodes.getValue(), outputLayerNodes.getValue(),
				learningRate.getValue());
		cout << "Neural Network - CUDA" << endl;
	} else {
		lindNet = new NeuralNetwork(inputLayerNodes.getValue(),
				hiddenLayerNodes.getValue(), outputLayerNodes.getValue(),
				learningRate.getValue());
		cout << "Neural Network - Sequentiell" << endl;
	}

	if (distributed.isSet()) {
		lindNet = new NeuralNetworkDistributed(argc, argv, *lindNet);
		cout << "# !!! Distributed !!! #" << endl;
	}

	// Do some training.
	std::chrono::high_resolution_clock::time_point start =
			std::chrono::high_resolution_clock::now();
	lindNet->train(trainingImages, trainingLabels,
			trainingErrorThreshold.getValue(), maxDerivation.getValue());
	std::chrono::high_resolution_clock::time_point end =
			std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> sec = std::chrono::duration_cast<
			std::chrono::duration<double>>(end - start);
	cout << "Taining done in " << sec.count() << "s" << endl;

	// Save the trained net.
	lindNet->saveYAML(netDefinitionPath.getValue());

	delete lindNet;
	exit(EXIT_SUCCESS);
}
