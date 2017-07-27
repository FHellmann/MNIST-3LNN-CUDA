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
#include "NeuralNetworkCUDA.h"
#include "NeuralNetworkEigen.h"

using namespace std;
using namespace TCLAP;

unsigned char KEY_ESC = 27;

int main(int argc, char* argv[]) {

	CmdLine parser("Train the LindNet with the MNIST database.");

	ValueArg<string> mnistPath("", "mnist",
			"Folder containing the MNIST files.", true, "", "path", parser);

	ValueArg<string> netDefinitionPath("n", "lindNet",
			"yaml file for saving the resulting net.", false, "lindNet.yaml",
			"path", parser);

	ValueArg<string> networkType("t", "networkType",
			"The neural network type (sequentiell, parallel, cuda, eigen).", false, "sequentiell", "type", parser);

	ValueArg<size_t> iterations("", "iterations",
			"Maximum number of iterations over the whole data set. Currenlty only supported for the cuda and eigen versions.",
			false, 1, "positive integer", parser);

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

//	// WTF happens here?!
//	cout << "Foobar begin" << endl;
//	cv::Mat const& foobar = 5;
//	cout << "Foobar ongoing" << endl;
//	cv::imshow("Hand writing", foobar);
//	cout << "Foobar end" << endl;
//	cv::waitKey(0);

	int inputLayerNodes = 28*28;
	int hiddenLayerNodes = 28;
	int outputLayerNodes = 10;
	double learningRate = 0.2;

	// Default is sequentiell
	NeuralNetwork* lindNet = nullptr;

	string networkTypeSelection = networkType.getValue();
	if(networkTypeSelection.compare("parallel") == 0) {
		lindNet = new NeuralNetworkParallel(inputLayerNodes, hiddenLayerNodes, outputLayerNodes, learningRate);
		cout << "Neural Network - Parallel" << endl;
	} else if (networkTypeSelection.compare("cuda") == 0) {
		lindNet = new NeuralNetworkCUDA(inputLayerNodes, hiddenLayerNodes, outputLayerNodes, learningRate, iterations.getValue());
		cout << "Neural Network - CUDA" << endl;
	} else if (networkTypeSelection.compare("eigen") == 0) {
		lindNet = new NeuralNetworkEigen(inputLayerNodes, hiddenLayerNodes, outputLayerNodes, learningRate, iterations.getValue());
		cout << "Neural Network - Eigen" << endl;
	} else {
		lindNet = new NeuralNetwork(inputLayerNodes, hiddenLayerNodes, outputLayerNodes, learningRate);
		cout << "Neural Network - Sequentiell" << endl;
	}

	// Do some training.
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	lindNet->train(trainingImages, trainingLabels, 0.06, 0.005);
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
	cout << "Taining done in " << sec.count() << "s" << endl;

	// Save the trained net.
	lindNet->saveYAML(netDefinitionPath.getValue());

	delete lindNet;
	exit(EXIT_SUCCESS);
}
