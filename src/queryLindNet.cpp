#include <string>
#include <tclap/CmdLine.h>
#include "NeuralNetwork.h"
#include "NeuralNetworkParallel.h"
#include "NeuralNetworkCUDA.h"
#include "MNISTDataset.h"

using namespace std;
using namespace TCLAP;

int main(int argc, char* argv[]) {

	CmdLine parser("Use an already trained net to classify given images.");

	ValueArg<string> netDefinitionPath("n", "lindNet",
			"File containing a LindNet definition.", true, "", "path", parser);

	ValueArg<string> mnistPath("", "mnist",
			"Folder containing the MNIST files.", true, "", "path", parser);

	ValueArg<string> networkType("t", "networkType",
			"The neural network type (sequentiell, parallel, cuda).", false, "sequentiell", "type", parser);

	try {
		parser.parse(argc, argv);
	} catch (ArgParseException const& e) {
		cerr << e.what() << endl;
		exit(EXIT_FAILURE);
	}

	// Read the training data.
	string imagePath = mnistPath.getValue() + "/t10k-images-idx3-ubyte";
	MNISTImageDataset testingImages(imagePath);
	testingImages.load();

	string labelPath = mnistPath.getValue() + "/t10k-labels-idx1-ubyte";
	MNISTLableDataset testingLabels(labelPath);
	testingLabels.load();

	// Default is sequentiell
	NeuralNetwork* net = nullptr;

	string networkTypeSelection = networkType.getValue();
	if(networkTypeSelection.compare("parallel") == 0) {
		net = new NeuralNetworkParallel(0, 0, 0, 0.0);
		cout << "Neural Network - Parallel" << endl;
	} else if (networkTypeSelection.compare("cuda") == 0) {
		net = new NeuralNetworkCUDA(0, 0, 0, 0.0);
		cout << "Neural Network - CUDA" << endl;
	} else {
		net = new NeuralNetwork(0, 0, 0, 0.0);
		cout << "Neural Network - Sequentiell" << endl;
	}

	net->loadYAML(netDefinitionPath.getValue());

	size_t const showProgressEach = 1000;

	int every_ten_percent = testingImages.size() / 10;
	size_t correct = 0;
	for (size_t i = 0; i < testingImages.size(); ++i) {

		net->feedInput(testingImages[i]);
		net->feedForward();
		int const classification = net->getNetworkClassification();
		if (classification == testingLabels[i]) {
			++correct;
		}

		if ((i % every_ten_percent) == 0)
			cout << "x"; cout.flush();
	}
	cout << endl;

	double const classificationRate = static_cast<double>(correct) / static_cast<double>(testingImages.size());

	cout << "Test finished." << endl;
	cout << "Correct classifications: " << classificationRate * 100.0 << "%" << endl;

	delete net;
	exit(EXIT_SUCCESS);
}
