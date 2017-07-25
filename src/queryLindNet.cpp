#include <string>
#include <tclap/CmdLine.h>
#include "NeuralNetwork.h"
#include "MNISTDataset.h"
#include "utils/MNISTStats.h"

using namespace std;
using namespace TCLAP;

int main(int argc, char* argv[]) {

	CmdLine parser("Use an already trained net to classify given images.");

	ValueArg<string> netDefinitionPath("n", "lindNet",
			"File containing a LindNet definition.", true, "", "path", parser);

	ValueArg<string> mnistPath("", "mnist",
			"Folder containing the MNIST files.", true, "", "path", parser);

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

	NeuralNetwork net = NeuralNetwork::LoadYAML(netDefinitionPath.getValue());

	size_t const showProgressEach = 1000;
	MNISTStats mnistStats;

	int every_ten_percent = testingImages.size() * 5 / 100;
	int error = 0;
	for (size_t i = 0; i < testingImages.size(); ++i) {

		net.feedInput(testingImages[i]);
		net.feedForward();
		int const classification = net.getNetworkClassification();
		if (classification != testingLabels[i]) {
			++error;
		}

		mnistStats.displayTestingProgress(testingImages.size(), testingImages.size(), error, 5, 5);
		if(i % every_ten_percent == 0) {
			mnistStats.displayImage(testingImages[i], testingLabels[i], classification, 7,6);
		}
	}

	exit(EXIT_SUCCESS);
}
