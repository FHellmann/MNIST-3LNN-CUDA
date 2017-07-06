#include <string>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <tclap/CmdLine.h>
#include <algorithm>
#include <yaml-cpp/yaml.h>
#include "MNISTDataset.h"
#include "NeuralNetwork.hpp"

using namespace std;
using namespace TCLAP;

unsigned char KEY_ESC = 27;

void trainNetwork(NeuralNetwork&, MNISTImageDataset const&,
		MNISTLableDataset const&);

int main(int argc, char* argv[]) {

	CmdLine parser("Train the LindNet with the MNIST database.");

	ValueArg<string> mnistPath("", "mnist",
			"Folder containing the MNIST files.", true, "", "path", parser);

	ValueArg<string> netDefinitionPath("", "lindNet",
			"yaml file for saving the resulting net.", false, "lindNet.yaml",
			"path", parser);

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
	NeuralNetwork lindNet(28*28, 20, 10, 0.5);

	// Do some training.
	trainNetwork(lindNet, trainingImages, trainingLabels);

	// Save the trained net.
	lindNet.saveYAML(netDefinitionPath.getValue());

	exit(EXIT_SUCCESS);
}

void trainNetwork(NeuralNetwork& net, MNISTImageDataset const& images,
		MNISTLableDataset const& labels) {

	int errCount = 0;

	size_t showProgressEach = 1000;

	// Loop through all images in the file
	for (size_t imgCount = 0; imgCount < images.size(); imgCount++) {

		// Convert the MNIST image to a standardized vector format and feed into the network
		net.feedInput(images[imgCount]);

		// Feed forward all layers (from input to hidden to output) calculating all nodes' output
		net.feedForward();

		// Back propagate the error and adjust weights in all layers accordingly
		net.backPropagate(labels[imgCount]);

		// Classify image by choosing output cell with highest output
		int classification = net.getNetworkClassification();
		if (classification != labels[imgCount])
			errCount++;

		// Display progress during training
		//displayTrainingProgress(imgCount, errCount, 80);
		//displayImage(&img, lbl, classification, 7,6);
		if ((imgCount % showProgressEach) == 0)
			cout << "x"; cout.flush();
	}

	cout << endl;
}
