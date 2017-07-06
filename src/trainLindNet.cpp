/*
 * main.cpp
 *
 *  Created on: Jul 4, 2017
 *      Author: Stefan
 */
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
#include "NeuralNetworkIO.hpp"

using namespace std;
using namespace TCLAP;

unsigned char KEY_ESC = 27;

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

// TODO: Do some training.
	cout << "Press ESC or q to quit." << endl;
	MNISTLableDataset::iterator it = trainingLabels.begin();
	for (cv::Mat img : trainingImages) {
		cv::imshow("Hand writing", img);
		cout << "Lable: " << (int) *(it++) << endl;
		unsigned char key = cv::waitKey(0);
		if (key == KEY_ESC || key == 'q') {
			break;
		}
	}

	// Save the trained net.
	NeuralNetwork lindNet(4, 20, 10, 0.5);
	cout << lindNet << endl;
	saveNet(netDefinitionPath.getValue(), lindNet);

	exit(EXIT_SUCCESS);
}
