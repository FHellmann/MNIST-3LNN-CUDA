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
#include "MNISTDataset.h"

using namespace std;
using namespace TCLAP;

int main (int argc, char* argv[]) {

	CmdLine parser("Foobar info text");

	ValueArg<string> mnistPath (
			"",
			"mnist",
			"Folder containing the MNIST files.",
			true,
			"",
			"path",
			parser);

	try {
		parser.parse(argc, argv);
	} catch (ArgParseException const& e) {
		cerr << e.what() << endl;
		exit(EXIT_FAILURE);
	}

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

	MNISTLableDataset::iterator it = trainingLabels.begin();
	for (cv::Mat img : trainingImages) {
		cv::imshow("Hand writing", img);
		cout << "Lable: " << (int)*(it++) << endl;
		cv::waitKey(0);
	}

	exit (EXIT_SUCCESS);
}
