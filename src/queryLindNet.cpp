/*
 * main.cpp
 *
 *  Created on: Jul 4, 2017
 *      Author: Stefan
 */
#include <string>
#include <tclap/CmdLine.h>
#include "3lnn.h"
#include "3lnn_io.h"

using namespace std;
using namespace TCLAP;

int main (int argc, char* argv[]) {

	CmdLine parser("Use an already trained net to classify given images.");

	ValueArg<string> netDefinitionPath (
			"n",
			"lindNet",
			"File containing a LindNet definition.",
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

	Network* net = loadNet(netDefinitionPath.getValue());

	if (net) {
		cout << *net << endl;
	} else {
		cerr << "Net '" << netDefinitionPath.getValue() << "' could not be loaded." << endl;
		exit (EXIT_FAILURE);
	}

	if (net) {
		delete net;
		net = nullptr;
	}

	exit (EXIT_SUCCESS);
}
