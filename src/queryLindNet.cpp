#include <string>
#include <tclap/CmdLine.h>
#include "NeuralNetwork.hpp"

using namespace std;
using namespace TCLAP;

int main(int argc, char* argv[]) {

	CmdLine parser("Use an already trained net to classify given images.");

	ValueArg<string> netDefinitionPath("n", "lindNet",
			"File containing a LindNet definition.", true, "", "path", parser);

	try {
		parser.parse(argc, argv);
	} catch (ArgParseException const& e) {
		cerr << e.what() << endl;
		exit(EXIT_FAILURE);
	}

	NeuralNetwork net = NeuralNetwork::LoadYAML(netDefinitionPath.getValue());

	cout << net << endl;

	exit(EXIT_SUCCESS);
}
