#include "NeuralNetworkDistributed.h"

using namespace std;

NeuralNetworkDistributed::NeuralNetworkDistributed(int _argc, char** _argv,
		std::string _netDefinitionPath, NeuralNetwork& _nn) :
		argc(_argc), argv(_argv), netDefinitionPath(_netDefinitionPath), nn(_nn) {
}

/**
 * Get the delta weights of a complete layer.
 *
 * @param nn The neural network to get the weights of.
 * @param type The layer to get the weights from.
 */
double* getDeltaWeightsByLayer(NeuralNetwork &nn, NeuralNetwork::LayerType type,
		double* initWeights = nullptr) {
	NeuralNetwork::Layer *layer = nn.getLayer(type);
	double *weights = new double[layer->nodes.size()
			* layer->nodes[0]->weights.size()];
	for (int i = 0; i < layer->nodes.size(); i++) {
		NeuralNetwork::Layer::Node *n = layer->getNode(i);
		for (int w = 0; w < n->weights.size(); w++) {
			int index = i * n->weights.size() + w;
			weights[index] = n->weights.at(w)
					- (initWeights ? initWeights[index] : 0);
		}
	}
	if (initWeights)
		delete[] initWeights;
	return weights;
}

/**
 * Update all weights of the specified layer with the delta weights.
 *
 * @param nn The neural network which needs to update its weights.
 * @param type The layer with all the weights.
 * @param deltaWeights The weights to add.
 * @param iterCount The amount of delta weights for each weight.
 */
void updateWeights(NeuralNetwork &nn, NeuralNetwork::LayerType type,
		double* deltaWeights, int const iterCount = 1) {
	int workerWeightCount = nn.getLayer(type)->nodes.size()
			* nn.getLayer(type)->nodes[0]->weights.size();
	NeuralNetwork::Layer *layer = nn.getLayer(type);
	for (int i = 0; i < layer->nodes.size(); i++) {
		NeuralNetwork::Layer::Node *n = layer->getNode(i);
		for (int w = 0; w < n->weights.size(); w++) {
			for (int itr = 1; itr < iterCount; itr++) {
				n->weights.at(w) += deltaWeights[itr * workerWeightCount
						+ i * n->weights.size() + w];
			}
		}
	}
}

double NeuralNetworkDistributed::train(MNISTImageDataset const& images,
		MNISTLableDataset const& labels, double const training_error_threshold,
		double const max_derivation) {
	MPI_Init(&argc, &argv);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int curr_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &curr_rank);

	bool needsFurtherTraining = true;
	double error = std::numeric_limits<double>::max();

	MNISTImageDataset const *imageDataset = &images;
	MNISTLableDataset const *labelDataset = &labels;

	int imageCount;
	int imageSize;

	int workSize;
	int startWork;
	int endWork;

	chrono::high_resolution_clock::time_point time;

	int hiddenWeightsCount = nn.getLayer(HIDDEN)->nodes.size()
			* nn.getLayer(HIDDEN)->nodes[0]->weights.size();
	int outputWeightsCount = nn.getLayer(OUTPUT)->nodes.size()
			* nn.getLayer(OUTPUT)->nodes[0]->weights.size();

	if (curr_rank == 0) {
		// #################################################################################
		// Master
		// #################################################################################

		// 1. Send (MPI_Scatter) training set and init data to slaves
		time = logStart("Send init data to slaves...");
		imageCount = imageDataset->size();
		imageSize = (*imageDataset)[0].rows * (*imageDataset)[0].cols;
		MPI_Bcast(&imageCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&imageSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
		logEnd(time);

		time = logStart("Send training set to slaves...");
		uchar *imageArray = new uchar[imageCount * imageSize];
		uint8_t *labelArray = new uint8_t[imageCount];
		for (int i = 0; i < imageDataset->size(); i++) {
			std::copy((*imageDataset)[i].datastart, (*imageDataset)[i].dataend,
					&(imageArray[i * imageSize]));
			labelArray[i] = (*labelDataset)[i];
		}
		MPI_Bcast(&imageArray[0], imageCount * imageSize, MPI_UNSIGNED_CHAR, 0,
		MPI_COMM_WORLD);
		delete[] imageArray;
		MPI_Bcast(&labelArray[0], imageCount, MPI_UINT8_T, 0, MPI_COMM_WORLD);
		delete[] labelArray;
		logEnd(time);
	} else {
		// #################################################################################
		// Slave
		// #################################################################################

		// 1. Receive (MPI_Scatter) training set and init data
		time = logStart("Receive init data...", curr_rank);
		MPI_Bcast(&imageCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&imageSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
		logEnd(time, curr_rank);

		workSize = imageCount / (world_size - 1);
		startWork = workSize * (curr_rank - 1);
		endWork = workSize * curr_rank;

		// 1a. Receive images
		time = logStart("Receive training images...", curr_rank);
		uchar *imageArray = new uchar[imageCount * imageSize];
		MPI_Bcast(&imageArray[0], imageCount * imageSize, MPI_UNSIGNED_CHAR, 0,
		MPI_COMM_WORLD);
		int matrixSize = sqrt(imageSize);
		vector<cv::Mat> imageMatrixes;
		for (int i = startWork; i < endWork; i++) {
			uchar *image = new uchar[imageSize];
			std::copy(&(imageArray[i * imageSize]),
					&(imageArray[(i + 1) * imageSize]), &(image[0]));
			imageMatrixes.push_back(
					cv::Mat(matrixSize, matrixSize, CV_8UC1, image));
		}
		imageDataset = new MNISTImageDataset(imageMatrixes);
		delete[] imageArray;
		logEnd(time, curr_rank);

		// 1b. Receive labels
		time = logStart("Receive training images...", curr_rank);
		uint8_t *labelArray = new uint8_t[imageCount];
		MPI_Bcast(&labelArray[0], imageCount, MPI_UINT8_T, 0, MPI_COMM_WORLD);
		vector<uint8_t> labelVector;
		for (int i = startWork; i < endWork; i++)
			labelVector.push_back(labelArray[i]);
		labelDataset = new MNISTLableDataset(labelVector);
		delete[] labelArray;
		logEnd(time, curr_rank);
	}

	log("# Start Training-LOOP #", curr_rank);

	while (needsFurtherTraining) {
		double *weightsHidden;
		double *weightsOutput;
		double localError = 0;

		// 2a. Send/Receive (MPI_Bcast) hidden weights
		//time = logStart("Send/Receive hidden weights to slaves...", curr_rank);
		if (curr_rank == 0)
			weightsHidden = getDeltaWeightsByLayer(nn, HIDDEN);
		else
			weightsHidden = new double[hiddenWeightsCount];
		MPI_Bcast(&weightsHidden[0], hiddenWeightsCount, MPI_DOUBLE, 0,
		MPI_COMM_WORLD);
		if (curr_rank == 0)
			delete[] weightsHidden;
		else
			updateWeights(nn, HIDDEN, weightsHidden);
		//logEnd(time, curr_rank);

		// 2b. Send/Receive (MPI_Bcast) output weights
		//time = logStart("Send/Receive output weights and update...", curr_rank);
		if (curr_rank == 0)
			weightsOutput = getDeltaWeightsByLayer(nn, OUTPUT);
		else
			weightsOutput = new double[outputWeightsCount];
		MPI_Bcast(&weightsOutput[0], outputWeightsCount, MPI_DOUBLE, 0,
		MPI_COMM_WORLD);
		if (curr_rank == 0)
			delete[] weightsOutput;
		else
			updateWeights(nn, OUTPUT, weightsOutput);
		//logEnd(time, curr_rank);

		if (curr_rank > 0) {
			// 3. Perform training step - Slaves ONLY
			time = logStart("Execute training...", curr_rank);
			localError = nn.train(*imageDataset, *labelDataset, 100.0, 0.0);
			logEnd(time, curr_rank);
		}

		// 4a. (MPI_Gather) delta weights - HIDDEN
		//time = logStart("Send/Receive hidden delta weights...", curr_rank);
		double *deltaWeightsHidden;
		if (curr_rank == 0)
			deltaWeightsHidden = new double[world_size * hiddenWeightsCount];
		else
			deltaWeightsHidden = getDeltaWeightsByLayer(nn, HIDDEN,
					weightsHidden);
		MPI_Gather(&deltaWeightsHidden[0], hiddenWeightsCount, MPI_DOUBLE,
				&deltaWeightsHidden[0], hiddenWeightsCount, MPI_DOUBLE, 0,
				MPI_COMM_WORLD);
		if (curr_rank == 0)
			updateWeights(nn, HIDDEN, deltaWeightsHidden, world_size);
		else
			delete[] deltaWeightsHidden;
		//logEnd(time, curr_rank);

		// 4b. (MPI_Gather) delta weights - OUTPUT
		//time = logStart("Send/Receive output delta weights...", curr_rank);
		double *deltaWeightsOutput;
		if (curr_rank == 0)
			deltaWeightsOutput = new double[world_size * outputWeightsCount];
		else
			deltaWeightsOutput = getDeltaWeightsByLayer(nn, OUTPUT,
					weightsOutput);
		MPI_Gather(&deltaWeightsOutput[0], outputWeightsCount, MPI_DOUBLE,
				&deltaWeightsOutput[0], outputWeightsCount, MPI_DOUBLE, 0,
				MPI_COMM_WORLD);
		if (curr_rank == 0)
			updateWeights(nn, OUTPUT, deltaWeightsOutput, world_size);
		else
			delete[] deltaWeightsOutput;
		//logEnd(time, curr_rank);

		// 4c. (MPI_Reduce) local errors
		//time = logStart("Send/Receive local error...", curr_rank);
		double errorSum;
		MPI_Reduce(&localError, &errorSum, 1, MPI_DOUBLE, MPI_SUM, 0,
		MPI_COMM_WORLD);
		//logEnd(time, curr_rank);

		if (curr_rank == 0) {
			// 4. Check whether stop or repeat (back to 2.) - Master ONLY
			//time = logStart("Check whether stop or repeat...");
			errorSum /= world_size - 1; // -1 to remove the master

			if (errorSum < error) {
				error = errorSum;
			}

			if (errorSum < training_error_threshold
					|| errorSum > error + max_derivation) {
				needsFurtherTraining = false;
			}
			//logEnd(time, curr_rank);

			log("Error: " + to_string(errorSum * 100.0) + "%");
		}

		// 5. Wait for command from master to go on or exit
		//time = logStart("Notify processes to go on or exit...", curr_rank);
		MPI_Bcast(&needsFurtherTraining, 1, MPI_INT, 0, MPI_COMM_WORLD);
		//logEnd(time, curr_rank);
	}

	MPI_Finalize();

	if(curr_rank == 0) {
		saveYAML(netDefinitionPath);
	}
}

bool NeuralNetworkDistributed::saveYAML(string const& path) {
	ofstream fout(path);

	fout << nn;

	return true;
}
