#include "NeuralNetworkDistributed.h"

#define LOG_MASTER 1
#define LOG_SLAVE 0

using namespace std;

NeuralNetworkDistributed::NeuralNetworkDistributed(int _argc, char** _argv,
		NeuralNetwork& _nn) : argc(_argc), argv(_argv), nn(_nn) {
}

double* getWeightsByLayer(NeuralNetwork &nn, NeuralNetwork::LayerType type) {
	NeuralNetwork::Layer *layer = nn.getLayer(type);
	double *weights = new double[layer->nodes.size() * layer->nodes[0]->weights.size()];
	for(int i=0; i < layer->nodes.size(); i++) {
		NeuralNetwork::Layer::Node *n = layer->getNode(i);
		for(int w=0; w < n->weights.size(); w++) {
			weights[i*n->weights.size() + w] = n->weights.at(w);
		}
	}
	return weights;
}

double* getDeltaWeightsByLayer(NeuralNetwork &nn, NeuralNetwork::LayerType type, double* initWeights) {
	//cout << endl;
	//cout << "getDeltaWeightsByLayer: " << type << endl;
	NeuralNetwork::Layer *layer = nn.getLayer(type);
	double *weights = new double[layer->nodes.size() * layer->nodes[0]->weights.size()];
	for(int i=0; i < layer->nodes.size(); i++) {
		NeuralNetwork::Layer::Node *n = layer->getNode(i);
		for(int w=0; w < n->weights.size(); w++) {
			weights[i*n->weights.size() + w] = n->weights.at(w) - initWeights[i*n->weights.size() + w];
			//cout << "\t" << weights[i*n->weights.size() + w] << endl;
		}
	}
	delete[] initWeights;
	return weights;
}

double getGlobalError(int const globalSize, double* errorData, int const errorSize) {
	cout << "Sum global error: ";
	int globalError = 0;
	for(int i=0; i < errorSize; i++) {
		if(i > 0)
			cout << " + ";
		cout << errorData[i];
		globalError += errorData[i];
	}
	delete[] errorData;
	cout << " = " << (static_cast<double>(globalError) / static_cast<double>(errorSize)) << endl;
	return static_cast<double>(globalError) / static_cast<double>(errorSize);
}

void updateWeights(NeuralNetwork &nn, NeuralNetwork::LayerType type, double* deltaWeights, int const iterCount) {
	//cout << endl;
	//cout << "updateWeights: " << type << endl;
	NeuralNetwork::Layer *layer = nn.getLayer(type);
	for(int i=0; i < layer->nodes.size(); i++) {
		NeuralNetwork::Layer::Node *n = layer->getNode(i);
		for(int w=0; w < n->weights.size(); w++) {
			for(int itr=0; itr < iterCount; itr++) {
				//cout << "\t" << n->weights.at(w) << " | ";
				n->weights.at(w) += deltaWeights[i * n->weights.size() + itr * n->weights.size() + w];
				//cout << n->weights.at(w) << endl;
			}
		}
	}
	delete[] deltaWeights;
}

void setWeights(NeuralNetwork &nn, NeuralNetwork::LayerType type, double* deltaWeights) {
	NeuralNetwork::Layer *layer = nn.getLayer(type);
	for(int i=0; i < layer->nodes.size(); i++) {
		NeuralNetwork::Layer::Node *n = layer->getNode(i);
		for(int w=0; w < n->weights.size(); w++) {
			n->weights.at(w) = deltaWeights[i * n->weights.size() + w];
		}
	}
}

double logStart(int const rank, std::string const msg) {
	if(!LOG_MASTER && rank == 0)
		return 0;
	if(!LOG_SLAVE && rank > 0)
		return 0;

	if(rank == 0)
		cout << "MASTER: ";
	else
		cout << "SLAVE-" << rank << ": ";
	cout << msg;
	cout.flush();
	return MPI_Wtime();
}

void logEnd(int const rank, double const time) {
	if(!LOG_MASTER && rank == 0)
		return;
	if(!LOG_SLAVE && rank > 0)
		return;

	cout << "FINISHED! (Time=" << MPI_Wtime() - time << "s)" << endl;
	cout.flush();
}

double NeuralNetworkDistributed::train(MNISTImageDataset const& images,
		MNISTLableDataset const& labels,
		double const training_error_threshold,
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

	double time;

	int hiddenWeightsCount = nn.getLayer(HIDDEN)->nodes.size() * nn.getLayer(HIDDEN)->nodes[0]->weights.size();
	int outputWeightsCount = nn.getLayer(OUTPUT)->nodes.size() * nn.getLayer(OUTPUT)->nodes[0]->weights.size();

    if(curr_rank == 0) {
        // #################################################################################
        // Master
        // #################################################################################

    	// 1. Send (MPI_Scatter) training set and init data to slaves
    	time = logStart(curr_rank, "Send init data to slaves...");
    	imageCount = imageDataset->size();
    	imageSize = (*imageDataset)[0].rows * (*imageDataset)[0].cols;
		MPI_Bcast(&imageCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&imageSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
		logEnd(curr_rank, time);

    	time = logStart(curr_rank, "Send training setto slaves...");
    	uchar *imageArray = new uchar[imageCount*imageSize];
    	uint8_t *labelArray = new uint8_t[imageCount];
    	for(int i=0; i < imageDataset->size(); i++) {
    		std::copy((*imageDataset)[i].datastart, (*imageDataset)[i].dataend, &(imageArray[i]));
    		labelArray[i] = (*labelDataset)[i];
    	}
    	MPI_Bcast(&imageArray[0], imageCount * imageSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
		delete[] imageArray;
		MPI_Bcast(&labelArray[0], imageCount, MPI_UINT8_T, 0, MPI_COMM_WORLD);
		delete[] labelArray;
		logEnd(curr_rank, time);
    } else {
        // #################################################################################
        // Slave
        // #################################################################################

    	// 1. Receive (MPI_Scatter) training set and init data
    	//time = logStart(curr_rank, "Receive init data...");
		MPI_Bcast(&imageCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&imageSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
		//logEnd(curr_rank, time);

    	workSize = imageCount / (world_size - 1);
    	startWork = workSize * (curr_rank - 1);
    	endWork = workSize * curr_rank;

		// Receive images
    	time = logStart(curr_rank, "Receive training images...");
    	uchar *imageArray = new uchar[imageCount*imageSize];
    	MPI_Bcast(&imageArray[0], imageCount * imageSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    	int matrixSize = imageSize / imageSize;
    	vector<cv::Mat> imageMatrixes;
    	for(int i=startWork; i < endWork; i++)
    		imageMatrixes.push_back(cv::Mat(matrixSize, matrixSize, CV_8UC1, imageArray[i]));
    	imageDataset = new MNISTImageDataset(imageMatrixes);
		logEnd(curr_rank, time);

    	// Receive labels
    	time = logStart(curr_rank, "Receive training images...");
    	uint8_t *labelArray = new uint8_t[imageCount];
    	MPI_Bcast(&labelArray[0], imageCount, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    	vector<uint8_t> labelVector;
    	for(int i=startWork; i < endWork; i++)
    		labelVector.push_back(labelArray[i]);
    	labelDataset = new MNISTLableDataset(labelVector);
		logEnd(curr_rank, time);
    }

    cout << endl;
    logStart(curr_rank, "# Start Training-LOOP #");
    cout << endl;
    cout << endl;

	while(needsFurtherTraining) {
		double *weightsHidden;
		double *weightsOutput;
		double localError = 0;

		if(curr_rank == 0) {
			// 2. Send (MPI_Bcast) weights to slaves
			time = logStart(curr_rank, "Send hidden weights to slaves...");
			weightsHidden = getWeightsByLayer(nn, HIDDEN);
			MPI_Bcast(&weightsHidden[0], hiddenWeightsCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			delete[] weightsHidden;
			logEnd(curr_rank, time);

			time = logStart(curr_rank, "Send output weights to slaves...");
			weightsOutput = getWeightsByLayer(nn, OUTPUT);
			MPI_Bcast(&weightsOutput[0], outputWeightsCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			delete[] weightsOutput;
			logEnd(curr_rank, time);
		} else {
			// 2. Receive (MPI_Bcast) weights
			time = logStart(curr_rank, "Receive hidden weights and update...");
			weightsHidden = new double[hiddenWeightsCount];
			MPI_Bcast(&weightsHidden[0], hiddenWeightsCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			setWeights(nn, HIDDEN, weightsHidden);
			logEnd(curr_rank, time);

			time = logStart(curr_rank, "Receive output weights and update...");
			weightsOutput = new double[outputWeightsCount];
			MPI_Bcast(&weightsOutput[0], outputWeightsCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			setWeights(nn, OUTPUT, weightsOutput);
			logEnd(curr_rank, time);

	    	// 3. Perform training step
	    	time = logStart(curr_rank, "Execute training...");
	    	localError = nn.train(*imageDataset, *labelDataset, 100.0, 0.0);
			logEnd(curr_rank, time);
		}

		// 4a. (MPI_Gather) delta weights - HIDDEN
    	time = logStart(curr_rank, "Send/Receive hidden delta weights...");
		double *deltaWeightsHidden;
		if(curr_rank == 0)
			deltaWeightsHidden = new double[world_size * hiddenWeightsCount];
		else
			deltaWeightsHidden = getDeltaWeightsByLayer(nn, HIDDEN, weightsHidden);
		MPI_Gather(&deltaWeightsHidden[0], hiddenWeightsCount, MPI_DOUBLE, &deltaWeightsHidden[0], hiddenWeightsCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if(curr_rank == 0)
			updateWeights(nn, HIDDEN, deltaWeightsHidden, world_size);
		else
			delete[] deltaWeightsHidden;
		logEnd(curr_rank, time);

		// 4b. (MPI_Gather) delta weights - OUTPUT
		time = logStart(curr_rank, "Send/Receive output delta weights...");
		double *deltaWeightsOutput;
		if(curr_rank == 0)
			deltaWeightsOutput = new double[world_size * outputWeightsCount];
		else
			deltaWeightsOutput = getDeltaWeightsByLayer(nn, OUTPUT, weightsOutput);
		MPI_Gather(&deltaWeightsOutput[0], outputWeightsCount, MPI_DOUBLE, &deltaWeightsOutput[0], outputWeightsCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if(curr_rank == 0)
			updateWeights(nn, OUTPUT, deltaWeightsOutput, world_size);
		else
			delete[] deltaWeightsOutput;
		logEnd(curr_rank, time);

		// 4c. (MPI_Reduce) local errors
		time = logStart(curr_rank, "Send/Receive local error...");
		double errorSum;
		MPI_Reduce(&localError, &errorSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		logEnd(curr_rank, time);

		// 4. Check whether stop or repeat (back to 2.)
    	if(curr_rank == 0) {
			time = logStart(curr_rank, "Check whether stop or repeat...");
			errorSum /= world_size - 1;

			if (errorSum < error) {
				error = errorSum;
			}

			if(errorSum < training_error_threshold || errorSum > error + max_derivation) {
				needsFurtherTraining = false;
			}
			logEnd(curr_rank, time);

			cout << "MASTER: Error: " << errorSum * 100.0 << "%" << endl;
    	}

		// 5. Wait for command from master to go on or exit
    	time = logStart(curr_rank, "Notify processes to go on or exit...");
		MPI_Bcast(&needsFurtherTraining, 1, MPI_INT, 0, MPI_COMM_WORLD);
		logEnd(curr_rank, time);
	}

	MPI_Finalize();
}

bool NeuralNetworkDistributed::saveYAML(string const& path) {
	ofstream fout(path);

	fout << nn;

	return true;
}
