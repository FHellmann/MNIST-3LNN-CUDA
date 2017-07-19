#include "NeuralNetworkDistributed.h"

using namespace std;

NeuralNetworkDistributed::NeuralNetworkDistributed(int _argc, char** _argv,
		NeuralNetwork& _nn) : argc(_argc), argv(_argv), nn(_nn) {
}

/*
void mergeNeuralNetworks(NeuralNetworkDistributed& omp_in, NeuralNetworkDistributed& omp_out, NeuralNetworkDistributed* reset) {
	for(int l=0; l < omp_in.layers.size(); l++) {
		NeuralNetwork::Layer* layerIn = omp_in.layers.at(l);
		NeuralNetwork::Layer* layerOut = omp_out.layers.at(l);
		NeuralNetwork::Layer* layerReset = reset->layers.at(l);
		for(int n=0; n < layerIn->nodes.size(); n++) {
			NeuralNetwork::Layer::Node* nodeIn = layerIn->nodes.at(n);
			NeuralNetwork::Layer::Node* nodeOut = layerOut->nodes.at(n);
			NeuralNetwork::Layer::Node* nodeReset = layerReset->nodes.at(n);
			for(int w=0; w < nodeIn->weights.size(); w++) {
				nodeOut->weights.at(w) += nodeIn->weights.at(w) - nodeReset->weights.at(w);
			}
		}
	}
}
*/

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

double getGlobalError(int const globalSize, double* errorData, int const errorSize) {
	cout << "Sum global error: ";
	int globalError = 0;
	for(int i=0; i < errorSize; i++) {
		cout << errorData[i] << " + ";
		globalError += errorData[i];
	}
	delete[] errorData;
	cout << "= " << (static_cast<double>(globalError) / static_cast<double>(errorSize)) << endl;
	return static_cast<double>(globalError) / static_cast<double>(errorSize);
}

void updateWeights(NeuralNetwork &nn, NeuralNetwork::LayerType type, double* deltaWeights, int const iterCount) {
	NeuralNetwork::Layer *layer = nn.getLayer(type);
	for(int i=0; i < layer->nodes.size(); i++) {
		NeuralNetwork::Layer::Node *n = layer->getNode(i);
		for(int w=0; w < n->weights.size(); w++) {
			for(int itr=0; itr < iterCount; itr++) {
				n->weights.at(w) += deltaWeights[itr * n->weights.size()];
			}
		}
	}
	delete[] deltaWeights;
}

void setWeights(NeuralNetwork &nn, NeuralNetwork::LayerType type, double* deltaWeights, int const iterCount) {
	NeuralNetwork::Layer *layer = nn.getLayer(type);
	for(int i=0; i < layer->nodes.size(); i++) {
		NeuralNetwork::Layer::Node *n = layer->getNode(i);
		for(int w=0; w < n->weights.size(); w++) {
			for(int itr=0; itr < iterCount; itr++) {
				n->weights.at(w) = deltaWeights[itr * n->weights.size()];
			}
		}
	}
	delete[] deltaWeights;
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

    MNISTImageDataset const *imageDataset = &images;
    MNISTLableDataset const *labelDataset = &labels;

	int hiddenWeightsCount = nn.getLayer(HIDDEN)->nodes.size() * nn.getLayer(HIDDEN)->nodes[0]->weights.size();
	int outputWeightsCount = nn.getLayer(OUTPUT)->nodes.size() * nn.getLayer(OUTPUT)->nodes[0]->weights.size();

    if(curr_rank == 0) {
        // #################################################################################
        // Master
        // #################################################################################

    	bool needsFurtherTraining = true;
    	double error = std::numeric_limits<double>::max();

    	// 1. Send (MPI_Scatter) training set and init data to slaves
    	cout << "Master: Send training set and init data to slaves...";

    	int imageCount = imageDataset->size();
    	int imageSize = (*imageDataset)[0].rows * (*imageDataset)[0].cols;
		MPI_Bcast(&imageCount, 1, MPI_INT, curr_rank, MPI_COMM_WORLD);
		MPI_Bcast(&imageSize, 1, MPI_INT, curr_rank, MPI_COMM_WORLD);

    	uchar *imageArray = new uchar[imageCount*imageSize];
    	uint8_t *labelArray = new uint8_t[imageCount];
    	for(int i=0; i < imageDataset->size(); i++) {
    		std::copy((*imageDataset)[i].datastart, (*imageDataset)[i].dataend, &(imageArray[i]));
    		labelArray[i] = (*labelDataset)[i];
    	}
    	MPI_Bcast(&imageArray[0], imageCount * imageSize, MPI_UNSIGNED_CHAR, curr_rank, MPI_COMM_WORLD);
		delete[] imageArray;
		MPI_Bcast(&labelArray[0], imageCount, MPI_UINT8_T, curr_rank, MPI_COMM_WORLD);
		delete[] labelArray;

		cout << "FINISHED!" << endl;

    	// 2. Receive (MPI_Gather) initialization information from slaves
    	cout << "Master: Receive initialization information from slaves...";

    	int *ready = new int[world_size];
    	MPI_Gather(NULL, 0, MPI_INT, &ready[0], 1, MPI_INT, curr_rank, MPI_COMM_WORLD);
		delete[] ready;

		cout << "FINISHED!" << endl;

		while(needsFurtherTraining) {
			// 3. Send (MPI_Bcast) weights to slaves
	    	cout << "Master: Send weights to slaves...";
			double *weightsHidden = getWeightsByLayer(nn, HIDDEN);
			MPI_Bcast(&weightsHidden[0], hiddenWeightsCount, MPI_DOUBLE, curr_rank, MPI_COMM_WORLD);
			delete[] weightsHidden;

			double *weightsOutput = getWeightsByLayer(nn, OUTPUT);
			MPI_Bcast(&weightsOutput[0], outputWeightsCount, MPI_DOUBLE, curr_rank, MPI_COMM_WORLD);
			delete[] weightsOutput;
			cout << "FINISHED!" << endl;

			// 4. Receive (MPI_Gather) all delta weights from slaves
	    	cout << "Master: Receive all delta weights from slaves...";
			//double *deltaWeightsHidden = new double[(world_size-1)*hiddenWeightsCount];
			//MPI_Gather(NULL, 0, MPI_DOUBLE, &deltaWeightsHidden[0], (world_size-1)*hiddenWeightsCount, MPI_DOUBLE, curr_rank, MPI_COMM_WORLD);
	    	for(int worker=curr_rank; worker < world_size; worker++) {
	    		double *deltaWeightsHidden = new double[hiddenWeightsCount];
	    		MPI_Recv(&deltaWeightsHidden[0], hiddenWeightsCount, MPI_DOUBLE, worker, 2, MPI_COMM_WORLD, NULL);
				updateWeights(nn, HIDDEN, deltaWeightsHidden, world_size - 1);
	    	}

			//double *deltaWeightsOutput = new double[(world_size-1)*outputWeightsCount];
			//MPI_Gather(NULL, 0, MPI_DOUBLE, &deltaWeightsOutput[0], (world_size-1)*outputWeightsCount, MPI_DOUBLE, curr_rank, MPI_COMM_WORLD);
	    	for(int worker=curr_rank; worker < world_size; worker++) {
	    		double *deltaWeightsOutput = new double[outputWeightsCount];
	    		MPI_Recv(&deltaWeightsOutput[0], outputWeightsCount, MPI_DOUBLE, worker, 3, MPI_COMM_WORLD, NULL);
				updateWeights(nn, HIDDEN, deltaWeightsOutput, world_size - 1);
	    	}

			double newError;
			MPI_Reduce(NULL, &error, world_size - 1, MPI_DOUBLE, MPI_SUM, curr_rank, MPI_COMM_WORLD);
			newError /= world_size - 1;

			cout << "FINISHED!" << endl;

			// 5. Check whether stop or repeat (back to 3.)
	    	cout << "Master: Check whether stop or repeat...";
			//updateWeights(nn, HIDDEN, deltaWeightsHidden, world_size - 1);
			//updateWeights(nn, OUTPUT, deltaWeightsOutput, world_size - 1);

			if (newError < error) {
				error = newError;
			}

			if(newError < training_error_threshold || newError > error + max_derivation) {
				needsFurtherTraining = false;
			}
			cout << "FINISHED!" << endl;

			// 6. Notify slaves to go on or exit
	    	cout << "Master: Notify slaves to go on or exit...";
			MPI_Bcast(&needsFurtherTraining, 1, MPI_INT, curr_rank, MPI_COMM_WORLD);
			cout << "FINISHED!" << endl;

			cout << "Master: Error: " << newError * 100.0 << "%" << endl;
		}
    } else {
        // #################################################################################
        // Slave
        // #################################################################################

    	bool needsFurtherTraining = true;

    	// 1. Receive (MPI_Scatter) training set and init data
    	//cout << "Slave-" << curr_rank << ": Receive training set and init data...";
    	int imageCount;
    	int imageSize;
		MPI_Bcast(&imageCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&imageSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
		//cout << "(imageCount=" << imageCount << ", imageSize=" << imageSize << ", ";

		int workSize = imageCount / (world_size - 1);
		int startWork = workSize * (curr_rank - 1);
		int endWork = workSize * curr_rank;

		//cout << "workSize=" << workSize << ", startWork=" << startWork << ", ";

		// Receive images
    	uchar *imageArray = new uchar[imageCount*imageSize];
    	MPI_Bcast(&imageArray[0], imageCount * imageSize, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    	int matrixSize = imageSize / imageSize;
    	vector<cv::Mat> imageMatrixes;
    	for(int i=startWork; i < endWork; i++)
    		imageMatrixes.push_back(cv::Mat(matrixSize, matrixSize, CV_8UC1, imageArray[i]));
    	imageDataset = new MNISTImageDataset(imageMatrixes);
    	// Receive labels
    	uint8_t *labelArray = new uint8_t[imageCount];
    	MPI_Bcast(&labelArray[0], imageCount, MPI_UINT8_T, 0, MPI_COMM_WORLD);
    	vector<uint8_t> labelVector;
    	for(int i=startWork; i < endWork; i++)
    		labelVector.push_back(labelArray[i]);
    	labelDataset = new MNISTLableDataset(labelVector);

		//cout << "FINISHED!" << endl;

    	// 2. Send (MPI_Gather) initialization finished
    	//cout << "Slave-" << curr_rank << ": Send initialization finished...";
    	bool finished = true;
    	MPI_Gather(&finished, 1, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
    	//cout << "FINISHED!" << endl;

    	while(needsFurtherTraining) {
    		// 3. Receive (MPI_Bcast) weights
        	//cout << "Slave-" << curr_rank << ": Receive weights...";
    		double *weightsHidden = new double[hiddenWeightsCount];
    		MPI_Bcast(&weightsHidden[0], hiddenWeightsCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    		double *weightsOutput = new double[outputWeightsCount];
    		MPI_Bcast(&weightsOutput[0], outputWeightsCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        	//cout << "FINISHED!" << endl;

    		// 4. Update Weights
        	//cout << "Slave-" << curr_rank << ": Update Weights...";
    		setWeights(nn, HIDDEN, weightsHidden, 1);
    		setWeights(nn, OUTPUT, weightsOutput, 1);
        	//cout << "FINISHED!" << endl;

        	// 5. Perform training step
        	//cout << "Slave-" << curr_rank << ": Execute training step...";
        	double error = nn.train(*imageDataset, *labelDataset, 100.0, 0.0);
        	//cout << "FINISHED!" << endl;

			// 6. Send (MPI_Gather) delta weight
	    	cout << "Slave-" << curr_rank << ": Send delta weights...";
			double *deltaWeightsHidden = getWeightsByLayer(nn, HIDDEN);
			MPI_Send(&deltaWeightsHidden[0], hiddenWeightsCount, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
			//MPI_Gather(&deltaWeightsHidden[0], hiddenWeightsCount, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			delete[] deltaWeightsHidden;

			double *deltaWeightsOutput = getWeightsByLayer(nn, OUTPUT);
			MPI_Send(&deltaWeightsOutput[0], outputWeightsCount, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
			//MPI_Gather(&deltaWeightsOutput[0], outputWeightsCount, MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			delete[] deltaWeightsOutput;

			MPI_Reduce(&error, NULL, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	    	cout << "FINISHED!" << endl;

			// 7. Wait for command from master to go on or exit
	    	cout << "Slave-" << curr_rank << ": Wait for command from master to go on or exit...";
			MPI_Bcast(&needsFurtherTraining, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
	    	cout << "FINISHED!" << endl;
    	}
    }

	MPI_Finalize();
}

bool NeuralNetworkDistributed::saveYAML(string const& path) {
	ofstream fout(path);

	fout << nn;

	return true;
}
