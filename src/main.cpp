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

using namespace std;
using namespace TCLAP;

vector<cv::Mat> loadMNISTImages(string const& path);

template <typename T>
T bigToLittleEndian(T& bigEndian) {

	uint8_t const* const data = reinterpret_cast<uint8_t const* const>(&bigEndian);
	T littleEndian = 0;
	int shift = 0;
	for (int i = sizeof(T) - 1; i >= 0; --i, shift += 8) {
		littleEndian |= (data[i] << shift);
	}
	return littleEndian;
}

struct IdxHeader {
	union {
		unsigned int magicnumber;
		struct {
			uint8_t dimensions;
			uint8_t datatype;
			uint16_t zero;
		};
	};
};

enum IdxDatatypes {
	UBYTE  = 0x08,
	SBYTE  = 0x09,
	SHORT  = 0x0B,
	INT    = 0x0C,
	FLOAT  = 0x0D,
	DOUBLE = 0x0E,
};

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
	vector<cv::Mat> images = loadMNISTImages(imagePath);

	for (cv::Mat const& img : images) {
		cv::imshow("Hand writing", img);
		cv::waitKey(0);
	}
	exit (EXIT_SUCCESS);
}

vector<cv::Mat> loadMNISTImages(string const& imagesFile) {

	vector<cv::Mat> images;
	/*
	For a "file format specification" see http://yann.lecun.com/exdb/mnist/
	This is what it states:

	THE IDX FILE FORMAT
	the IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.

	The basic format is

	magic number
	size in dimension 0
	size in dimension 1
	size in dimension 2
	.....
	size in dimension N
	data

	The magic number is an integer (MSB first). The first 2 bytes are always 0.

	The third byte codes the type of the data:
	0x08: unsigned byte
	0x09: signed byte
	0x0B: short (2 bytes)
	0x0C: int (4 bytes)
	0x0D: float (4 bytes)
	0x0E: double (8 bytes)

	The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....

	The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).

	The data is stored like in a C array, i.e. the index in the last dimension changes the fastest.
	*/
	FILE *f = fopen(imagesFile.c_str(), "rb");

	// Determine the file size.
	IdxHeader header;
	fread(&header, sizeof(header), 1, f);
	header.magicnumber = bigToLittleEndian(header.magicnumber);
	cout << "Header size: " << sizeof(IdxHeader) << endl;
	cout << "Magic Number: " << std::hex << header.magicnumber << endl;
	cout << "Datatype: " << std::hex << (int)header.datatype << endl;
	cout << "Dimensions: " << std::hex << (int)header.dimensions << endl;

	if (header.datatype != UBYTE) {
		cerr << "Unhandled datatype: " << (int) header.datatype << endl;
		fclose(f);
		return images;
	}

	if (header.dimensions != 3) {
		cerr << "Not exactly 3 dimensions. Not supported." << endl;
		fclose(f);
		return images;
	}

	// Read the dimensions
	uint32_t dimX;
	uint32_t dimY;
	uint32_t num;
	fread(&num, sizeof(uint32_t), 1, f);
	fread(&dimX, sizeof(uint32_t), 1, f);
	fread(&dimY, sizeof(uint32_t), 1, f);

	num = bigToLittleEndian(num);
	dimX = bigToLittleEndian(dimX);
	dimY = bigToLittleEndian(dimY);

	cout << std::dec << "Reading " << num << " images of size (" << dimX << "," << dimY << ")" << endl;

	images.reserve(num);
	for (int i = 0; i < num; ++i) {
		cv::Mat img(dimX, dimY, CV_8UC1);
		fread(img.data, sizeof(uint8_t), dimX * dimY, f);
		images.push_back(img);
	}
//	fseek(f, 16, SEEK_CUR);
//	unsigned int imageSize = 28*28;
//	char *images = new char[num*imageSize];
//	size_t res = fread(images, 1, num*imageSize, f);
	fclose(f);
//	if (num*imageSize != res)
//	{
//		delete[] images;
//		return images;
//	}
////	f = fopen(labelsFile.c_str(), "rb");
////	fseek(f, 8, SEEK_CUR);
////	char *labels = new char[num];
////	res = fread(labels, 1, num, f);
////	fclose(f);
////	if (num != res)
////	{
////		delete[] images;
////		delete[] labels;
////		return;
////	}
//
//	for (unsigned int i=0; i<num; ++i)
//	{
//		Ptr<OR_mnistObj> curr(new OR_mnistObj);
//		curr->label = labels[i];
//
//		curr->image = Mat(28, 28, CV_8U);
//		unsigned int imageIdx = i*imageSize;
//		for (int j=0; j<curr->image.rows; ++j)
//		{
//			char *im = curr->image.ptr<char>(j);
//			for (int k=0; k<curr->image.cols; ++k)
//			{
//				im[k] = images[imageIdx + j*28 + k];
//			}
//		}
//
//		dataset_.push_back(curr);
//	}
//	delete[] images;
//	delete[] labels;
	return images;
}
