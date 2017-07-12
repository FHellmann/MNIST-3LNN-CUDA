#ifndef MNISTDATASET_HPP_
#define MNISTDATASET_HPP_

#include "MNISTDataset.h"
#include <iostream>

template<typename T>
MNISTDataset<T>::MNISTDataset(std::string const& fileName) :
		m_FileName(fileName), m_Count(0), m_Datatype(
				static_cast<IdxDatatype>(0)) {
}

template<typename T>
MNISTDataset<T>::~MNISTDataset() {
}

/**
 * Struct for easier file loading.
 */
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

/** Function needed for loading the images. */
template<typename T>
T bigToLittleEndian(T& bigEndian) {

	uint8_t const* const data =
			reinterpret_cast<uint8_t const* const >(&bigEndian);
	T littleEndian = 0;
	int shift = 0;
	for (int i = sizeof(T) - 1; i >= 0; --i, shift += 8) {
		littleEndian |= (data[i] << shift);
	}
	return littleEndian;
}

template<typename T>
bool MNISTDataset<T>::load() {
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
	FILE *f = fopen(m_FileName.c_str(), "rb");

	if (nullptr == f) {
		std::cerr << "Unable to open file '" << m_FileName << "'." << std::endl;
		return false;
	}

	IdxHeader header;
	fread(&header, sizeof(header), 1, f);
	header.magicnumber = bigToLittleEndian(header.magicnumber);
	std::cout << "Header size: " << sizeof(IdxHeader) << std::endl;
	std::cout << "Magic Number: " << std::hex << header.magicnumber
			<< std::endl;
	std::cout << "Datatype: " << std::hex << (int) header.datatype << std::endl;
	std::cout << "Dimensions: " << std::hex << (int) header.dimensions
			<< std::endl;

	if (header.datatype != UBYTE) {
		std::cerr << "Unhandled datatype: " << (int) header.datatype
				<< std::endl;
		fclose(f);
		return false;
	}

	if (header.dimensions != 3) {
		std::cerr << "Not exactly 3 dimensions. Not supported." << std::endl;
		fclose(f);
		return false;
	}

	// Read the dimensions
	uint32_t dimX;
	uint32_t dimY;
	uint32_t num;
	fread(&num, sizeof(uint32_t), 1, f);
	fread(&dimX, sizeof(uint32_t), 1, f);
	fread(&dimY, sizeof(uint32_t), 1, f);

	m_Count = bigToLittleEndian(num);
	dimX = bigToLittleEndian(dimX);
	dimY = bigToLittleEndian(dimY);

	std::cout << std::dec << "Reading " << m_Count << " images of size ("
			<< dimX << "," << dimY << ")" << std::endl;

	m_Images.reserve(m_Count);
	// Actually read the images.
	for (int i = 0; i < m_Count; ++i) {
		cv::Mat img(dimX, dimY, CV_8UC1);
		fread(img.data, sizeof(uint8_t), dimX * dimY, f);
		m_Images.push_back(img);
	}

	fclose(f);
	return true;
}

template<typename T>
typename MNISTDataset<T>::IdxDatatype MNISTDataset<T>::getDatatype() const {
	return m_Datatype;
}

template<typename T>
std::string MNISTDataset<T>::getPath() const {
	return m_FileName;
}

template<typename T>
size_t MNISTDataset<T>::size() const {
	return m_Images.size();
}

template<typename T>
T& MNISTDataset<T>::operator[](int const& idx) {
	return m_Images[idx];
}

template<typename T>
T const& MNISTDataset<T>::operator[](int const& idx) const {
	return m_Images[idx];
}

template<typename T>
typename MNISTDataset<T>::iterator MNISTDataset<T>::begin() {
	return m_Images.begin();
}

template<typename T>
typename MNISTDataset<T>::const_iterator MNISTDataset<T>::begin() const {
	return m_Images.begin();
}

template<typename T>
typename MNISTDataset<T>::iterator MNISTDataset<T>::end() {
	return m_Images.end();
}

template<typename T>
typename MNISTDataset<T>::const_iterator MNISTDataset<T>::end() const {
	return m_Images.end();
}

template<typename T>
T& MNISTDataset<T>::front() {
	return m_Images.front();
}

template<typename T>
T const& MNISTDataset<T>::front() const {
	return m_Images.front();
}

template<typename T>
T& MNISTDataset<T>::back() {
	return m_Images.back();
}

template<typename T>
T const& MNISTDataset<T>::back() const {
	return m_Images.back();
}

#endif /* MNISTDATASET_HPP_ */
