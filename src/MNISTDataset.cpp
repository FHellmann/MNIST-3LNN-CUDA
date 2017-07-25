#include "MNISTDataset.hpp"
#include <iostream>

/** Specialize for labels. */
template<>
bool MNISTDataset<uint8_t>::load() {

	FILE *f = fopen(m_FileName.c_str(), "rb");

	if (nullptr == f) {
		std::cerr << "Unable to open file '" << m_FileName << "'." << std::endl;
		return false;
	}

	IdxHeader header;
	fread(&header, sizeof(header), 1, f);
	header.magicnumber = bigToLittleEndian(header.magicnumber);
	/*
	std::cout << "Header size: " << sizeof(IdxHeader) << std::endl;
	std::cout << "Magic Number: " << std::hex << header.magicnumber
			<< std::endl;
	std::cout << "Datatype: " << std::hex << (int) header.datatype << std::endl;
	std::cout << "Dimensions: " << std::hex << (int) header.dimensions
			<< std::endl;
			*/

	if (header.datatype != UBYTE) {
		std::cerr << "Unhandled datatype: " << (int) header.datatype
				<< std::endl;
		fclose(f);
		return false;
	}

	if (header.dimensions != 1) {
		std::cerr << "Not exactly 1 dimension. Not supported." << std::endl;
		fclose(f);
		return false;
	}

	// Read the dimensions
	uint32_t num;
	fread(&num, sizeof(uint32_t), 1, f);

	m_Count = bigToLittleEndian(num);

	//std::cout << std::dec << "Reading " << m_Count << " labels." << std::endl;

	m_Images.reserve(m_Count);
	// Actually read the images.
	for (int i = 0; i < m_Count; ++i) {
		uint8_t label;
		fread(&label, sizeof(uint8_t), 1, f);
		m_Images.push_back(label);
	}

	fclose(f);
	return true;
}

template class MNISTDataset<cv::Mat> ;
template class MNISTDataset<uint8_t> ;
