/*
 * 3lnn_io.h
 *
 *  Created on: 05.07.2017
 *      Author: Stefan
 */

#ifndef THREE_LNN_IO_H_
#define THREE_LNN_IO_H_

#include <string>
#include <ostream>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include "NeuralNetwork.hpp"

using namespace std;

class NeuralNetwork;

std::ostream& operator<< (std::ostream& out, NeuralNetwork const& net);
bool saveNet(std::string const& path, NeuralNetwork const& net);
NeuralNetwork* loadNet(std::string const& path);

#endif /* 3LNN_IO_H_ */
