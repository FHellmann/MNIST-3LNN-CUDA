/*
 * 3lnn_io.h
 *
 *  Created on: 05.07.2017
 *      Author: Stefan
 */

#ifndef THREE_LNN_IO_H_
#define THREE_LNN_IO_H_

#include <string>

class Network;

bool saveNet(std::string const& path, Network const& net);

#endif /* 3LNN_IO_H_ */
