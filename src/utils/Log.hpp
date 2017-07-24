#ifndef LOG_HPP_
#define LOG_HPP_

#include <iostream>
#include <chrono>

#define LOG_MASTER 1
#define LOG_SLAVE 0

/**
 * A simple logging method to print something out.
 *
 * @param rank The rank of the process.
 * @param msg The message to print out.
 */
static void log(std::string const msg, int rank = 0) {
	if ((!LOG_MASTER && rank == 0) || (!LOG_SLAVE && rank > 0))
		return;

	std::cout << (rank == 0 ? "MASTER" : "SLAVE-" + rank) << ": " << msg << std::endl;
	std::cout.flush();
}

/**
 * The logging method to measure the execution time of everything which
 * is between logStart and logEnd.
 *
 * @param rank The rank of the process.
 * @param msg The message to print out.
 */
static std::chrono::high_resolution_clock::time_point logStart(std::string const msg, int rank = 0) {
	if ((!LOG_MASTER && rank == 0) || (!LOG_SLAVE && rank > 0))
		return std::chrono::high_resolution_clock::now();

	std::cout << (rank == 0 ? "MASTER" : "SLAVE-" + rank) << ": " << msg;
	std::cout.flush();

	return std::chrono::high_resolution_clock::now();
}

/**
 * The finishing logging method to logStart(int, string). It will
 * print the needed time between logStart and logEnd.
 *
 * @param rank The rank of the process.
 * @param time The time which was returned from logStart.
 */
static void logEnd(std::chrono::high_resolution_clock::time_point const time, int rank = 0) {
	if ((!LOG_MASTER && rank == 0) || (!LOG_SLAVE && rank > 0))
		return;

	std::cout << "FINISHED! (Time=" << std::chrono::duration_cast<
			std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - time).count() << "s)" << std::endl;
	std::cout.flush();
}

#endif
