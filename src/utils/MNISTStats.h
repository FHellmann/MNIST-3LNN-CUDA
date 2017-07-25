#ifndef MNISTSTATS_H_
#define MNISTSTATS_H_

#include "Screen.h"
#include <opencv2/core/core.hpp>


class MNISTStats {
public:
	Screen screen;

	/**
	 * Outputs a 28x28 text frame at a defined screen position
	 * @param row Row of terminal screen
	 * @param col Column of terminal screen
	 */
	void displayImageFrame(int y, int x);

	/**
	 * Outputs a 28x28 MNIST image as charachters ("."s and "X"s)
	 * @param img Pointer to a MNIST image
	 * @param lbl Target classification
	 * @param cls Actual classification
	 * @param row Row of terminal screen
	 * @param col Column of terminal screen
	 */
	void displayImage(cv::Mat const img, int lbl, int cls, int row, int col);

	/**
	 * Outputs reading progress while processing MNIST training images
	 * @param imgCount Number of images already read from the MNIST file
	 * @param errCount Number of errors (images incorrectly classified)
	 * @param y Row of terminal screen
	 * @param x Column of terminal screen
	 */
	void displayTrainingProgress(int iter, int maxImageCount, int imgCount, int errCount, int y, int x);

	void displayTestingProgress(int maxImageCount, int imgCount, int errCount, int y, int x);
};


#endif
