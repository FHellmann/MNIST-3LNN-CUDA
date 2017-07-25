#ifndef SCREEN_H_
#define SCREEN_H_

#include <stdio.h>
#include <stdlib.h>
#include <string>


class Screen {
public:
	enum Color {WHITE, RED, GREEN, YELLOW, BLUE, CYAN};
	const Color DEFAULT_TEXT_COLOR = WHITE;

	/**
	 * Clear terminal screen by printing an escape sequence
	 */
	void clearScreen();

	/**
	 * Set text color in terminal by printing an escape sequence
	 */
	void setColor(Color c);

	/**
	 * Set cursor position to given coordinates in the terminal window
	 * @param row Row number in terminal screen
	 * @param col Column number in terminal screen
	 */
	void locateCursor(const int row, const int col);
};


#endif /* SCREEN_H_ */
