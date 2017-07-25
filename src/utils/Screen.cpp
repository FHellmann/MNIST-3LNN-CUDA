#include "Screen.h"

void Screen::clearScreen(){
	printf("\033[2J"); // Clear screen
	system("clear");
}

void Screen::setColor(Color c){
    std::string esc;
    switch (c) {
        case RED   : esc = "1;31";
            break;
        case GREEN : esc = "1;32";
            break;
        case YELLOW: esc = "1;33";
            break;
        case BLUE  : esc = "1;34";
            break;
        case CYAN  : esc = "1;36";
            break;
        default : esc = "0;00"; // default WHITE
            break;
    }
    printf("%c[%sm",27,esc.c_str());
}

void Screen::locateCursor(const int row, const int col){
    printf("%c[%d;%dH",27,row,col);
}
