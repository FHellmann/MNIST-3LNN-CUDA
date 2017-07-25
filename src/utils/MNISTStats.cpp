#include "MNISTStats.h"

/*
void MNISTStats::displayImageFrame(int row, int col){

    if (col!=0 && row!=0) screen.locateCursor(row, col);

    printf("------------------------------\n");

    for (int i=0; i<MNIST_IMG_HEIGHT; i++){
        for (int o=0; o<col-1; o++) printf(" ");
        printf("|                            |\n");
    }

    for (int o=0; o<col-1; o++) printf(" ");
    printf("------------------------------\n");

}
*/

void MNISTStats::displayImage(cv::Mat const img, int lbl, int cls, int row, int col){
    char imgStr[(img.rows * img.cols)+((col+1)*img.rows)+1];
    strcpy(imgStr, "");

    Screen::Color textColor = Screen::Color::RED;
    if(lbl == cls) {
    	textColor = Screen::Color::GREEN;
    }
	screen.setColor(textColor);

    for (int y=0; y<img.rows; y++){

        for (int o=0; o<col-2; o++) strcat(imgStr," ");
//        strcat(imgStr,"|");

        for (int x=0; x<img.cols; x++){
            strcat(imgStr, img.data[y*img.rows+x] ? "X" : "." );
        }
        strcat(imgStr,"\n");
    }

    if (col!=0 && row!=0) screen.locateCursor(row, 0);
    printf("%s",imgStr);

    printf("     Label:%d   Classification:%d\n\n",lbl,cls);

}

void MNISTStats::displayTrainingProgress(int iter, int maxImageCount, int imgCount, int errCount, int y, int x){
	screen.setColor(Screen::Color::WHITE);

    double progress = (double)(imgCount+1)/(double)(maxImageCount)*100;

    if (x!=0 && y!=0) screen.locateCursor(y, x);

    printf("%d: TRAINING: Reading image No. %5d of %5d images [%3d%%]  ",iter,(imgCount+1),maxImageCount,(int)progress);


    double accuracy = 1 - ((double)errCount/(double)(imgCount+1));

    printf("Result: Correct=%5d  Incorrect=%5d  Accuracy=%5.4f%% \n",imgCount+1-errCount, errCount, accuracy*100);

}

void MNISTStats::displayTestingProgress(int maxImageCount, int imgCount, int errCount, int y, int x){
	screen.setColor(Screen::Color::WHITE);

    double progress = (double)(imgCount+1)/(double)(maxImageCount)*100;

    if (x!=0 && y!=0) screen.locateCursor(y, x);

    printf("1: TESTING:  Reading image No. %5d of %5d images [%3d%%]  ",(imgCount+1),maxImageCount,(int)progress);


    double accuracy = 1 - ((double)errCount/(double)(imgCount+1));

    printf("Result: Correct=%5d  Incorrect=%5d  Accuracy=%5.4f%% \n",imgCount+1-errCount, errCount, accuracy*100);

}
