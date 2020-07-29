#include <iostream>
#include <opencv2/opencv.hpp>
#include "IDcard2.h"


int main() {
    cv::VideoCapture capture("/media/shilei/ʩ/VID_20200502_111804.mp4");
    cv::VideoWriter writer("result.mp4", CV_FOURCC('D', 'I', 'V', 'X'), 30, cv::Size(1920, 1080));
    cv::Mat frame;
    IDcard card("../ANN/savePB/model_new.pb", "../ANN/savePB/model_new.pbtxt");
    cv::namedWindow("video", cv::WINDOW_NORMAL);
//    cv::namedWindow("findID_result", cv::WINDOW_NORMAL);
    int i=0;
    while(1)
    {
        i++;
        capture>>frame;
//        std::cout<<frame.size<<std::endl;
        if(!frame.data)
            break;

        std::vector<char> ID=card.getID(frame);

//        cv::imshow("video", frame);
//        cv::waitKey(1);
        if(i>945&&i<1174)
            writer<<frame;
    }
    return 0;
}