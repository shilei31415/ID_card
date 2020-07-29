//
// Created by shilei on 20-5-2.
//

#ifndef ID_CARD_IDCARD_H
#define ID_CARD_IDCARD_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <vector>

class IDcard {
public:
    std::vector<char> getID(cv::Mat &frame);
    std::vector<cv::RotatedRect> findID(cv::Mat &gray_image);
    std::vector<cv::Mat> cut_char(cv::RotatedRect ROI, cv::Mat &image);

    IDcard(std::string ANN_path);
    IDcard();
    bool isLoad;
private:
    int forward(cv::Mat &input);
    cv::dnn::Net net;
    char ID[11]={'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'X'};
};


#endif //ID_CARD_IDCARD_H
