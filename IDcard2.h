//
// Created by shilei on 20-5-5.
//

#ifndef ID_CARD_IDCARD2_H
#define ID_CARD_IDCARD2_H


#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <vector>

class IDcard {
public:
    std::vector<char> getID(cv::Mat &frame);
    std::vector<cv::RotatedRect> findID(cv::Mat &gray_image);
    std::string get_char(cv::RotatedRect ROI, cv::Mat &image);

    IDcard(std::string ANN_path, std::string ANN_txt_path);
    IDcard();
    bool isLoad;
//    字符串分割,用于制作数据集
    void cut_char(cv::Mat& temp);
private:
    std::string forward(cv::Mat &input);
    cv::dnn::Net net;
    char ID[12]={'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'X', 'e'};

    cv::Mat canny_result;
    int edgeThresh = 20;
};


#endif //ID_CARD_IDCARD2_H
