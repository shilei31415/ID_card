//
// Created by shilei on 20-5-2.
//

#include "IDcard.h"

inline std::string getName(std::string type)
{
    static int i=0;
    char num[10];
    sprintf(num,"%d",i++);
    return "../picture/"+std::string(num)+type;
}

IDcard::IDcard(std::string ANN_path)
{
    net = cv::dnn::readNetFromTensorflow(ANN_path);
    if(net.empty())
    {
        isLoad = false;
        std::cout<<"神将网络加载失败"<<std::endl;
    }
    else{
        for(int i=0;i<net.getLayerNames().size();i++)
            std::cout<<net.getLayerNames()[i]<<std::endl;
        std::cout<<"successfuly load ANN"<<std::endl;
        isLoad=true;
    }
}

IDcard::IDcard() {

}

std::vector<cv::RotatedRect> IDcard::findID(cv::Mat &gray_image) {
    std::vector<cv::RotatedRect> ROI;
    if(gray_image.channels()>1)
        cv::cvtColor(gray_image, gray_image, cv::COLOR_BGR2GRAY);
//    亮度阈值 与环境亮度有关
    gray_image=gray_image<130;
//    cv::imshow("gray", gray_image);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(gray_image, contours,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//    膨胀 与图片大小数字大小有关
    cv::drawContours(gray_image, contours, -1, cv::Scalar(255, 255, 255), 30);
    contours.clear();
    cv::findContours(gray_image, contours,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//    cv::imshow("gray_after", gray_image);

    for(int i=0;i<contours.size();i++)
    {
        cv::RotatedRect minRect = cv::minAreaRect(contours[i]);
        cv::Point2f vertex[4];
        minRect.points(vertex);
        if(MAX(minRect.size.width,minRect.size.height)/
           MIN(minRect.size.width,minRect.size.height)>7) {
            ROI.push_back(minRect);
        }
    }
    return ROI;
}

std::vector<char> IDcard::getID(cv::Mat &frame) {
    cv::Mat temp = frame.clone();
    std::vector<cv::RotatedRect> ROI = findID(temp);
//    std::cout<<ROI.size()<<std::endl;
    for(int j=0;j<ROI.size();j++) {
//        字符分割
        cut_char(ROI[j], frame);
//        显示
        cv::Point2f points[4];
        ROI[j].points(points);
        for (int i = 0; i < 4; i++)
            cv::line(frame, points[i], points[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
    }
    cv::imshow("findID", frame);

    return std::vector<char>();
}

std::vector<cv::Mat> IDcard::cut_char(cv::RotatedRect ROI, cv::Mat &image) {
//    旋转校正
    cv::Mat rot_mat = cv::getRotationMatrix2D(ROI.center, ROI.angle, 1.0);
    cv::Mat temp;
    cv::warpAffine(image, temp, rot_mat, image.size());

    int x=ROI.center.x - (ROI.size.width / 2), y=ROI.center.y - (ROI.size.height/2);
    if(x<0||x+ROI.size.width>1920||y<0||y+ROI.size.height>1080)
        return std::vector<cv::Mat>();

    cv::Mat result = temp(cv::Rect(x, y, ROI.size.width, ROI.size.height));

//    阈值化 与环境亮度有关
    cv::Mat gray;
    if(result.channels()>1)
    {
        cv::cvtColor(result, gray, cv::COLOR_BGR2GRAY);
        gray=gray<135;
    }
    else
        gray=result<135;

    cv::imshow("result", gray);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    for(int i=0;i<contours.size();i++) {
        cv::Rect rect = cv::boundingRect(contours[i]);
        cv::Mat char_image = result(rect);
        int top=0, left=0;
        if(char_image.cols>char_image.rows)
            left=0, top=char_image.cols-char_image.rows;
        if(char_image.cols<char_image.rows)
            left=char_image.rows-char_image.cols, top=0;
//        std::cout<<char_image.size<<"  "<<top<<"  "<<left<<"  ";

        cv::copyMakeBorder(result(rect).clone(),char_image,
                           top,0,left,0,cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
//        std::cout<<char_image.size<<std::endl;
        cv::imwrite(getName(".png"), char_image);
    }

    return std::vector<cv::Mat>();
}