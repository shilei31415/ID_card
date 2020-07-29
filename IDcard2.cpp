//
// Created by shilei on 20-5-5.
//

#include "IDcard2.h"
#include "MyLayer.hpp"
#include <opencv2/dnn/layer.details.hpp>

inline std::string getName(std::string type)
{
    static int i=0;
    char num[10];
    sprintf(num,"%d",i++);
    return "../picture/"+std::string(num)+type;
}

inline double cal_mean_stddev(cv::Mat &gray) {
    cv::Mat mat_mean, mat_stddev;
    if(gray.channels()>1)
        cvtColor(gray, gray, CV_RGB2GRAY); // 转换为灰度图
    meanStdDev(gray, mat_mean, mat_stddev);
    double m;
    m = mat_mean.at<double>(0, 0);
    return m;
}

IDcard::IDcard(std::string ANN_path, std::string ANN_txt_path)
{
    CV_DNN_REGISTER_LAYER_CLASS(LeakyRelu, LeakyReluLayer)
    net = cv::dnn::readNetFromTensorflow(ANN_path, ANN_txt_path);
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

std::vector<char> IDcard::getID(cv::Mat &frame) {
    cv::Mat temp = frame.clone();
    std::vector<cv::RotatedRect> ROI = findID(temp);
    for(int j=0;j<ROI.size();j++) {
        std::string ID_result=get_char(ROI[j], frame);

        cv::Point2f points[4];
        ROI[j].points(points);
        for (int i = 0; i < 4; i++)
            cv::line(frame, points[i], points[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);

        cv::putText(frame, ID_result, points[1], cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(255, 50, 100), 3);
    }

    return std::vector<char>();
}

std::vector<cv::RotatedRect> IDcard::findID(cv::Mat &gray_image) {
    std::vector<cv::RotatedRect> ROI;
    if(gray_image.channels()>1)
        cv::cvtColor(gray_image, gray_image, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray_image, gray_image, cv::Size(3, 3), 1.5, 1.5);
    cv::Canny(gray_image, canny_result, edgeThresh, edgeThresh*3, 3);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(canny_result, contours,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::drawContours(canny_result, contours, -1, cv::Scalar(255, 255, 255), 30);
    contours.clear();
    cv::findContours(canny_result, contours,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//    cv::imshow("findID_result", canny_result);
    for(int i=0; i<contours.size();i++)
    {
        cv::RotatedRect minRect = cv::minAreaRect(contours[i]);
        if(MAX(minRect.size.width,minRect.size.height)/
           MIN(minRect.size.width,minRect.size.height)>7)
        {
            ROI.push_back(minRect);
        }
    }
    return ROI;
}

std::string IDcard::get_char(cv::RotatedRect ROI, cv::Mat &image) {
    cv::Mat rot_mat;
//    std::cout<<ROI.size.width<<"  "<<ROI.size.height<<"  "<<ROI.angle<<std::endl;
    if(fabs(ROI.angle)>60)
        ROI.angle=90-fabs(ROI.angle);
    rot_mat = cv::getRotationMatrix2D(ROI.center, ROI.angle, 1.0);
//    if(ROI.size.width<ROI.size.height)
//        rot_mat = cv::getRotationMatrix2D(ROI.center, ROI.angle+90, 1.0);
    cv::Mat temp;
    cv::warpAffine(image, temp, rot_mat, image.size());

    int x=ROI.center.x - (ROI.size.width / 2), y=ROI.center.y - (ROI.size.height/2);
    if(x<0||x+ROI.size.width>1920||y<0||y+ROI.size.height>1080)
        return "";

    cv::Mat result = temp(cv::Rect(x, y, ROI.size.width, ROI.size.height));

//    cv::waitKey(0);
//    cv::imwrite(getName(std::to_string(ROI.angle)+".png"), result);

//    cut_char(result);
    return forward(result);
}

void IDcard::cut_char(cv::Mat &src) {
    cv::Mat temp=src.clone();
    if(temp.channels()>1)
        cv::cvtColor(temp, temp, cv::COLOR_BGR2GRAY);
    double mean = cal_mean_stddev(temp);
    temp=temp<mean*0.85;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(temp, contours,
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    for(int i=0;i<contours.size();i++)
    {
        try {
            cv::Rect rect = cv::boundingRect(cv::Mat(contours[i]));
            rect.x-=5;
            rect.y-=5;
            rect.width+=10;
            rect.height+=10;
            cv::Mat num_img = src(rect);
            cv::imwrite(getName(".png"), num_img);
        }catch (cv::Exception& e) {}

    }

    cv::imshow("result", temp);
    cv::waitKey(1);
}

std::string IDcard::forward(cv::Mat &input) {
    std::string result="";
    if(isLoad==false)
        return result;
    cv::resize(input, input, cv::Size(448, 40));
    input.convertTo(input,CV_32FC3);
    input=input/255.0;
    input=cv::dnn::blobFromImage(input,1.0,cv::Size(448, 40));
    net.setInput(input,"image");
    cv::Mat pred=net.forward("result");
    float *row=pred.ptr<float>(0);
    for(int i=0; i<18; i++)
    {
        int max=0;
        for(int j=0;j<12;j++)
            if(row[i*12+j]>row[i*12+max])
                max=j;
        result+=this->ID[max];
//        std::cout<<this->ID[max]<<"  ";
    }
//    std::cout<<result<<std::endl;

    return result;
}
