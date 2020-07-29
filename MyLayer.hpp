//
// Created by shilei on 20-5-18.
//

#ifndef ID_CARD_MYLAYER_HPP
#define ID_CARD_MYLAYER_HPP

#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <string>

class LeakyReluLayer: public  cv::dnn::Layer
{
public:
    LeakyReluLayer(const cv::dnn::LayerParams &params):cv::dnn::Layer(params)
    {
        alpha=params.get<float>("alpha");
//        std::cout<<params.type<<"params:\n"<<"alpha: "<<alpha<<std::endl;
    }

    static cv::Ptr<cv::dnn::Layer> create(cv::dnn::LayerParams& params)
    {
        return cv::Ptr<cv::dnn::Layer>(new LeakyReluLayer(params));
    }

    virtual void forward(cv::InputArrayOfArrays inputs_arr,
                         cv::OutputArrayOfArrays outputs_arr,
                         cv::OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        std::vector<cv::Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        cv::Mat& inp = inputs[0];
        cv::Mat& out = outputs[0];
        const float* inpData = (float*)inp.data;
        float* outData = (float*)out.data;

        const int batchSize = inp.size[0];
        const int rows = inp.size[1];
        const int clos = inp.size[2];
        const int channel = inp.size[3];

        for(int b=0; b<batchSize; b++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < clos; c++) {
                    for (int ch = 0; ch < channel; ch++) {
                        int id = inp.step[0] * b + inp.step[1] * r + inp.step[2] * c + inp.step[3] * ch;
                        *(float *)(out.data + id) = Leaky_relu(*(float *) (inp.data + id));
                    }
                }
            }
        }
    }

private:
    float alpha;
    float Leaky_relu(float input)
    {
        if(input>=0)
            return input;
        else
            return alpha*input;
    }
};

#endif //ID_CARD_MYLAYER_HPP
