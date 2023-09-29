//
// Created by yeleussinova on 9/22/23.
//

#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <stack>
#include "net.h"
#include <chrono>

using namespace std::chrono;

struct Point{
    float _x;
    float _y;
};
struct bbox{
    float x1;
    float y1;
    float x2;
    float y2;
    float s;
    Point point[5];
};

struct box{
    float cx;
    float cy;
    float sx;
    float sy;
};

class Detector
{

public:
    Detector();

    void Init(const std::string &model_param, const std::string &model_bin);

    Detector(const std::string &model_param, const std::string &model_bin, bool retinaface = false);

    inline void Release();

    void nms(std::vector<bbox> &input_boxes, float NMS_THRESH);

    void Detect(cv::Mat& bgr, std::vector<bbox>& boxes);

    void create_anchor(std::vector<box> &anchor, int w, int h);


    bbox chooseOneFace(std::vector<bbox> &inpt_boxes, float scale);

    cv::Mat alignFace(cv::Mat img, bbox input_box);

    static inline bool cmp(bbox a, bbox b);

    cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h);
    ~Detector();

public:
    float _nms;
    float _threshold;
    float _mean_val[3];

    ncnn::Net *Net;

    int outputHeight =112;
    int outputWidth = 112;

    bool default_square = true;
    float inner_padding_factor = 0.25;
};
#endif //