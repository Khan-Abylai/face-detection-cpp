//
// Created by yeleussinova on 9/22/23.
//

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "FaceDetector.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    string imgPath = "../test/Arman.jpg";

    string param = "../model/face.param";
    string bin = "../model/face.bin";
    const int max_side = 640;

    Detector detector(param, bin, false);

    cv::Mat img = cv::imread(imgPath.c_str());


    // scale
    float long_side = std::max(img.cols, img.rows);
    float scale = max_side/long_side;
    cv::Mat img_scale;
    cv::resize(img, img_scale, cv::Size(img.cols*scale, img.rows*scale));

    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imgPath.c_str());
        return -1;
    }
    std::vector<bbox> boxes;
    bbox oneBox;

    detector.Detect(img_scale, boxes);

    if (boxes.size() > 0){
        cout<< "Face detected: " << boxes.size();
        oneBox = detector.chooseOneFace(boxes, scale);
        cv::Mat aligned = detector.alignFace(img, oneBox);
        cv::imwrite("../res/test_crop.png", aligned);
        cv::Rect rect(oneBox.x1, oneBox.y1, oneBox.x2 - oneBox.x1, oneBox.y2 - oneBox.y1);
        cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1, 8, 0);
    }

    cv::imwrite("../res/test_res.png", img);

    return 0;
}