//
// Created by yeleussinova on 9/22/23.
//

#include "FaceDetector.h"
#include <algorithm>
#include "omp.h"
using namespace std;

//Detector::Detector():
//        _nms(0.4),
//        _threshold(0.6),
//        _mean_val{104.f, 117.f, 123.f},
//        Net(new ncnn::Net())
//{
//}

inline void Detector::Release(){
    if (Net != nullptr)
    {
        delete Net;
        Net = nullptr;
    }
}

Detector::Detector(const std::string &model_param, const std::string &model_bin, bool retinaface):
        _nms(0.4),
        _threshold(0.75),
        _mean_val{104.f, 117.f, 123.f},
        Net(new ncnn::Net())
{
    Init(model_param, model_bin);
}

void Detector::Init(const std::string &model_param, const std::string &model_bin)
{
    int ret = Net->load_param(model_param.c_str());
    ret = Net->load_model(model_bin.c_str());
    cout << "ret: " << ret << endl;
 }

void Detector::Detect(cv::Mat& bgr, std::vector<bbox>& boxes)
{

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, bgr.cols, bgr.rows);
    in.substract_mean_normalize(_mean_val, 0);
    ncnn::Extractor ex = Net->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input(0, in);
    ncnn::Mat out, out1, out2;

    // loc
    ex.extract("output0", out);

    // class
    ex.extract("530", out1);

    //landmark
    ex.extract("529", out2);


    std::vector<box> anchor;

    create_anchor(anchor,  bgr.cols, bgr.rows);

    std::vector<bbox > total_box;
    float *ptr = out.channel(0);
    float *ptr1 = out1.channel(0);
    float *landms = out2.channel(0);

    // #pragma omp parallel for num_threads(2)
    for (int i = 0; i < anchor.size(); ++i)
    {
        if (*(ptr1+1) > _threshold)
        {
            box tmp = anchor[i];
            box tmp1;
            bbox result;

            // loc and conf
            tmp1.cx = tmp.cx + *ptr * 0.1 * tmp.sx;
            tmp1.cy = tmp.cy + *(ptr+1) * 0.1 * tmp.sy;
            tmp1.sx = tmp.sx * exp(*(ptr+2) * 0.2);
            tmp1.sy = tmp.sy * exp(*(ptr+3) * 0.2);

            result.x1 = (tmp1.cx - tmp1.sx/2) * in.w;
            if (result.x1<0)
                result.x1 = 0;
            result.y1 = (tmp1.cy - tmp1.sy/2) * in.h;
            if (result.y1<0)
                result.y1 = 0;
            result.x2 = (tmp1.cx + tmp1.sx/2) * in.w;
            if (result.x2>in.w)
                result.x2 = in.w;
            result.y2 = (tmp1.cy + tmp1.sy/2)* in.h;
            if (result.y2>in.h)
                result.y2 = in.h;
            result.s = *(ptr1 + 1);

            // landmark
            for (int j = 0; j < 5; ++j)
            {
                result.point[j]._x =( tmp.cx + *(landms + (j<<1)) * 0.1 * tmp.sx ) * in.w;
                result.point[j]._y =( tmp.cy + *(landms + (j<<1) + 1) * 0.1 * tmp.sy ) * in.h;
            }

            total_box.push_back(result);
        }
        ptr += 4;
        ptr1 += 2;
        landms += 10;
    }

    std::sort(total_box.begin(), total_box.end(), cmp);
    nms(total_box, _nms);

    for (const auto & j : total_box)
    {
        boxes.push_back(j);
    }

}

inline bool Detector::cmp(bbox a, bbox b) {
    if (a.s > b.s)
        return true;
    return false;
}


Detector::~Detector(){
    Release();
}

void Detector::create_anchor(std::vector<box> &anchor, int w, int h)
{
//    anchor.reserve(num_boxes);
    anchor.clear();
    std::vector<std::vector<int> > feature_map(4), min_sizes(4);
    float steps[] = {8, 16, 32, 64};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));
    }
    std::vector<int> minsize1 = {10, 16, 24};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {32, 48};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {64, 96};
    min_sizes[2] = minsize3;
    std::vector<int> minsize4 = {128, 192, 256};
    min_sizes[3] = minsize4;


    for (int k = 0; k < feature_map.size(); ++k)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l = 0; l < min_size.size(); ++l)
                {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    float cx = (j + 0.5) * steps[k]/w;
                    float cy = (i + 0.5) * steps[k]/h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }

    }

}

void Detector::nms(std::vector<bbox> &input_boxes, float NMS_THRESH)
{
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float   h = std::max(float(0), yy2 - yy1 + 1);
            float   inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

bbox Detector::chooseOneFace(std::vector<bbox> &input_boxes, float scale){
    bbox result;
    float maxArea = -1.0;
    if (input_boxes.size() > 1){
        for (auto &input_box: input_boxes){
            float area = abs(input_box.x2/scale - input_box.x1/scale) * abs(input_box.y2/scale - input_box.y1/scale);
            if (area > maxArea){
                maxArea = area;
                result.x1 = input_box.x1/scale;
                result.x2 = input_box.x2/scale;
                result.y1 = input_box.y1/scale;
                result.y2 = input_box.y2/scale;
                for(int i=0; i<5; i++){
                    result.point[i]._x = input_box.point[i]._x/scale;
                    result.point[i]._y = input_box.point[i]._y/scale;
                }
            }
        }
    }
    else{
        result.x1 = input_boxes[0].x1/scale;
        result.x2 = input_boxes[0].x2/scale;
        result.y1 = input_boxes[0].y1/scale;
        result.y2 = input_boxes[0].y2/scale;
        for(int i=0; i<5; i++){
            result.point[i]._x = input_boxes[0].point[i]._x/scale;
            result.point[i]._y = input_boxes[0].point[i]._y/scale;
        }
    }

    return result;
}

cv::Mat Detector::preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
    w = input_w;
    h = r_w * img.rows;
    x = 0;
    y = (input_h - h) / 2;
    } else {
    w = r_h * img.cols;
    h = input_h;
    x = (input_w - w) / 2;
    y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

std::vector<float> get_reference_facial_points(int outW, int outH, float innerPaddingFactor, std::pair<int, int> outerPadding, bool Square){
    float REFERENCE_FACIAL_POINTS[5][2] =  {{30.29459953, 51.69630051},
                                            {65.53179932, 51.50139999},
                                            {48.02519989, 71.73660278},
                                            {33.54930115, 92.3655014},
                                            {62.72990036, 92.20410156}};

    float tmp_5pts[5][2];
    pair<int, int> DEFAULT_CROP_SIZE = {96, 112}, tmpCropSize, diffSize;
    std::vector<float> res;

    if (Square){
        diffSize.first = max(DEFAULT_CROP_SIZE.first, DEFAULT_CROP_SIZE.second) - DEFAULT_CROP_SIZE.first;
        diffSize.second = max(DEFAULT_CROP_SIZE.first, DEFAULT_CROP_SIZE.second) - DEFAULT_CROP_SIZE.second;
        for (int i=0; i<5; i++){
            for(int j = 0; j<2; j++){
                if(j == 0){
                    tmp_5pts[i][j] = REFERENCE_FACIAL_POINTS[i][j] + float(diffSize.first/2);
                }
                else{
                    tmp_5pts[i][j] = REFERENCE_FACIAL_POINTS[i][j] + float(diffSize.second/2);

                }
                res.push_back(tmp_5pts[i][j]);
            }
        }
        tmpCropSize.first = DEFAULT_CROP_SIZE.first + diffSize.first;
        tmpCropSize.second = DEFAULT_CROP_SIZE.second + diffSize.second;
    }

    return res;
}

cv::Mat MeanAxis0(const cv::Mat &src) {
    int num = src.rows;
    int dim = src.cols;

    // x1 y1
    // x2 y2

    cv::Mat output(1, dim, CV_32FC1);
    for (int i = 0; i < dim; i++) {
        float sum = 0;
        for (int j = 0; j < num; j++) {
            sum += src.at<float>(j, i);
        }
        output.at<float>(0, i) = sum / num;
    }

    return output;
}

cv::Mat ElementwiseMinus(const cv::Mat &A, const cv::Mat &B) {
    cv::Mat output(A.rows, A.cols, A.type());
    assert(B.cols == A.cols);
    if (B.cols == A.cols) {
        for (int i = 0; i < A.rows; i++) {
            for (int j = 0; j < B.cols; j++) {
                output.at<float>(i, j) = A.at<float>(i, j) - B.at<float>(0, j);
            }
        }
    }

    return output;
}

cv::Mat VarAxis0(const cv::Mat &src) {
    cv::Mat temp_ = ElementwiseMinus(src, MeanAxis0(src));
    cv::multiply(temp_, temp_, temp_);
    return MeanAxis0(temp_);
}

int MatrixRank(cv::Mat M) {
    cv::Mat w, u, vt;
    cv::SVD::compute(M, w, u, vt);
    cv::Mat1b nonZeroSingularValues = w > 0.0001;
    int rank = countNonZero(nonZeroSingularValues);
    return rank;
}

cv::Mat SimilarTransform(const cv::Mat &src, const cv::Mat &dst) {
    int num = src.rows;
    int dim = src.cols;
    cv::Mat src_mean = MeanAxis0(src);
    cv::Mat dst_mean = MeanAxis0(dst);
    cv::Mat src_demean = ElementwiseMinus(src, src_mean);
    cv::Mat dst_demean = ElementwiseMinus(dst, dst_mean);
    cv::Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
    cv::Mat d(dim, 1, CV_32F);
    d.setTo(1.0f);
    if (cv::determinant(A) < 0) {
        d.at<float>(dim - 1, 0) = -1;

    }
    cv::Mat T = cv::Mat::eye(dim + 1, dim + 1, CV_32F);
    cv::Mat U, S, V;
    cv::SVD::compute(A, S, U, V);

    // the SVD function in opencv differ from scipy .

    int rank = MatrixRank(A);
    if (rank == 0) {
        assert(rank == 0);

    } else if (rank == dim - 1) {
        if (cv::determinant(U) * cv::determinant(V) > 0) {
            T.rowRange(0, dim).colRange(0, dim) = U * V;
        } else {
            int s = d.at<float>(dim - 1, 0) = -1;
            d.at<float>(dim - 1, 0) = -1;

            T.rowRange(0, dim).colRange(0, dim) = U * V;
            cv::Mat diag_ = cv::Mat::diag(d);
            cv::Mat twp = diag_ * V; //np.dot(np.diag(d), V.T)
            cv::Mat B = cv::Mat::zeros(3, 3, CV_8UC1);
            cv::Mat C = B.diag(0);
            T.rowRange(0, dim).colRange(0, dim) = U * twp;
            d.at<float>(dim - 1, 0) = s;
        }
    } else {
        cv::Mat diag_ = cv::Mat::diag(d);
        cv::Mat twp = diag_ * V.t(); //np.dot(np.diag(d), V.T)
        cv::Mat res = U * twp; // U
        T.rowRange(0, dim).colRange(0, dim) = -U.t() * twp;
    }
    cv::Mat var_ = VarAxis0(src_demean);
    float val = cv::sum(var_).val[0];
    cv::Mat res;
    cv::multiply(d, S, res);
    float scale = 1.0 / val * cv::sum(res).val[0];
    T.rowRange(0, dim).colRange(0, dim) = -T.rowRange(0, dim).colRange(0, dim).t();
    cv::Mat temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
    cv::Mat temp2 = src_mean.t();
    cv::Mat temp3 = temp1 * temp2;
    cv::Mat temp4 = scale * temp3;
    T.rowRange(0, dim).colRange(dim, dim + 1) = -(temp4 - dst_mean.t());
    T.rowRange(0, dim).colRange(0, dim) *= scale;
    return T;
}

cv::Mat warp_and_crop_face(cv::Mat srcImg, bbox landmarks, vector<float> reference_5pts, int outW, int outH ){

    float points_src[5][2] = {
            {landmarks.point[0]._x, landmarks.point[0]._y},
            {landmarks.point[1]._x, landmarks.point[1]._y},
            {landmarks.point[2]._x, landmarks.point[2]._y},
            {landmarks.point[3]._x, landmarks.point[3]._y},
            {landmarks.point[4]._x, landmarks.point[4]._y}
    };

    float points_dst[5][2] = {
            {reference_5pts[0], reference_5pts[1]},
            {reference_5pts[2], reference_5pts[3]},
            {reference_5pts[4], reference_5pts[5]},
            {reference_5pts[6], reference_5pts[7]},
            {reference_5pts[8], reference_5pts[9]}
    };

    cv::Mat src_mat(5, 2, CV_32FC1, points_src);
    cv::Mat dst_mat(5, 2, CV_32FC1, points_dst);

    cv::Mat transform = SimilarTransform(src_mat, dst_mat);
    cv::Mat aligned_face;
    cv::Mat transfer_mat = transform(cv::Rect(0, 0, 3, 2));

    cv::warpAffine(srcImg.clone(), aligned_face, transfer_mat, cv::Size(112, 112), 1, 0, 0);

    return aligned_face;
}

cv::Mat Detector::alignFace(cv::Mat img, bbox input_box){

    std::pair<int, int> outer_padding;
    outer_padding.first = 0;
    outer_padding.second = 0;

    std::vector<float> reference_5pts = get_reference_facial_points(outputWidth, outputHeight, inner_padding_factor, outer_padding, default_square);

    return warp_and_crop_face(img, input_box, reference_5pts, outputWidth, outputHeight);

}