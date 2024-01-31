#pragma once
#ifndef BUFDETECT_HPP
#define BUFDETECT_HPP
#include "YoloTRT.h"
#include <filesystem>
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
struct Bufarmor_point
{
  cv::Point2f point;
  double dis;
};
class ImageSubscriber;
class BufDetect
{
public:
  void setParam(YoloTRT* OuterYolo,int OuterTargetColor);
  BufDetect(ImageSubscriber* Outerfather,YoloTRT * OuterYolo, int OuterTargetColor);
  void newDetect(const cv::Mat & input, std::vector<cv::Point> & ansList);
  rclcpp::Logger roslogger;
  ImageSubscriber* father;
private:

  
  std::map<int,std::string>labels;
  cv::Mat drawBbox(const cv::Mat& input, std::vector<Detection>& res,std::map<int, std::string>& Labels);
  std::vector<rclcpp::TopicEndpointInfo>subscribers_info;
  cv::Mat delCircle(const cv::Mat & input, int raidus);
  YoloTRT * yolo = nullptr;
  std::vector<Detection> res;
  std::vector<Detection> RList, BigList, LittleList;     //��ɫ��ͬ��
  std::vector<Detection> RightBigList, RightLittleList;
  std::vector<cv::Point> tempans;
  int overlap(const Detection & bb1, const Detection & bb2);

  int red_or_blue2(const cv::Mat & input, int l, int t, int r, int b);

  //Bufarmor_point armor_point;
  cv::Mat img1;      //��debug��
  double getSameArea(cv::RotatedRect rect1, cv::RotatedRect rect2);
  std::vector<cv::Point> getSamePoint(
    cv::RotatedRect rect1, cv::RotatedRect rect2,
    cv::Point2d MayR);
  Point2d box_center = Point2d(0, 0);
  void checkpicture();
  std::vector<std::string> imgPaths;
  std::vector<vector<vector<cv::Point>>> template_R_countours;
  std::vector<cv::Mat> template_R_gray;
  double get_distance(cv::Point p1, cv::Point p2);
  bool getNeedHit2(const cv::Mat & input, cv::Point2d MayR, std::vector<cv::Point> & ans);
  int imgid = 0;
  int thresh1 = 7, thresh2 = 3, thresh3 = 76;
  double dot_thresh = 0.5;
  cv::Mat kernel1 = getStructuringElement(MORPH_RECT, Size(3, 3));      //�������ں�

  cv::Mat kernel2 = getStructuringElement(MORPH_RECT, Size(7, 7));      //�������ں�
  int targetColor = 1;    //1Ϊ�죬0Ϊ��
  int ROI_radius = 50;
  int near_thresh = 70;
  int last_targetColor = -1;
  int last_imgid = -1;
  int last_thresh1 = -1, last_thresh2 = -1, last_thresh3 = -1;
  int last_near_thresh = -1;
};
#endif //WIND_INFERENCE_INFERENCE_H
