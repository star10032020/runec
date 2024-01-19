
#pragma once
#ifndef BUFDETECT_HPP
#define BUFDETECT_HPP
#include "YoloTRT.h"
#include <filesystem>
#include<iostream>
#include<vector>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
struct Bufarmor_point {
	cv::Point2f point;
	double dis;
};
class BufDetect {
	public:
	BufDetect();
	BufDetect(YoloTRT* OuterYolo,int OuterTargetColor);
	void newDetect(const cv::Mat &input,std::vector<cv::Point>&ansList);
	std::vector<cv::Point>  Detect(cv::Mat input);
	void guidetect();
private:
	cv::Mat delCircle(const cv::Mat& input, int raidus);
	YoloTRT *yolo=nullptr;
	std::vector<Detection>res;
	std::vector<Detection>RList, BigList, LittleList;//颜色相同的
	std::vector<Detection>RightBigList, RightLittleList;
	std::vector<cv::Point>tempans;
	int overlap(const Detection &bb1, const Detection &bb2);

	int red_or_blue2(const cv::Mat &input,int l,int t,int r,int b);

	//Bufarmor_point armor_point;
	cv::Mat img1;//纯debug用
	double getSameArea(cv::RotatedRect rect1, cv::RotatedRect rect2);
	std::vector <cv::Point> getSamePoint(cv::RotatedRect rect1, cv::RotatedRect rect2,cv::Point2d MayR);
	cv::Point2f computeCentroid(const std::vector<cv::Point>& contour);//计算质心
	cv::Point2f normalizedVector(const cv::Point2f& start, const cv::Point2f& end);//归一化
	double isPointingTowards(const cv::Point2f& vectorA, const cv::Point2f& vectorB);//夹角余弦值
	cv::Mat Bupt2HSV(cv::Mat input);
	bool red_or_blue(cv::Mat input, vector<cv::Point>contour);
	cv::Point findDenseAreaAndSetROI(cv::Mat inputImage);
	double nearCenter(vector<cv::Point> contour);
	double R_Same(cv::Mat img1);
	double R_Same2(cv::Mat img1);
	Point2d box_center = Point2d(0, 0);
	void checkpicture();
	void checkRpicture();
	std::vector<std::string>imgPaths;
	std::vector<vector<vector<cv::Point>>>template_R_countours;
	std::vector<cv::Mat>template_R_gray;
	double get_distance(cv::Point p1,cv::Point p2);
	std::vector<double> calculateHuMoments(const std::vector<cv::Point>& contour);
	bool getNeedHit2(const cv::Mat &input,cv::Point2d MayR, std::vector<cv::Point>& ans);
	bool getNeedHit(vector<vector<cv::Point>>contours,vector<cv::Point>& targetcontour,cv::Point R_center);
	vector<vector<cv::Point>>getContours(vector<vector<cv::Point>>con1, cv::Point R_center,int R_Size);
	int imgid=0;
	int thresh1=7,thresh2=3,thresh3=76;
	double dot_thresh = 0.5;
	cv::Mat kernel1 = getStructuringElement(MORPH_RECT, Size(3, 3));//闭运算内核

	cv::Mat kernel2 = getStructuringElement(MORPH_RECT, Size(7, 7));//闭运算内核
	int targetColor=1;//1为红，0为蓝
	int ROI_radius = 50;
	int near_thresh=70;
	int last_targetColor=-1;
	int last_imgid = -1;
	int last_thresh1=-1, last_thresh2=-1,last_thresh3=-1;
	int last_near_thresh=-1;
};
#endif //WIND_INFERENCE_INFERENCE_H