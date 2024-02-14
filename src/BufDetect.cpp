#include "runec/BufDetect.hpp"
#include "runec/testImageSubscriber.hpp"
//#define DBG_IMSHOW cv::imshow
#define DBG_IMSHOW //
cv::Mat BufDetect::drawBbox(const cv::Mat& input, std::vector<Detection>& res,std::map<int, std::string>& Labels){
  cv::Mat img=input.clone();
for (size_t j = 0; j < res.size(); j++) {
		float l = res[j].bbox[0];
		float t = res[j].bbox[1];
		float r = res[j].bbox[2];
		float b = res[j].bbox[3];
		cv::Rect rect = cv::Rect(int(l), int(t), int(r - l), int(b - t));
		std::string name = Labels[(int)res[j].class_id];
		cv::rectangle(img, rect, cv::Scalar(0xFF, 0xFF, 0), 2);
		cv::putText(img, name, cv::Point(rect.x, rect.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0), 2);
	}
  return img;
}
cv::Mat BufDetect::delCircle(const cv::Mat & input, int radius)
{
  cv::Point center(input.cols / 2, input.rows / 2);

  // ָ���뾶

  // ��������
  cv::Mat mask = cv::Mat::ones(input.size(), input.type()) * 255;

  // ���ƺ�ɫԲ��
  cv::circle(mask, center, radius, cv::Scalar(0, 0, 0), -1);

  // Ӧ������
  cv::Mat result;
  cv::bitwise_and(input, mask, result);
  return result;
}
void BufDetect::setParam(YoloTRT* OuterYolo,int OuterTargetColor){
this->yolo=OuterYolo;
this->targetColor=OuterTargetColor;
}
BufDetect::BufDetect(ImageSubscriber* Outerfather,YoloTRT * OuterYolo = nullptr, int OuterTargetColor = -1):roslogger(Outerfather->roslogger)
{
  this->father=Outerfather;
  //this->roslogger=this->father->roslogger;
  this->yolo = OuterYolo;
  this->targetColor = OuterTargetColor;

  this->labels=this->yolo->getlabels();
}
int BufDetect::overlap(const Detection & bb1, const Detection & bb2)
{
  int l1 = bb1.bbox[0], t1 = bb1.bbox[1], r1 = bb1.bbox[2], b1 = bb1.bbox[3];
  int l2 = bb2.bbox[0], t2 = bb2.bbox[1], r2 = bb2.bbox[2], b2 = bb2.bbox[3];
  //����ˮƽ�ʹ�ֱ�����ϵ��ص�����
  int overlap_x = std::max(0, min(r1, r2) - max(l1, l2));
  int overlap_y = std::max(0, min(b1, b2) - max(t1, t2));

  //�����ص����
  int area = overlap_x * overlap_y;

  return area;
}
int BufDetect::red_or_blue2(const cv::Mat & input, int l, int t, int r, int b) //1红0蓝
//提取小图
{
  cv::Rect roi(l, t, (r - l), (b - t));
  cv::Mat image(input, roi);
  cv::Mat hsvimage;
  cv::cvtColor(image, hsvimage, cv::COLOR_BGR2HSV);      //必须在HSV下

  // 定义红色和蓝色的范围
  cv::Scalar lowerRed1(0, 50, 50), upperRed1(10, 255, 255);
  cv::Scalar lowerRed2(170, 50, 50), upperRed2(180, 250, 255);
  cv::Scalar lowerBlue(100, 50, 50), upperBlue(130, 255, 255);

  cv::Mat redMask1, redMask2, blueMask;
  cv::Mat redMask;

  // 提取红色和蓝色区域
  cv::inRange(hsvimage, lowerRed1, upperRed1, redMask1);
  cv::inRange(hsvimage, lowerRed2, upperRed2, redMask2);
  cv::inRange(hsvimage, lowerBlue, upperBlue, blueMask);

  cv::bitwise_or(redMask1, redMask2, redMask);
  int redArea = cv::countNonZero(redMask);
  int blueArea = cv::countNonZero(blueMask);

  //std::cout << "红色区域面积: " << redArea << ", 蓝色区域面积: " << blueArea << std::endl;
  if(subscribers_info.empty())//没被订阅时相当于debug
  RCLCPP_INFO(this->roslogger,"红色区域面积:%d,蓝色区域面积:%d",redArea,blueArea);

  if (redArea > blueArea) {
    //std::cout << "区域偏红" << std::endl;
    if(subscribers_info.empty())//没被订阅时相当于debug
    RCLCPP_INFO(this->roslogger,"区域偏红");
    return 1;
  } else {
    //std::cout << "区域偏蓝" << std::endl;
    if(subscribers_info.empty())//没被订阅时相当于debug
    RCLCPP_INFO(this->roslogger,"区域偏蓝");
    return 0;
  }
  return 3;
}
void BufDetect::newDetect(const cv::Mat & input, std::vector<cv::Point> & ansList)
{
  subscribers_info = this->father->get_subscriptions_info_by_topic("detection");
  ansList.clear();
  res.clear();
  this->yolo->detect(input, res);

  RList.clear(); BigList.clear(); LittleList.clear();
  for (Detection & result : res) {
    int cls = result.class_id;
    int l, t, r, b;
    l = result.bbox[0], t = result.bbox[1], r = result.bbox[2], b = result.bbox[3];
    if (this->red_or_blue2(input, l, t, r, b) != this->targetColor) {continue;}
    //	std::cout << "debug::this cls is " << cls << "\n";
    if (cls == 0) {
      RList.push_back(result);
    } else if (cls == 1) {
      BigList.push_back(result);
    } else if (cls == 2) {
      LittleList.push_back(result);
    }
  }

  RightBigList.clear(); RightLittleList.clear();
  for (Detection & LittleItem : LittleList) {
    for (Detection & BigItem : BigList) {
      int width = LittleItem.bbox[2] - LittleItem.bbox[0],
        height = LittleItem.bbox[3] - LittleItem.bbox[1];
      if (int(width * height) == 0) {continue;}
      int same_area = this->overlap(LittleItem, BigItem);
      double same_rate = same_area * 1.0 / (width * height);
      if (same_rate > 0.8) {                //��Ϊ��ȷ��LittleRoi
        RightBigList.push_back(BigItem);
        RightLittleList.push_back(LittleItem);
        break;
      }
    }
  }
  //发送颜色相同且重叠合适的绘制结果
if(subscribers_info.empty())
{
  std::vector<Detection>same_color_overVec;
  for(Detection&result:RList)same_color_overVec.push_back(result);
  for(Detection&result:RightLittleList)same_color_overVec.push_back(result);
  for(Detection&result:RightBigList)same_color_overVec.push_back(result);
  cv::Mat same_color_overImg=this->drawBbox(input,same_color_overVec,this->labels);

  this->father->DebugSender(same_color_overImg,"same_color_overImg");
}

  //std::cout << "debug::prepare to use the R and little img,RList.size() is " << RList.size() << "\n";
  for (Detection & RItem : RList) {
    cv::Point May_R((int)((RItem.bbox[0] + RItem.bbox[2]) / 2.0),
      (int)((RItem.bbox[1] + RItem.bbox[3]) / 2.0));
    for (int i = 0; i < RightBigList.size(); i++) {
      Detection BigItem = RightBigList[i], LittleItem = RightLittleList[i];
      cv::Point May_Big(int((BigItem.bbox[0] + BigItem.bbox[2]) / 2.0),
        int((BigItem.bbox[1] + BigItem.bbox[3]) / 2.0));
      double dis = this->get_distance(May_R, May_Big);
      //std::cout << "first data is " << dis << ",second is " << min(BigItem.bbox[2] - BigItem.bbox[0], BigItem.bbox[3] - BigItem.bbox[1]) * 1.2 << "\n";
      if (dis < min(BigItem.bbox[2] - BigItem.bbox[0], BigItem.bbox[3] - BigItem.bbox[1]) * 1.7) {
        //std::cout << "debug::do we enter in the preparing to getNeedHit2?\n";
        //����Сͼѡ��rect
        cv::Rect roi(int(LittleItem.bbox[0]), int(LittleItem.bbox[1]),
          int(LittleItem.bbox[2] - LittleItem.bbox[0]),
          int(LittleItem.bbox[3] - LittleItem.bbox[1]));
        //std::cout << "debug::" << roi << "\n";
        cv::Mat image2(input, roi);
        //printf("we have trying getNeedPoint\n")
        //ע�⣬��������ϵ�任��RItemӦ�ü�һ��

        cv::Point May_R2(May_R.x - LittleItem.bbox[0], May_R.y - LittleItem.bbox[1]);
        tempans.clear();
        this->getNeedHit2(image2, May_R2, tempans);
        //std::cout << "debug::tempans.size is " << tempans.size() << "\n";
        if (tempans.size() <= 0) {continue;}
        //tempansֱ�ӷ��ص�����ϵ����Сͼ���Ͻ�Ϊ��0��0���ģ���Ҫת��
        for (cv::Point tempItem : tempans) {
          ansList.push_back(
            cv::Point(
              tempItem.x + LittleItem.bbox[0],
              tempItem.y + LittleItem.bbox[1]));
        }
        if (ansList.size() >= 4) {
          ansList.push_back(May_R);
        }                        //加入R
        return;
      }
    }
  }
}


double BufDetect::getSameArea(cv::RotatedRect rect1, cv::RotatedRect rect2)
{
  cv::Point2f points1[4], points2[4];
  rect1.points(points1);
  rect2.points(points2);

  // ת��ΪPoint����
  std::vector<cv::Point> poly1, poly2;
  for (int i = 0; i < 4; i++) {
    poly1.push_back(points1[i]);
    poly2.push_back(points2[i]);
  }

  // ������������εĽ���
  std::vector<cv::Point> intersection;
  cv::intersectConvexConvex(poly1, poly2, intersection);

  // ����н���������������
  if (!intersection.empty()) {
    return cv::contourArea(intersection);
  }

  return 0.0f;       // �޽���
}
std::vector<cv::Point> BufDetect::getSamePoint(
  cv::RotatedRect rect1, cv::RotatedRect rect2,
  cv::Point2d MayR)
{
  cv::Point samePoint1(0, 0), samePoint2(0, 0);
  cv::Point2f vertices1f[4], vertices2f[4];
  rect1.points(vertices1f);
  rect2.points(vertices2f);
  cv::Point vertices1[4], vertices2[4];
  for (int i = 0; i < 4; i++) {
    cv::Point tp;
    tp.x = (int)vertices1f[i].x;
    tp.y = (int)vertices1f[i].y;
    vertices1[i] = tp;
  }
  for (int i = 0; i < 4; i++) {
    cv::Point tp;
    tp.x = (int)vertices2f[i].x;
    tp.y = (int)vertices2f[i].y;
    vertices2[i] = tp;
  }


  // ������ε����ĵ�
  cv::Point2f center1f = rect1.center;
  cv::Point2f center2f = rect2.center;
  cv::Point center1, center2;
  center1.x = (int)center1f.x, center1.y = (int)center1f.y;
  center2.x = (int)center2f.x, center2.y = (int)center2f.y;
  std::vector<cv::Point> innerPoints, inner1, inner2;

  for (int i = 0; i < 4; i++) {
    inner1.push_back(vertices1[i]);
    inner2.push_back(vertices2[i]);
  }
  for (int i = 0; i < 4; i++) {
    for (int j = i; j < 4; j++) {
      if (this->get_distance(inner1[j], center2) <= this->get_distance(inner1[i], center2)) {
        swap(inner1[i], inner1[j]);
      }
    }
  }
  for (int i = 0; i < 4; i++) {
    for (int j = i; j < 4; j++) {
      if (this->get_distance(inner2[j], center1) <= this->get_distance(inner2[i], center1)) {
        swap(inner2[i], inner2[j]);
      }
    }
  }
  for (int i = 0; i <= 1; i++) {
    innerPoints.push_back(inner1[i]);
  }
  for (int i = 0; i <= 1; i++) {
    innerPoints.push_back(inner2[i]);
  }
  //innerPoints.push_back(center1);
  //innerPoints.push_back(center2);
  //  cv::Moments m = cv::moments(innerPoints);
  // cv::Point centroid = cv::Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
  std::vector<cv::Point> result;
  if (innerPoints.size() == 0) {return result;}
  cv::Point centroid;
  for (int i = 0; i < innerPoints.size(); i++) {
    centroid.x += innerPoints[i].x, centroid.y += innerPoints[i].y;
  }
  centroid.x /= innerPoints.size(), centroid.y /= innerPoints.size();

  result.clear();
  //result.push_back(centroid);

  cv::Point targetVec;
  cv::Point micVec;
  double Rx = MayR.x * 1.0, Ry = MayR.y * 1.0;
  double focusX = centroid.x, focusY = centroid.y;
  micVec.x = (int)(focusX - Rx), micVec.y = (int)(focusY - Ry);
  targetVec.x = (int)(inner1[0].x - Rx), targetVec.y = (int)(inner1[0].y - Ry);
  if (micVec.x * targetVec.y - micVec.y * targetVec.x >= 0) {
    result.push_back(inner1[0]);
    result.push_back(inner1[1]);
  } else {
    result.push_back(inner1[1]);
    result.push_back(inner1[0]);
  }
  targetVec.x = (int)(inner2[0].x - Rx), targetVec.y = (int)(inner2[0].y - Ry);
  if (micVec.x * targetVec.y - micVec.y * targetVec.x >= 0) {
    result.push_back(inner2[0]);
    result.push_back(inner2[1]);
  } else {
    result.push_back(inner2[1]);
    result.push_back(inner2[0]);
  }
  std::vector<cv::Point> return_result;
  return_result.clear();
  return_result.push_back(result[2]);      //外右
  return_result.push_back(result[3]);      //外左
  return_result.push_back(result[1]);      //内左
  return_result.push_back(result[0]);      //内右
  return return_result;
}
bool BufDetect::getNeedHit2(const cv::Mat & input, cv::Point2d MayR, std::vector<cv::Point> & ans)//�����ڲ��R���⿴���ң���;���������ң���
//新版：外右，外左，内左，内右
{

  ans.clear();
  cv::Mat video1 = input.clone();

  if(subscribers_info.empty())//显示小图
{
  cv::Mat littleImg=video1.clone();
  this->father->DebugSender(littleImg,"littleImg");
  //cv::imshow("littleimg", video1);
}
  double alpha = 0.6;       // ���ƶԱȶ�
  int beta = -20;       // ��������
  // Bupt2HSV(video1);


  // �������ȺͶԱȶ�
  for (int y = 0; y < input.rows; y++) {
    for (int x = 0; x < input.cols; x++) {
      for (int c = 0; c < 3; c++) {
        video1.at<cv::Vec3b>(y, x)[c] =
          cv::saturate_cast<uchar>(alpha * input.at<cv::Vec3b>(y, x)[c] + beta);
      }
    }
  }
  //DBG_IMSHOW("low_roi",video1);
  //cv::imshow("cvt_roi", video1);


  cv::Mat gray;
  cv::Mat gray0, canny0;
  cv::Mat Show = input.clone();
  cv::cvtColor(video1, gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(input, gray0, cv::COLOR_BGR2GRAY);
  cv::Scalar color1(0, 255, 0), color0(0, 0, 255);
  //ʹ����ֵ��
  cv::Mat gray_thresh, gray_thresh2, gray_bit;
  cv::threshold(gray, gray_thresh2, 2, 255, cv::THRESH_BINARY);

  //cv::Canny(gray, canny0, 30, 120,3);
  //cv::threshold(gray, gray_thresh, thresh3, 255, cv::THRESH_BINARY);//��ͨ��ֵ��
  cv::threshold(gray, gray_thresh, 0, 255, cv::THRESH_OTSU);      //���
  //cv::threshold(canny0, gray_thresh, thresh3, 255, cv::THRESH_BINARY);
  gray_bit = gray_thresh.clone();
  gray_bit = this->delCircle(gray_bit, std::max(10, std::min(gray_bit.cols, gray_bit.rows) / 3));
  erode(gray_bit, gray_bit, kernel1);
  //erode(gray_bit, gray_bit, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));
  if(subscribers_info.empty())//发送阈值化且删除圆的图片
  {
    cv::Mat thresh_delImg=gray_bit.clone();
    this->father->DebugSender(thresh_delImg,"thresh_delImg");
    //cv::imshow("gray_bit", gray_bit);
  }
  //gray_bit = canny0 - gray_bit;
  //cv::subtract(canny0, gray_bit, gray_bit);
  //cv::erode(gray_bit, gray_bit, kernel1);
  //cv::dilate(gray_bit, gray_bit, kernel2);
  //morphologyEx(gray_bit, gray_bit, MORPH_CLOSE, kernel1);
  //cv::bitwise_xor(gray, gray_thresh, gray_bit);
 // DBG_IMSHOW("circle_gray_bit", gray_bit);
//  DBG_IMSHOW("canny0", canny0);

  // ʹ�ø�˹ģ������ͼ������
  // cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);


  // �洢��⵽��Բ
  std::vector<cv::Vec3f> circles;
  /*
  // Ӧ��HoughԲ�任
  cv::HoughCircles(canny0, circles, cv::HOUGH_GRADIENT, 1, canny0.rows /16, 120, 40, 1,50);
vector<vector<cv::Point>>countors;
  // ���Ƽ�⵽��Բ
  for (size_t i = 0; i < circles.size(); i++) {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // ����Բ��
        cv::circle(Show, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
        // ����Բ����
        cv::circle(Show, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
        std::wcout << "���ڻ���" << i + 1 << "��Բ\n";
  }
  DBG_IMSHOW("��Բ", Show);
  */
  //  cv::threshold(gray, gray, 150,255,cv::THRESH_BINARY);
  // cv::findContours(gray, countors, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
  //drawContours(mask, results, -1, Scalar(255, 0, 0), 2);
  //cv::drawContours(Show, countors, -1, Scalar(0, 255, 0), 2);


  /*
  //ʹ�ø��׵ĸ������м��
    //cv::Canny(gray, gray, 100,255);
    std::vector<cv::Vec4i> hierarchy;
    std::vector<cv::Point>mayBeResults;
    cv::findContours(canny0, countors, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    int max_id = -1,max_num=-1;
    for (int i = 0; i < countors.size();i++) {
          int id=i;
          int num = 1;
          if (hierarchy[id][2] != -1)continue;
          while (hierarchy[id][3]!=-1) {
                  num++;
                  id = hierarchy[id][3];
          }
          if (num > max_num) {
                  max_num = num;
                  max_id = i;
          }
          if (num >= 2) {
                  Rect boundRect = boundingRect(countors[i]);
                  mayBeResults.push_back((boundRect.br()+boundRect.tl())/2);
          }
    }
    std::cout << "max_id=" << max_id << "\n";
    if (max_id == -1)return false;
    vector<vector<cv::Point>>results;

    results.push_back(countors[max_id]);

    for (cv::Point p : mayBeResults) {
          if (this->targetColor == 1)cv::circle(Show, cv::Point(p.x,p.y), 10, color1, 3, 2, 0);
          else cv::circle(Show, cv::Point(p.x, p.y), 10, color0, 3, 1, 0);
    }

   */

  //ʹ���ܳ����ɸ��
  std::vector<std::vector<cv::Point>> countors;
  cv::findContours(gray_bit, countors, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);


  double max_perimeter1 = -1, max_perimeter2 = -1;
  int max_id1 = -1, max_id2 = -1;
  for (int i = 0; i < countors.size(); i++) {
    vector<cv::Point> countor = countors[i];
    double contour_area = cv::contourArea(countor);
    double contour_perimeter = cv::arcLength(countor, true);
    cv::RotatedRect rect = cv::minAreaRect(countor);
    double rate = contour_area / (rect.size.height * rect.size.width);
    if (abs(rect.size.width - rect.size.height) < 6) {continue;}
    if (rate > 0.7 || rate < 0.1) {continue;}
    if (contour_perimeter >= max_perimeter1) {
      max_id1 = i;
      max_perimeter1 = contour_perimeter;
    }
  }

  /*std::cout << "debug::����\n";
  cv::Mat drawImage=input.clone();
  cv::drawContours(drawImage, countors, 0, cv::Scalar(0, 0, 255),3);
  cv::imshow("����", drawImage);
  std::cout << "debug::max_id1=" << max_id1 << "\n";*/
  if (max_id1 == -1) {return false;}
  cv::RotatedRect rect1 = cv::minAreaRect(countors[max_id1]);
  for (int i = 0; i < countors.size(); i++) {
    vector<cv::Point> countor = countors[i];
    double contour_area = cv::contourArea(countor);
    double contour_perimeter = cv::arcLength(countor, true);
    cv::RotatedRect rect = cv::minAreaRect(countor);
    double rate = contour_area / (rect.size.height * rect.size.width);
    //if (abs(rect.size.width - rect.size.height) < 6)continue;
    cv::Point center1, center2;
    center1.x = (int)rect1.center.x;
    center1.y = (int)rect1.center.y;
    center2.x = (int)rect.center.x;
    center2.y = (int)rect.center.y;
    double littledis = this->get_distance(center1, center2);
    //if (littledis < max(min(rect1.size.width, rect1.size.height), min(rect.size.width, rect.size.height)) * 0.8)continue;
    //if (this->getSameArea(rect1, rect) >= 23)continue;
    //if (rate > 0.7 || rate < 0.1)continue;
    if (contour_perimeter > max_perimeter2 && contour_perimeter < max_perimeter1) {
      max_id2 = i;
      max_perimeter2 = contour_perimeter;
    }
  }
  //std::cout << "debug::max_id1=" << max_id1 << ",max_id2=" << max_id2 << "\n";
  if (max_id1 == -1 || max_id2 == -1) {return false;}
  cv::Point2f point1[4], point2[4];

  cv::RotatedRect rect2 = cv::minAreaRect(countors[max_id2]);
  //ϣ��������Ը
  //rect1.angle = rect2.angle;

  rect1.points(point1);
  rect2.points(point2);
  cv::Point pint1[4], pint2[4];
  for (int i = 0; i < 4; i++) {
    cv::Point tp;
    tp.x = (int)(point1[i].x);
    tp.y = (int)(point1[i].y);
    pint1[i] = tp;
  }
  for (int i = 0; i < 4; i++) {
    cv::Point tp;
    tp.x = (int)(point2[i].x);
    tp.y = (int)(point2[i].y);
    pint2[i] = tp;
  }
  for (int i = 0; i < 4; i++) {
    if (this->targetColor == 1) {
      cv::line(Show, pint1[i], pint1[(i + 1) % 4], color1, 2, 2, 0);
    } else {cv::line(Show, pint1[i], pint1[(i + 1) % 4], color0, 2, 2, 0);}
  }
  for (int i = 0; i < 4; i++) {
    if (this->targetColor == 1) {
      cv::line(Show, pint2[i], pint2[(i + 1) % 4], color1, 2, 2, 0);
    } else {cv::line(Show, pint2[i], pint2[(i + 1) % 4], color0, 2, 2, 0);}
  }
  std::vector<cv::Point> MayBeCenters = this->getSamePoint(rect1, rect2, MayR);
  ans.clear();
  for (cv::Point MayBeCenter : MayBeCenters) {    //ע�⣬����Сͼ������ϵ
    //std::cout <<"debug::" << MayBeCenter << "\n";
    if (this->targetColor == 1) {
      cv::circle(Show, MayBeCenter, 3, cv::Scalar(255, 255, 255), 2, 2, 0);
    } else {cv::circle(Show, MayBeCenter, 3, cv::Scalar(255, 255, 255), 2, 2, 0);}
    ans.push_back(MayBeCenter);
  }


  //DBG_IMSHOW("��Բ", Show);
  cv::imshow("��Ժ", Show);
  return ans.size() > 0;
}

void BufDetect::checkpicture()
{
  std::string folderPath = "F:\\rm2023\\��������\\data\\images";       // �滻Ϊ���ͼƬ�ļ���·��

  // C++17���·���Ƿ���ں��Ƿ�ΪĿ¼
  if (!std::filesystem::exists(folderPath) || !std::filesystem::is_directory(folderPath)) {
    std::cout << "Directory path is not valid." << std::endl;
    return;
  }

  // ����ָ��Ŀ¼�е������ļ�����������Ŀ¼��
  for (const auto & entry : filesystem::directory_iterator(folderPath)) {
    if (entry.is_regular_file()) {
      // ��ȡ�ļ�·��
      auto filePath = entry.path();
      // ��ȡ�ļ���չ����ת��ΪСд��
      std::string extension = filePath.extension().string();
      std::transform(
        extension.begin(), extension.end(), extension.begin(),
        [](unsigned char c) {return std::tolower(c);});

      // �����ͼƬ�ļ����������ӵ��б���
      if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
        extension == ".bmp")
      {
        this->imgPaths.push_back(filePath.string());
      }
    }
  }

  // ��������ҵ���ͼƬ·��
  for (const auto & imagePath : this->imgPaths) {
    std::cout << imagePath << std::endl;
  }

}
double BufDetect::get_distance(cv::Point p1, cv::Point p2)
{
  return sqrtf(powf((p1.x - p2.x), 2) + powf((p1.y - p2.y), 2));
}
