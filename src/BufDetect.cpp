#include"runec/BufDetect.hpp"
//#define DBG_IMSHOW cv::imshow
#define DBG_IMSHOW //
cv::Mat BufDetect::delCircle(const cv::Mat& input, int radius) {
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
BufDetect::BufDetect(YoloTRT* OuterYolo=nullptr,int OuterTargetColor=-1){
	this->yolo=OuterYolo;
	this->targetColor=OuterTargetColor;
}
int BufDetect::overlap(const Detection& bb1, const Detection& bb2) {
	int l1 = bb1.bbox[0], t1 = bb1.bbox[1], r1 = bb1.bbox[2], b1 = bb1.bbox[3];
	int l2 = bb2.bbox[0], t2 = bb2.bbox[1], r2 = bb2.bbox[2], b2 = bb2.bbox[3];
	//����ˮƽ�ʹ�ֱ�����ϵ��ص�����
	int overlap_x = std::max(0, min(r1, r2) - max(l1, l2));
	int overlap_y = std::max(0, min(b1, b2) - max(t1, t2));

	//�����ص����
	int area = overlap_x * overlap_y;

	return area;
}
int BufDetect::red_or_blue2(const cv::Mat& input, int l, int t, int r, int b) {//1��0��
	//��ȡСͼ
	cv::Rect roi(l, t, (r - l), (b - t));
	cv::Mat image(input, roi);

	// �����ɫ����ɫ�ķ�Χ
	cv::Scalar lowerRed(0, 0, 200), upperRed(50, 50, 255);
	cv::Scalar lowerBlue(200, 0, 0), upperBlue(255, 50, 50);

	cv::Mat redMask, blueMask;

	// ��ȡ��ɫ����ɫ����
	cv::inRange(image, lowerRed, upperRed, redMask);
	cv::inRange(image, lowerBlue, upperBlue, blueMask);

	int redArea = cv::countNonZero(redMask);
	int blueArea = cv::countNonZero(blueMask);

	//std::cout << "��ɫ�������: " << redArea << ", ��ɫ�������: " << blueArea << std::endl;

	if (redArea > blueArea) {
		//std::cout << "����ƫ��" << std::endl;
		return 1;
	}
	else {
		return 0;
		//std::cout << "����ƫ��" << std::endl;
	}
	return 3;
}
void BufDetect::newDetect(const cv::Mat& input, std::vector<cv::Point>&ansList) {
	ansList.clear();
	res.clear();
	this->yolo->detect(input,res);

	RList.clear(); BigList.clear(); LittleList.clear();
	for (Detection& result : res) {
		int cls = result.class_id;
		int l, t, r, b;
		l = result.bbox[0], t = result.bbox[1], r = result.bbox[2], b = result.bbox[3];
		if (this->red_or_blue2(input, l, t, r, b) != this->targetColor)continue;
	//	std::cout << "debug::this cls is " << cls << "\n";
		if (cls == 0) {
			RList.push_back(result);
		}
		else
			if (cls == 1) {
				BigList.push_back(result);
			}
			else
				if (cls == 2) {
					LittleList.push_back(result);
				}
	}
	RightBigList.clear(); RightLittleList.clear();
	for (Detection& LittleItem : LittleList) {
		for (Detection& BigItem : BigList) {
			int width = LittleItem.bbox[2] - LittleItem.bbox[0], height = LittleItem.bbox[3] - LittleItem.bbox[1];
			if (int(width * height) == 0)continue;
			int same_area = this->overlap(LittleItem, BigItem);
			double same_rate = same_area * 1.0 / (width * height);
			if (same_rate > 0.8)//��Ϊ��ȷ��LittleRoi
			{
				RightBigList.push_back(BigItem);
				RightLittleList.push_back(LittleItem);
				break;
			}
		}
	}
	//std::cout << "debug::prepare to use the R and little img,RList.size() is " << RList.size() << "\n";
	for (Detection& RItem : RList) {
		cv::Point May_R(int((RItem.bbox[0] + RItem.bbox[2]) / 2.0), int((RItem.bbox[1] + RItem.bbox[3]) / 2.0));
		for (int i = 0; i < RightBigList.size(); i++)
		{
			Detection BigItem = RightBigList[i], LittleItem = RightLittleList[i];
			cv::Point May_Big(int((BigItem.bbox[0] + BigItem.bbox[2]) / 2.0), int((BigItem.bbox[1] + BigItem.bbox[3]) / 2.0));
			double dis = this->get_distance(May_R, May_Big);
			//std::cout << "first data is " << dis << ",second is " << min(BigItem.bbox[2] - BigItem.bbox[0], BigItem.bbox[3] - BigItem.bbox[1]) * 1.2 << "\n";
			if (dis < min(BigItem.bbox[2] - BigItem.bbox[0], BigItem.bbox[3] - BigItem.bbox[1]) * 1.7)
			{
				//std::cout << "debug::do we enter in the preparing to getNeedHit2?\n";
				//����Сͼѡ��rect
				cv::Rect roi(int(LittleItem.bbox[0]), int(LittleItem.bbox[1]),
					int(LittleItem.bbox[2] - LittleItem.bbox[0]), int(LittleItem.bbox[3] - LittleItem.bbox[1]));
				//std::cout << "debug::" << roi << "\n";
				cv::Mat image2(input, roi);
				//printf("we have trying getNeedPoint\n")
				//ע�⣬��������ϵ�任��RItemӦ�ü�һ��

				cv::Point May_R2(May_R.x - LittleItem.bbox[0], May_R.y - LittleItem.bbox[1]);
				tempans.clear();
				this->getNeedHit2(image2, May_R2, tempans);
				//std::cout << "debug::tempans.size is " << tempans.size() << "\n";
				if (tempans.size() <= 0)continue;
				//tempansֱ�ӷ��ص�����ϵ����Сͼ���Ͻ�Ϊ��0��0���ģ���Ҫת��
				for (cv::Point tempItem : tempans) {
					ansList.push_back(cv::Point(tempItem.x+LittleItem.bbox[0],tempItem.y+LittleItem.bbox[1]));
				}
				return;
			}
		}
	}
}



double BufDetect::getSameArea(cv::RotatedRect rect1, cv::RotatedRect rect2) {
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

	return 0.0f; // �޽���
}
double BufDetect::isPointingTowards(const cv::Point2f& vectorA, const cv::Point2f& vectorB) {
	cv::Point2f normA = vectorA / cv::norm(vectorA);
	cv::Point2f normB = vectorB / cv::norm(vectorB);

	// �������������ĵ��
	float dot = normA.x * normB.x + normA.y * normB.y;
	return dot;
}
cv::Point2f BufDetect::computeCentroid(const std::vector<cv::Point>& contour) {
	cv::Moments m = cv::moments(contour, true);
	return cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
}
cv::Point2f BufDetect::normalizedVector(const cv::Point2f& start, const cv::Point2f& end) {
	cv::Point2f vec = end - start;
	float magnitude = std::sqrt(vec.x * vec.x + vec.y * vec.y);
	return vec / magnitude;
}
std::vector<cv::Point> BufDetect::getSamePoint(cv::RotatedRect rect1, cv::RotatedRect rect2, cv::Point2d MayR) {
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
	for (int i = 0; i < 4; i++)
	{
		for (int j = i; j < 4; j++) {
			if (this->get_distance(inner1[j], center2) <= this->get_distance(inner1[i], center2)) {
				swap(inner1[i], inner1[j]);
			}
		}
	}
	for (int i = 0; i < 4; i++)
	{
		for (int j = i; j < 4; j++) {
			if (this->get_distance(inner2[j], center1) <= this->get_distance(inner2[i], center1)) {
				swap(inner2[i], inner2[j]);
			}
		}
	}
	for (int i = 0; i <= 1; i++)
		innerPoints.push_back(inner1[i]);
	for (int i = 0; i <= 1; i++)
		innerPoints.push_back(inner2[i]);
	//innerPoints.push_back(center1);
	//innerPoints.push_back(center2);
  //  cv::Moments m = cv::moments(innerPoints);
   // cv::Point centroid = cv::Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
	std::vector<cv::Point>result;
	if (innerPoints.size() == 0)return result;
	cv::Point centroid;
	for (int i = 0; i < innerPoints.size(); i++)
		centroid.x += innerPoints[i].x, centroid.y += innerPoints[i].y;
	centroid.x /= innerPoints.size(), centroid.y /= innerPoints.size();

	result.clear();
	//result.push_back(centroid);

	cv::Point targetVec;
	cv::Point micVec;
	double Rx = MayR.x * 1.0, Ry = MayR.y * 1.0;
	double focusX = centroid.x, focusY = centroid.y;
	micVec.x = (int)(focusX - Rx), micVec.y = (int)(focusY - Ry);
	targetVec.x = (int)(inner1[0].x - Rx), targetVec.y = (int)(inner1[0].y - Ry);
	if (micVec.x * targetVec.y - micVec.y * targetVec.x >= 0)
	{
		result.push_back(inner1[0]);
		result.push_back(inner1[1]);
	}
	else {
		result.push_back(inner1[1]);
		result.push_back(inner1[0]);
	}
	targetVec.x = (int)(inner2[0].x - Rx), targetVec.y = (int)(inner2[0].y - Ry);
	if (micVec.x * targetVec.y - micVec.y * targetVec.x >= 0)
	{
		result.push_back(inner2[0]);
		result.push_back(inner2[1]);
	}
	else {
		result.push_back(inner2[1]);
		result.push_back(inner2[0]);
	}
	return result;
}
double BufDetect::R_Same2(cv::Mat img1) {

	if (cv::sum(img1)[0] <= 1)return -30;
	cv::Mat img2;
	cv::cvtColor(img1, img2, cv::COLOR_BGR2GRAY);



	double max_value = 0;
	for (cv::Mat templ : this->template_R_gray) {
		cv::Mat result;
		cv::Mat img3;
		cv::resize(img2, img3, templ.size());

		// DBG_IMSHOW("templ", templ);
		 //DBG_IMSHOW("img3", img3);
		 //cv::waitKey(200);

		cv::matchTemplate(img3, templ, result, cv::TM_CCOEFF_NORMED);

		// ��һ���������
	  //  cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

		// ��λ���ƥ��λ��
		double minVal, maxVal;
		cv::Point minLoc, maxLoc, matchLoc;
		cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
		max_value = max(max_value, maxVal);
		// std::cout << "max_val=" << maxVal <<",min_val="<<minVal << "\n";
		 // TM_CCOEFF_NORMED ���������ƥ��λ�������ֵλ��
		matchLoc = maxLoc;
	}
	std::cout << "max_value_degree=" << max_value << "\n";
	return max_value;
	// �����ο��ƥ������
	//cv::rectangle(img, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar::all(0), 2, 8, 0);
	//cv::rectangle(result, matchLoc, cv::Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), cv::Scalar::all(0), 2, 8, 0);

}
double BufDetect::R_Same(cv::Mat img1) {

	//�����ã�ֱ�ӷ��ذ�
	return 0.2;

	cv::Mat edges1;
	cv::Mat img3;
	cv::Mat img2;
	//cv::resize(img1, img3, cv::Size(50, 50));
	img3 = img1.clone();
	cv::cvtColor(img3, img2, cv::COLOR_BGR2GRAY);
	cv::Canny(img2, edges1, 80, 255); // �������ֵ������Ҫ��������ͼ����е���
	//cv::dilate(edges1, edges1, kernel1);
	cv::Mat edges1_show;
	cv::resize(edges1, edges1_show, cv::Size(360, 360));
	DBG_IMSHOW("Same_Degree", edges1_show);
	if (cv::sum(edges1)[0] <= 1)return 1000;
	double result = 0.0;
	int sum = 0;
	std::vector<std::vector<cv::Point>> contours1;
	contours1.clear();
	cv::findContours(edges1, contours1, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	vector<cv::Point>countor_t;
	//cv::approxPolyDP(contours1[0], countor_t, 13, true);
	countor_t = contours1[0];
	for (auto contours2 : this->template_R_countours) {
		// ���ұ�Ե�����������

		//cv::findContours(edges2, contours2, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	 //  std::cout << "contour_t.size()=" << countor_t.size() << "\n";
		double this_double = cv::matchShapes(countor_t, contours2[0], cv::CONTOURS_MATCH_I1, 0);

		// double this_double = 0;
	   // std::cout << "this_double=" << this_double<<"\n";
		if (this_double > 100)continue;
		sum++;
		result += this_double;
		//result = powf(result, 1.0 / sum);
		//result = powf(result, (sum - 1));
	}
	// �Ƚ�������״������
	//double result = cv::matchShapes(contours1[0], contours2[0], cv::CONTOURS_MATCH_I1, 0);

	if (sum != 0)
		result /= sum;
	else result = 10000;
	std::cout << "imgs_same_degree=" << result << "\n";
	return result;
}
std::vector<double> BufDetect::calculateHuMoments(const std::vector<cv::Point>& contour) {
	cv::Moments moments = cv::moments(contour);
	std::vector<double> huMoments(7);
	cv::HuMoments(moments, huMoments);
	return huMoments;
}
vector<vector<cv::Point>>BufDetect::getContours(vector<vector<cv::Point>>con1, cv::Point R_center, int R_Size) {
	vector<vector<cv::Point>>results;

	for (vector<cv::Point>contour : con1) {
		Rect boundRect = boundingRect(contour);
		cv::Point bCenter = (boundRect.tl() + boundRect.br()) / 2;
		if (abs(R_center.x - bCenter.x) <= boundRect.width
			&& abs(R_center.y - bCenter.y) <= boundRect.height     //������2�����䷶Χ����һ��
			&& (boundRect.width * boundRect.height) > 600
			&& (boundRect.width * boundRect.height) > R_Size * 10
			) {
			results.push_back(contour);
		}
	}
	cv::Mat mask(this->img1.size(), CV_8UC3, Scalar(0, 0, 0));
	drawContours(mask, results, -1, Scalar(255, 0, 0), 2);
	cv::Mat mask_show;
	cv::resize(mask, mask_show, cv::Size(360, 360));
	DBG_IMSHOW("Contours", mask_show);
	return results;
}
bool BufDetect::getNeedHit2(const cv::Mat &input, cv::Point2d MayR, std::vector<cv::Point>& ans) {//�����ڲ��R���⿴���ң���;���������ң���
	ans.clear();
	cv::Mat video1 = input.clone();
	cv::imshow("littleimg", video1);
	double alpha = 0.6; // ���ƶԱȶ�
	int beta = -20; // ��������
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
	cv::threshold(gray, gray_thresh, 0, 255, cv::THRESH_OTSU);//���
	//cv::threshold(canny0, gray_thresh, thresh3, 255, cv::THRESH_BINARY);
	gray_bit = gray_thresh.clone();
	gray_bit = this->delCircle(gray_bit, std::max(10,std::min(gray_bit.cols,gray_bit.rows)/3));
	erode(gray_bit, gray_bit,kernel1);
	//erode(gray_bit, gray_bit, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));
	cv::imshow("gray_bit", gray_bit);
	//gray_bit = canny0 - gray_bit;
	//cv::subtract(canny0, gray_bit, gray_bit);
	//cv::erode(gray_bit, gray_bit, kernel1);
	//cv::dilate(gray_bit, gray_bit, kernel2);
	//morphologyEx(gray_bit, gray_bit, MORPH_CLOSE, kernel1);
	//cv::bitwise_xor(gray, gray_thresh, gray_bit);
	DBG_IMSHOW("circle_gray_bit", gray_bit);
	DBG_IMSHOW("canny0", canny0);

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
	std::vector<std::vector<cv::Point>>countors;
	cv::findContours(gray_bit, countors, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	
	double max_perimeter1 = -1, max_perimeter2 = -1;
	int max_id1 = -1, max_id2 = -1;
	for (int i = 0; i < countors.size(); i++)
	{
		vector<cv::Point>countor = countors[i];
		double contour_area = cv::contourArea(countor);
		double contour_perimeter = cv::arcLength(countor, true);
		cv::RotatedRect rect = cv::minAreaRect(countor);
		double rate = contour_area / (rect.size.height * rect.size.width);
		if (abs(rect.size.width - rect.size.height) < 6)continue;
		if (rate > 0.7 || rate < 0.1)continue;
		if (contour_perimeter >= max_perimeter1)
		{
			max_id1 = i;
			max_perimeter1 = contour_perimeter;
		}
	}

	/*std::cout << "debug::����\n";
	cv::Mat drawImage=input.clone();
	cv::drawContours(drawImage, countors, 0, cv::Scalar(0, 0, 255),3);
	cv::imshow("����", drawImage);
	std::cout << "debug::max_id1=" << max_id1 << "\n";*/
	if (max_id1 == -1)return false;
	cv::RotatedRect rect1 = cv::minAreaRect(countors[max_id1]);
	for (int i = 0; i < countors.size(); i++)
	{
		vector<cv::Point>countor = countors[i];
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
		if (contour_perimeter > max_perimeter2 && contour_perimeter < max_perimeter1)
		{
			max_id2 = i;
			max_perimeter2 = contour_perimeter;
		}
	}
	//std::cout << "debug::max_id1=" << max_id1 << ",max_id2=" << max_id2 << "\n";
	if (max_id1 == -1 || max_id2 == -1)return false;
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
		if (this->targetColor == 1)cv::line(Show, pint1[i], pint1[(i + 1) % 4], color1, 2, 2, 0);
		else cv::line(Show, pint1[i], pint1[(i + 1) % 4], color0, 2, 2, 0);
	}
	for (int i = 0; i < 4; i++) {
		if (this->targetColor == 1)cv::line(Show, pint2[i], pint2[(i + 1) % 4], color1, 2, 2, 0);
		else cv::line(Show, pint2[i], pint2[(i + 1) % 4], color0, 2, 2, 0);
	}
	std::vector<cv::Point> MayBeCenters = this->getSamePoint(rect1, rect2, MayR);
	ans.clear();
	for (cv::Point MayBeCenter : MayBeCenters)//ע�⣬����Сͼ������ϵ
	{
		//std::cout <<"debug::" << MayBeCenter << "\n";
		if (this->targetColor == 1)cv::circle(Show, MayBeCenter, 3, cv::Scalar(255, 255, 255), 2, 2, 0);
		else cv::circle(Show, MayBeCenter, 3, cv::Scalar(255, 255, 255), 2, 2, 0);
		ans.push_back(MayBeCenter);
	}



	//DBG_IMSHOW("��Բ", Show);
	cv::imshow("��Ժ", Show);
	return ans.size() > 0;
}
bool BufDetect::getNeedHit(vector<vector<cv::Point>>contours, vector<cv::Point>& targetcontour, cv::Point R_center) {
	std::cout << "getNeedHit,the size is " << contours.size() << "\n";

	cv::Point2f rSignCentroid(R_center.x * 1.0, R_center.y * 1.0);

	for (vector<cv::Point>contour : contours)
	{
		cv::Point2f armorCentroid = computeCentroid(contour);
		std::vector<double> huMoments = this->calculateHuMoments(contour);//��͹���
		cv::Point2f vectorToRSign = normalizedVector(armorCentroid, rSignCentroid);
		// ... ����������Ӹ����߼����Ƚ�Hu�أ���������״ ...

		// ����������͹����͹ȱ��
		std::vector<int> hullIndices;
		cv::convexHull(contour, hullIndices, true);
		std::vector<cv::Vec4i> convexityDefects;
		cv::convexityDefects(contour, hullIndices, convexityDefects);

		// ����͹ȱ�ݣ��ж���״
		for (const auto& defect : convexityDefects) {
			float depth = defect[3] / 256.0; // ͹ȱ�ݵ����
			int startIdx = defect[0];
			int endIdx = defect[1];
			int farthestIdx = defect[2];
			if (depth > 10) { // ��ֵ������ʵ���������
				// �����������͹ȱ�ݣ���������Ҷ
				cv::Point start = contour[startIdx];
				cv::Point end = contour[endIdx];
				cv::Point farthest = contour[farthestIdx];
				cv::Point2f farthestf(farthest.x * 1.0, farthest.y * 1.0);

				cv::Point2f vectorToDefect = normalizedVector(armorCentroid, farthestf);

				double dot = this->isPointingTowards(vectorToRSign, vectorToDefect);
				std::cout << "dot=" << dot << "\n";
				std::cout << "vectorToRSigh=" << vectorToRSign << ",vectorToDetect=" << vectorToDefect << "\n";
				if (dot >= this->dot_thresh) {
					targetcontour = contour;

					return true;
				}
			}
		}


	}
	// ���û��������͹ȱ�ݣ������Ǵ���
	return false;
}
cv::Point BufDetect::findDenseAreaAndSetROI(cv::Mat inputImage) {
	// Ԥ��������ֵ��
  //  cv::Mat binaryImage;
   // cv::threshold(inputImage, binaryImage, 127, 255, cv::THRESH_BINARY);

	// �������
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(inputImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Ѱ���������
	double maxArea = 0;
	std::vector<cv::Point> maxContour;
	for (const auto& contour : contours) {
		double area = cv::contourArea(contour);
		if (area > maxArea) {
			maxArea = area;
			maxContour = contour;
		}
	}

	// ��������
	cv::Moments m = cv::moments(maxContour);
	cv::Point centroid = cv::Point(m.m10 / m.m00, m.m01 / m.m00);

	return centroid;
}
bool BufDetect::red_or_blue(cv::Mat input, vector<cv::Point>countor) {
	cv::Mat mask = cv::Mat::zeros(input.size(), CV_8UC1);

	// ʹ������������ͼ���ϻ������İ�ɫ����
	vector<vector<cv::Point>>countours;
	countours.push_back(countor);
	cv::drawContours(mask, countours, 0, cv::Scalar(255), cv::FILLED);

	// ��������������ͼ���ƽ����ɫ
	cv::Scalar avgColor = cv::mean(input, mask);

	// �ж���������ƫ����ɫ���Ǻ�ɫ
	// ע�⣺OpenCV����ɫͨ����˳����BGR
	if (avgColor[2] > avgColor[0]) { // ��ɫͨ����ƽ��ֵ������ɫͨ��
		return true;
	}
	else {
		return false;
	}
	return false;
}
cv::Mat BufDetect::Bupt2HSV(cv::Mat input) {
	cv::Mat hsvImage;
	cv::cvtColor(input, hsvImage, cv::COLOR_BGR2HSV);

	// ����HSV��ɫ�ռ��к�ɫ����ɫ����ֵ
	// ��Щֵ��Ҫ��������Ӧ�ý��е���
	cv::Scalar redLower1(0, 120, 30), redUpper1(10, 255, 255);
	cv::Scalar redLower2(160, 120, 30), redUpper2(180, 255, 255);
	cv::Scalar blueLower(100, 150, 50), blueUpper(140, 255, 255);

	// ����targetColor��ֵѡ��ͬ����ɫ��ֵ
	cv::Mat colorMask;
	if (targetColor == 1) {
		// Ѱ�Һ�ɫ
		cv::Mat mask1, mask2;
		cv::inRange(hsvImage, redLower1, redUpper1, mask1);
		cv::inRange(hsvImage, redLower2, redUpper2, mask2);
		// �ϲ�������ɫ���������
		cv::bitwise_or(mask1, mask2, colorMask);
		//colorMask = mask1 | mask2;
	}
	else {
		// Ѱ����ɫ
		cv::inRange(hsvImage, blueLower, blueUpper, colorMask);
	}

	// �����������̬ѧ�����Լ������
//    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
  //  cv::morphologyEx(colorMask, colorMask, cv::MORPH_CLOSE, kernel);

	// ���������ؽ��ͼ��
	cv::Mat resultImage;
	input.copyTo(resultImage, colorMask); // ��������������

	cv::Mat temp_show;
	cv::resize(resultImage, temp_show, cv::Size(360, 360));
	DBG_IMSHOW("HSV", temp_show);

	return resultImage;
}
double BufDetect::nearCenter(vector<cv::Point> contour) {


	// ����ÿ�������ı߽����
	cv::Rect boundingBox = cv::boundingRect(contour);

	// ���������ľ�
	cv::Moments m = cv::moments(contour);
	cv::Point centroid = cv::Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));

	// �߽���ε�����
	cv::Point boundingBoxCenter = (boundingBox.tl() + boundingBox.br()) * 0.5;

	// �������ĵ��߽�������ĵľ���
	double dx = centroid.x - boundingBoxCenter.x;
	double dy = centroid.y - boundingBoxCenter.y;

	// �ж������ľ��г̶ȣ�������Ը���ʵ�����������ֵ
	double distance = sqrt(dx * dx + dy * dy);
	double maxDistance = sqrt(boundingBox.width * boundingBox.width + boundingBox.height * boundingBox.height) / 2.0;

	// ������ӳ�䵽0��1֮��
	double normalizedDistance = distance / maxDistance;
	normalizedDistance = std::min(normalizedDistance, 1.0); // ȷ�����ᳬ��1

	return 1 - normalizedDistance;
}
void BufDetect::checkpicture() {
	std::string folderPath = "F:\\rm2023\\��������\\data\\images"; // �滻Ϊ���ͼƬ�ļ���·��

	// C++17���·���Ƿ���ں��Ƿ�ΪĿ¼
	if (!std::filesystem::exists(folderPath) || !std::filesystem::is_directory(folderPath)) {
		std::cout << "Directory path is not valid." << std::endl;
		return;
	}

	// ����ָ��Ŀ¼�е������ļ�����������Ŀ¼��
	for (const auto& entry : filesystem::directory_iterator(folderPath)) {
		if (entry.is_regular_file()) {
			// ��ȡ�ļ�·��
			auto filePath = entry.path();
			// ��ȡ�ļ���չ����ת��ΪСд��
			std::string extension = filePath.extension().string();
			std::transform(extension.begin(), extension.end(), extension.begin(),
				[](unsigned char c) { return std::tolower(c); });

			// �����ͼƬ�ļ����������ӵ��б���
			if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp") {
				this->imgPaths.push_back(filePath.string());
			}
		}
	}

	// ��������ҵ���ͼƬ·��
	for (const auto& imagePath : this->imgPaths) {
		std::cout << imagePath << std::endl;
	}

}
void BufDetect::checkRpicture() {
	std::string folderPath = "F:\\rm2023\\template"; // �滻Ϊ���ͼƬ�ļ���·��

	// C++17���·���Ƿ���ں��Ƿ�ΪĿ¼
	if (!std::filesystem::exists(folderPath) || !std::filesystem::is_directory(folderPath)) {
		std::cout << "Rune Directory path is not valid." << std::endl;
		return;
	}

	vector<std::string>R_Path;
	// ����ָ��Ŀ¼�е������ļ�����������Ŀ¼��
	for (const auto& entry : filesystem::directory_iterator(folderPath)) {
		if (entry.is_regular_file()) {
			// ��ȡ�ļ�·��
			auto filePath = entry.path();
			// ��ȡ�ļ���չ����ת��ΪСд��
			std::string extension = filePath.extension().string();
			std::transform(extension.begin(), extension.end(), extension.begin(),
				[](unsigned char c) { return std::tolower(c); });

			// �����ͼƬ�ļ����������ӵ��б���
			if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp") {
				R_Path.push_back(filePath.string());
			}
		}
	}

	// ��������ҵ���ͼƬ·��
	for (const auto& EachPath : R_Path) {

		std::cout << "��������template:" << EachPath << std::endl;
		vector<vector<cv::Point>>countors;
		countors.clear();
		cv::Mat img1, edges1;
		img1 = cv::imread(EachPath, cv::IMREAD_GRAYSCALE);

		cv::Canny(img1, edges1, 100, 200);

		cv::findContours(edges1, countors, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		vector<cv::Point>countor_t;
		//cv::approxPolyDP(countors[0], countor_t, 13, true);
		countor_t = countors[0];
		countors[0] = countor_t;
		this->template_R_countours.push_back(countors);
		this->template_R_gray.push_back(img1);
	}

}
BufDetect::BufDetect() {
	cv::Mat img;
	checkpicture();
	checkRpicture();
	namedWindow("���Ƶ�Ԫ");

	createTrackbar("ͼƬid", "���Ƶ�Ԫ", &imgid, this->imgPaths.size() - 1);
	createTrackbar("�����Һ�", "���Ƶ�Ԫ", &this->targetColor, 1);
	createTrackbar("thresh1", "���Ƶ�Ԫ", &thresh1, 255);//��Сֵ��0
	createTrackbar("thresh2", "���Ƶ�Ԫ", &thresh2, 255);
	createTrackbar("thresh3", "���Ƶ�Ԫ", &thresh3, 255);
}
double BufDetect::get_distance(cv::Point p1, cv::Point p2) {
	return sqrtf(powf((p1.x - p2.x), 2) + powf((p1.y - p2.y), 2));
}
void BufDetect::guidetect() {

	if (last_targetColor == targetColor && thresh1 == last_thresh1 && thresh2 == last_thresh2 && this->last_imgid == this->imgid && this->last_thresh3 == this->thresh3)
	{

	}
	else {
		last_targetColor = targetColor, last_thresh1 = thresh1, last_thresh2 = thresh2, last_imgid = imgid, last_thresh3 = thresh3;
		cv::Mat input = cv::imread(this->imgPaths[this->imgid]);
		this->Detect(input);
	}
	return;
}
std::vector<cv::Point> BufDetect::Detect(cv::Mat input) {

	std::vector<cv::Point> ans;
	ans.clear();
	DBG_IMSHOW("ԭͼ", input);
	Mat video1 = input.clone();
	this->img1 = input.clone();
	//  circle(input, Point(input.cols, input.rows) / 2, 10, Scalar(255, 255, 255), 1);
	Mat video_th;


	Mat Show = input.clone();


	// ����һ��ͨ�����ȥ
	// ����ͨ��
	vector<Mat> channels;
	cv::split(input, channels);


	/*
	// ͨ���������ֵ��
	if (this->targetColor == 1)//1Ϊ��ɫ
		subtract(channels[2], channels[0], video1);
	else
		subtract(channels[0], channels[2], video1);
		*/



	double alpha = 0.3; // ���ƶԱȶ�
	int beta = -50; // ��������
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

	// ת�ɻҶ�ͼ���ٶ�ֵ��

	//ת��HSV����
	//video1 = Bupt2HSV(video1).clone();

	vector<Mat> channels_low;
	cv::split(video1, channels_low);

	cv::cvtColor(video1, video1, COLOR_BGR2GRAY);
	cv::threshold(video1, video1, thresh1, 255, THRESH_BINARY);//ͨ�����������R�ƻ��ϴ�
	cv::dilate(video1, video1, kernel2);

	//namedWindow("�Ҷ�ͼ��ֵ��");
	cv::Mat video1_show;
	cv::resize(video1, video1_show, cv::Size(360, 360));
	DBG_IMSHOW("��ֵ����R(û��bitsize)", video1_show);

	/*
	// ͨ���������ֵ��
	if (this->targetColor == 1)//1Ϊ��ɫ
		subtract(channels[2], channels[0], video_th);
	else
		subtract(channels[0], channels[2], video_th);
	*/
	//ʹ�ý����˶Աȶȵİ汾:
	if (this->targetColor == 1)//1Ϊ��ɫ
		subtract(channels_low[2], channels_low[0], video_th);
	else
		subtract(channels_low[0], channels_low[2], video_th);



	cv::threshold(video_th, video_th, thresh2, 255, THRESH_BINARY);

	//cv::bitwise_and(video_th, video1, video1);
	// imshow("jjj", video_th);



	/*
	//����2 ���ͶԱȶ���ͨ�����


	double alpha = 0.5; // ���ƶԱȶ�
	int beta = -50; // ��������

	// �������ȺͶԱȶ�
	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			for (int c = 0; c < 3; c++) {
				video1.at<cv::Vec3b>(y, x)[c] =
					cv::saturate_cast<uchar>(alpha * input.at<cv::Vec3b>(y, x)[c] + beta);
			}
		}
	}
	//cv::waitKey(2000);
	vector<Mat> channels;
	split(video1, channels);
	cvtColor(video1, video1, COLOR_BGR2GRAY);
	threshold(video1, video1, thresh1, 255, THRESH_BINARY);
	cv::Mat video1_show;
   cv::resize(video1, video1_show,cv::Size(360, 360));
	DBG_IMSHOW("�Ҷ�ͼ��ֵ��", video1_show);


	if (this->targetColor == 1)//1Ϊ��ɫ
		subtract(channels[2], channels[0], video_th);
	else
		subtract(channels[0], channels[2], video_th);
	threshold(video_th, video_th, thresh2, 255, THRESH_BINARY);

	//namedWindow("�Ҷ�ͼ��ֵ��");
  */


  // ȡ����
  //bitwise_and(video_th, video1, video1);
//  cv::Mat video_1_show;
  //cv::resize(video1, video_1_show, cv::Size(360, 360));
  //imshow("ȡ����", video_1_show);




  //������
	cv::morphologyEx(video_th, video_th, MORPH_CLOSE, kernel1);
	cv::dilate(video_th, video_th, kernel2);
	cv::Mat video_th_show;
	cv::resize(video_th, video_th_show, cv::Size(360, 360));

	DBG_IMSHOW("��ֵ����װ�װ�", video_th_show);
	// imshow("video1", video1);
	// this->sendDebugImage("video1", video1);

	vector<vector<Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(video1, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);//�����ҵ�����Ԥ�������ƻ�һЩ����R������


	// cv::Mat mask(video1.size(), CV_8UC3, Scalar(0, 0, 0));
	 //drawContours(mask, contours, -1, Scalar(255, 0, 0), 2);
	 //cv::Mat mask_show;
	 //cv::resize(mask, mask_show,cv::Size(360,360));
	 //imshow("Contours", mask_show);



	Point2d R_center;
	bool find_R_center = false;


	vector<cv::Rect>R_MayBe;

	vector<vector<cv::Point>>contours_color_same;
	std::vector<cv::Vec4i> hierarchy_color_same;
	for (int i = 0; i < contours.size(); i++) {
		vector<cv::Point>contour = contours[i];
		cv::Vec4i h = hierarchy[i];
		if (red_or_blue(input, contour) == this->targetColor) {
			contours_color_same.push_back(contour);
			hierarchy_color_same.push_back(h);
		}
	}

	for (int i = 0; i < contours_color_same.size(); i++)
	{
		Rect boundRect = boundingRect(contours_color_same[i]);

		RotatedRect RoRect = minAreaRect(contours_color_same[i]);

		double rect_ratio = boundRect.width / boundRect.height;
		rect_ratio = max(rect_ratio, 1 / rect_ratio);


		//double rect_area = contourArea(contours[i]);
		double rect_area = boundRect.width * boundRect.height;
		double Rorect_area = RoRect.size.height * RoRect.size.width;//�����������е�������



		// if (rect_area < 600)
		// rectangle(input, boundRect.tl(), boundRect.br(), Scalar(255, 102, 204), 2, 8);  // for testing 

		//  ��������  --->  ����Ϊ�˲鿴�����ֲ�����ʵ������

		Point2f rect_center = (boundRect.br() + boundRect.tl()) / 2;
		double near_degree = nearCenter(contours_color_same[i]);

		double contour_area = cv::contourArea(contours_color_same[i]);
		double contour_perimeter = cv::arcLength(contours_color_same[i], true);


		double rate2;//�ܳ�������ı�

		std::cout << "(contour_perimeter/4.0)*(contour_perimeter / 4.0)/(contour_area*1.0)=" << (contour_perimeter / 4.0) * (contour_perimeter / 4.0) / (contour_area * 1.0) << "\n";
		//cout << "near_degree=" << near_degree << ",near_thresh=" << near_thresh / 100.0 << "\n";
		if (
			rect_area > 20
			//&& rect_area < 1000

			&& max(boundRect.width, boundRect.height) / max(input.rows, input.cols) <= 1 / 15.0
			&& min(boundRect.width, boundRect.height) / min(input.rows, input.cols) <= 1 / 15.0

			&&
			max(boundRect.width, boundRect.height) < 50
			&&
			//rect_ratio<=2.5
			true
			&&
			abs(boundRect.width - boundRect.height) <= 5
			&&
			// near_degree >= near_thresh / 100.0
			true
			&& get_distance(rect_center, box_center) > 200

			&&
			(contour_perimeter / 4.0) * (contour_perimeter / 4.0) / (contour_area * 1.0) >= 0.7
			&&
			(contour_perimeter / 4.0) * (contour_perimeter / 4.0) / (contour_area * 1.0) <= 2.1
			&&
			hierarchy_color_same[i][2] == -1//û�ж���
			&&
			hierarchy_color_same[i][3] == -1//û�и���

			)// && rect_ratio > 0.8 && rect_ratio < 1.2)
		{
			std::cout << "\nthis contour is pass the step1\n";
			if (this->targetColor == 0)
				cv::line(Show, boundRect.br(), boundRect.tl(), cv::Scalar(0, 0, 255), 7, 2);
			else
				cv::line(Show, boundRect.br(), boundRect.tl(), cv::Scalar(0, 255, 0), 7, 2);
			std::cout << "pos:" << boundRect.x * 1.0 / input.cols << "," << boundRect.y * 1.0 / input.rows << "\n";
			//  if (red_or_blue(input, contours[i]) != this->targetColor)continue; //�Ѿ���ǰ�봦����

			  // ����ڰ����ر���
			Mat roi = video1(boundRect);
			// imshow("roi", roi);
			// this->sendDebugImage("roi", roi);

			double rate = (sum(roi / 255)[0]) * 1.0 / (roi.cols * roi.rows);
			rate = (sum(roi / 255)[0] * 1.0) / Rorect_area;

			// static_cast<double>(countNonZero(roi)) / static_cast<double>(roi.cols * roi.rows)  
			std::cout << "rate=" << rate << ",contour_area/size=" << contour_area * 1.0 / Rorect_area << "\n";
			if (rate > 0.62 && contour_area * 1.0 / Rorect_area >= 0.72)
			{
				double R_degree = this->R_Same(input(boundRect));//��״̫С��ûɶ��

				if (R_degree > 9.5)continue;

				// double R_degree = this->R_Same2(input(boundRect));
				 //if (R_degree < 0.70)continue;
				 // rectangle(Show, boundRect.tl(), boundRect.br(), Scalar(255, 255, 0), 2, 8);
				find_R_center = true;
				R_center = rect_center;
				R_MayBe.push_back(boundRect);
				//break;
			}
		}
		else {
			std::cout << "\nthis contour is false:\n";
			std::cout << "pos:" << boundRect.x * 1.0 / input.cols << "," << boundRect.y * 1.0 / input.rows << "\n";
			std::cout << "rect_area>20:: " << (rect_area > 20) << "\n";
			//&& rect_area < 1000
			if (!(rect_area > 20))std::cout << "rect_area:: " << rect_area << "\n";
			std::cout << "max(boundRect)/max(input):" << (max(boundRect.width, boundRect.height) / max(input.rows, input.cols) <= 1 / 15.0) << "\n";
			if (!(max(boundRect.width, boundRect.height) / max(input.rows, input.cols) <= 1 / 15.0))std::cout << "max(boundRect)/max(input)=" << max(boundRect.width, boundRect.height) / max(input.rows, input.cols) << "\n";

			std::cout << "min(boundRect)/min(input):: " << (min(boundRect.width, boundRect.height) / min(input.rows, input.cols) <= 1 / 15.0) << "\n";
			if (!(min(boundRect.width, boundRect.height) / min(input.rows, input.cols) <= 1 / 15.0))std::cout << "min(boundRect)/min(input)=" << min(boundRect.width, boundRect.height) / min(input.rows, input.cols) << "\n";

			std::cout << "max(boudRect.width,boundRect.height)<50:: " << (max(boundRect.width, boundRect.height) < 50) << "\n";
			if (!(max(boundRect.width, boundRect.height) < 50))std::cout << "max(boudRect.width,boundRect.height)=" << max(boundRect.width, boundRect.height) << "\n";


			std::cout << "abs(boundRect.width-boundRect.height)<=5:: " << (abs(boundRect.width - boundRect.height) <= 5) << "\n";
			if (!(abs(boundRect.width - boundRect.height) <= 5))std::cout << "abs(boundRect.width-boundRect.height)=" << abs(boundRect.width - boundRect.height) << "\n";

			std::cout << "get_disdance(rect_center,box_center)>200:: " << (get_distance(rect_center, box_center) > 200) << "\n";
			if (!(get_distance(rect_center, box_center) > 200))std::cout << "get_distance=" << get_distance(rect_center, box_center) << "\n";
			std::cout << "(contour_perimeter/4.0)*(contour_perimeter / 4.0)/(contour_area*1.0)>=0.7:: " << ((contour_perimeter / 4.0) * (contour_perimeter / 4.0) / (contour_area * 1.0) >= 0.7) << "\n";
			if (!((contour_perimeter / 4.0) * (contour_perimeter / 4.0) / (contour_area * 1.0) >= 0.7))std::cout << "(contour_perimeter/4.0)*(contour_perimeter / 4.0)/(contour_area*1.0)=" << (contour_perimeter / 4.0) * (contour_perimeter / 4.0) / (contour_area * 1.0) << "\n";
			std::cout << "(contour_perimeter/4.0)*(contour_perimeter / 4.0)/(contour_area*1.0)<=2.1:: " << ((contour_perimeter / 4.0) * (contour_perimeter / 4.0) / (contour_area * 1.0) <= 2.1) << "\n";
			if (!((contour_perimeter / 4.0) * (contour_perimeter / 4.0) / (contour_area * 1.0) <= 2.1))std::cout << "(contour_perimeter/4.0)*(contour_perimeter / 4.0)/(contour_area*1.0)=" << (contour_perimeter / 4.0) * (contour_perimeter / 4.0) / (contour_area * 1.0) << "\n";

		}

	}
	int max_id = 0;
	bool ok_r_and_hit = false;


	if (!find_R_center) {
		std::wcout << "û���ҵ�R_center\n";
		ans.clear();
		return ans;
	}

	cv::Mat bitthresh;
	cv::bitwise_or(video1, video_th, bitthresh);
	//cv::erode(bitthresh, bitthresh, kernel2);

	cv::Mat bitthresh_show;
	cv::resize(bitthresh, bitthresh_show, cv::Size(360, 360));
	DBG_IMSHOW("bitset", bitthresh_show);

	vector<vector<cv::Point>>thresh_countors, thresh_countors_color_same;
	cv::findContours(bitthresh, thresh_countors, RETR_TREE, CHAIN_APPROX_SIMPLE);

	for (vector<cv::Point>contour : thresh_countors) {
		if (red_or_blue(input, contour) == this->targetColor) {
			thresh_countors_color_same.push_back(contour);
		}
	}


	vector<cv::Point>hit_contour, max_hit_contour;
	hit_contour.clear();
	for (int i = 0; i < R_MayBe.size(); i++) {
		hit_contour.clear();
		cv::Rect boundRect = R_MayBe[i], MaxRect = R_MayBe[max_id];
		cv::Point temp_R_center = (boundRect.br() + boundRect.tl()) / 2;
		int temp_R_Size = boundRect.width * boundRect.height;
		if (this->getNeedHit(this->getContours(thresh_countors_color_same, temp_R_center, temp_R_Size), hit_contour, temp_R_center))
		{
			if (ok_r_and_hit)
			{
				if (boundRect.width * boundRect.height > MaxRect.width * MaxRect.height)
				{
					max_id = i;
					max_hit_contour = hit_contour;
					hit_contour.clear();
				}
			}
			else {
				ok_r_and_hit = true;
				max_id = i;
				max_hit_contour = hit_contour;
				hit_contour.clear();
			}
		}
	}
	if (!ok_r_and_hit) {
		std::wcout << "û�кϷ���R!\n";
		ans.clear();
		return ans;
	}

	if (this->targetColor == 1)
		rectangle(Show, R_MayBe[max_id].tl(), R_MayBe[max_id].br(), Scalar(0, 255, 0), 4, 8);
	else
		rectangle(Show, R_MayBe[max_id].tl(), R_MayBe[max_id].br(), Scalar(0, 0, 255), 4, 8);
	R_center = (R_MayBe[max_id].tl() + R_MayBe[max_id].br()) / 2;
	int R_Size = R_MayBe[max_id].width * R_MayBe[max_id].height;
	std::cout << "���󳤿���ֵ:" << abs(R_MayBe[max_id].width - R_MayBe[max_id].height) << "\n";
	cv::Point ROI = findDenseAreaAndSetROI(video1);
	// cv::circle(Show, ROI, ROI_radius, cv::Scalar(255, 255, 255), 3, 1);


	vector<vector<Point>> armors;
	// morphologyEx(video_th, video_th, MORPH_OPEN, kernel2);
	// erode(video_th, video_th, kernel2);
	// floodFill(video_th, Point(0, 0), Scalar(0));//��ˮ



	std::wcout << "�ҵ�������װ�װ�����\n";
	Rect max_hitRect = boundingRect(max_hit_contour);

	if (this->targetColor == 1)
		rectangle(Show, max_hitRect.tl(), max_hitRect.br(), Scalar(0, 255, 0), 4, 8);
	else
		rectangle(Show, max_hitRect.tl(), max_hitRect.br(), Scalar(0, 0, 255), 4, 8);
	cv::Mat hit_img = input(max_hitRect);

	cv::Mat Show_1;
	cv::resize(Show, Show_1, cv::Size(360, 360));
	DBG_IMSHOW("�ҵ�����R", Show_1);
	std::wcout << "���ҵ�������װ�װ�����\n";
	//R_centerӦ�ñ��Сͼ����ϵ��
	cv::Point R_center_temp0 = R_center;
	cv::Point R_center_temp1 = R_center_temp0 - max_hitRect.tl();
	cv::Point2d R_center2 = R_center_temp1;
	if (this->getNeedHit2(hit_img, R_center2, ans) == false) {//���ݴ�Χ��С��Χ
		ans.clear();
		return ans;
	}
	/*
	cv::findContours(video_th, armors, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<RotatedRect> armor_rects; // ���������б�
	vector<double> outline_area;
	for (int i = 0; i < armors.size(); i++)
	{
		RotatedRect rotatedRect = minAreaRect(armors[i]);
		Point2f vertices[4];

		double rect_ratio = min(rotatedRect.size.width, rotatedRect.size.height) / max(rotatedRect.size.width, rotatedRect.size.height);
		double rect_area = contourArea(armors[i]);

		if (rect_area > 300.0 && rect_area < 1000.0 && rect_ratio > 0.3 && rect_ratio < 0.7)
		{
			rotatedRect.points(vertices);
			armor_rects.push_back(rotatedRect);
		}

		// ���ƾ��ε�������
		// for (int j = 0; j < 4; j++)
		// {
		//     line(input, vertices[j], vertices[(j+1)%4], Scalar(255, 0, 0), 2);
		// }
	}


	//'''    ƥ���Ӧ��װ�װ�   '''
	vector<cv::RotatedRect> target_rects;
	bool find_ok = false;  //�ɹ��ҵ�װ�װ壿
	if (armor_rects.size() > 1)
	{
		for (int i = 0; i < armor_rects.size() - 1; i++)
		{
			for (int j = i + 1; j < armor_rects.size(); j++)
			{
				double distance = get_distance(armor_rects[i].center, armor_rects[j].center);
				double angle_value = abs(armor_rects[i].angle - armor_rects[j].angle);
				// cout << "angle_value:" << angle_value << endl;

				double area_value = abs(armor_rects[i].size.area() - armor_rects[j].size.area());
				// cout << "area_value: " << area_value << endl;

				if (distance < 100. && angle_value < 5.)// && area_value < 500.)
				{
					target_rects.push_back(armor_rects[i]);
					target_rects.push_back(armor_rects[j]);

					find_ok = true;
					break;
				}
			}
			if (find_ok == true)
				break;
		}
	}


	double radius;
	bool hasTarget = 0;
	//'''   Ѱ��װ�װ�   '''
	if (target_rects.size() != 2)
	{
		find_ok = false;
	   // log("Can not find the armor box!!!!!!!!!!!!", find_ok);
		std::cout << "could not find the armor box!!!\n";
	}
	else
	{
		Point2f target_center;  //# Ŀ�����ĵ�
		cv::RotatedRect target_armor_in, target_armor_out;
		//'''  ��Ŀ����������ĵ� '''
		target_center = (target_rects[0].center + target_rects[1].center) / 2;
		circle(Show, target_center, 12, Scalar(0, 255, 0), 2);

		double dis1 = get_distance(R_center, target_rects[0].center);
		double dis2 = get_distance(R_center, target_rects[1].center);
		if (dis1 < dis2)
		{
			target_armor_in = target_rects[0];
			target_armor_out = target_rects[1];
		}
		else
		{
			target_armor_in = target_rects[1];
			target_armor_out = target_rects[0];
		}

		Point2f rect_in[4];
		Point2f rect_out[4];
		target_armor_in.points(rect_in);  //����С��Ӿ����ĸ��˵㸴�Ƹ�rect����
		target_armor_out.points(rect_out);  //����С��Ӿ����ĸ��˵㸴�Ƹ�rect����
		//'''   ���Ŀ�����   '''
		for (int i = 0; i < 4; i++)
		{
			line(Show, rect_in[i], rect_in[(i + 1) % 4], Scalar(46, 255, 89), 1);
			line(Show, rect_out[i], rect_out[(i + 1) % 4], Scalar(140, 82, 255), 1);
		}


		/////       ��װ�װ�Ľǵ�
		vector<Bufarmor_point> armor_points;
		for (int i = 0; i < 4; i++) {
			Bufarmor_point point;
			point.point = rect_in[i];
			point.dis = get_distance(target_center, rect_in[i]);
			armor_points.push_back(point);
		}
		for (int i = 0; i < 4; i++) {
			Bufarmor_point point;
			point.point = rect_out[i];
			point.dis = get_distance(target_center, rect_out[i]);
			armor_points.push_back(point);
		}
		sort(armor_points.begin(), armor_points.end(), [](Bufarmor_point a, Bufarmor_point b)
			{return a.dis < b.dis; });
		for (int i = 0; i < 4; i++)
		{
			cv::circle(Show, armor_points[i].point, 3, cv::Scalar(255, 0, 255), -1);
		}

		std::vector<cv::Point2f> armor_points_pixel;
		armor_points_pixel = { armor_points[0].point - target_center, armor_points[1].point - target_center,
							  armor_points[2].point - target_center, armor_points[3].point - target_center };

		radius = get_distance(R_center, target_center);
		//log("radius", radius);
		std::cout << "radius:" << radius << "\n";
	}
	cv::Mat Show_2;
	cv::resize(Show, Show_2, cv::Size(360, 360));
	DBG_IMSHOW("result", Show_2);
	std_msgs::Header header;
	sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "bgr8", input).toImageMsg();
	publisher.publish(msg);

	// this->sendDebugImage("result", input);
	//cv::waitKey(1);
*/
	std::vector<cv::Point>realans;//���ԭ����ϵ
	for (cv::Point eachPoint : ans) {
		cv::Point realPoint = eachPoint;
		realPoint = realPoint + max_hitRect.tl();

		realans.push_back(realPoint);
	}
	return realans;
}
