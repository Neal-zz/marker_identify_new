#pragma once
#include <opencv2/opencv.hpp>
#include <math.h>

struct PatternContainer {
	PatternContainer()
		: p1(-1,-1),
		p2(-1, -1),
		p3(-1, -1),
		p4(-1, -1),
		p5(-1, -1),
		p6(-1, -1),
		p7(-1, -1),
		p8(-1, -1)
	{
	}

	int getId() {
		int id = -1;
		if (p1.x < 0 || p2.x < 0 || p3.x < 0 || p4.x < 0 ||
			p5.x < 0 || p6.x < 0 || p7.x < 0 || p8.x < 0) {
			std::cout << "PatternContainer is not initialized..." << std::endl;
			return id;
		}

		cv::Mat xy = (cv::Mat_<float>(2, 2) << 
			(p2.x - p1.x), (p3.x - p1.x),
			(p2.y - p1.y), (p3.y - p1.y));  // [p2-p1 p4-p1] * [a;b] = pi - p1; xy * [a;b] = c.
		cv::Mat xy_inv = xy.inv();

		// 6
		cv::Mat c = (cv::Mat_<float>(2, 1) <<
			(p6.x - p1.x),
			(p6.y - p1.y));
		cv::Mat ab = xy_inv * c;
		cv::Point2f pos6(ab.at<float>(0, 0), ab.at<float>(1, 0));

		// 7
		c = (cv::Mat_<float>(2, 1) <<
			(p7.x - p1.x),
			(p7.y - p1.y));
		ab = xy_inv * c;
		cv::Point2f pos7(ab.at<float>(0, 0), ab.at<float>(1, 0));

		// 8
		c = (cv::Mat_<float>(2, 1) <<
			(p8.x - p1.x),
			(p8.y - p1.y));
		ab = xy_inv * c;
		cv::Point2f pos8(ab.at<float>(0, 0), ab.at<float>(1, 0));

		float thresh = 0.2;
		// 93
		if (norm(pos6 - cv::Point2f(0.38, 2.40)) < thresh &&
			norm(pos7 - cv::Point2f(2.39, 1.35)) < thresh &&
			norm(pos8 - cv::Point2f(1.99, 0.00)) < thresh) {
			id = 93;
		}
		// 92
		else if (norm(pos6 - cv::Point2f(0.37, 2.03)) < thresh &&
			norm(pos7 - cv::Point2f(2.47, 1.39)) < thresh &&
			norm(pos8 - cv::Point2f(2.02, 0.00)) < thresh) {
			id = 92;
		}
		// 91
		else if (norm(pos6 - cv::Point2f(0.00, 2.33)) < thresh &&
			norm(pos7 - cv::Point2f(2.38, 1.39)) < thresh &&
			norm(pos8 - cv::Point2f(2.02, 0.00)) < thresh) {
			id = 91;
		}
		// 72
		else if (norm(pos6 - cv::Point2f(0.34, 2.31)) < thresh &&
			norm(pos7 - cv::Point2f(2.41, 1.07)) < thresh &&
			norm(pos8 - cv::Point2f(2.07, 0.00)) < thresh) {
			id = 72;
		}
		// 73
		else if (norm(pos6 - cv::Point2f(0.95, 2.38)) < thresh &&
			norm(pos7 - cv::Point2f(2.42, 1.09)) < thresh &&
			norm(pos8 - cv::Point2f(2.07, 0.00)) < thresh) {
			id = 73;
		}
		// 74
		else if (norm(pos6 - cv::Point2f(1.29, 2.33)) < thresh &&
			norm(pos7 - cv::Point2f(2.36, 1.03)) < thresh &&
			norm(pos8 - cv::Point2f(2.05, -0.01)) < thresh) {
			id = 74;
		}

		return id;
	}

	cv::Point2f p1;
	cv::Point2f p2;
	cv::Point2f p3;
	cv::Point2f p4;
	cv::Point2f p5; 
	cv::Point2f p6;
	cv::Point2f p7;
	cv::Point2f p8;
};

void findSquares(const cv::Mat& image, std::vector<std::vector<cv::Point>>& squares);

bool find8Points(const cv::Mat& image, std::vector<std::vector<cv::Point2f>>& result, std::vector<float>& pR,
	std::vector<int>& minx, std::vector<int>& miny, std::vector<int>& maxx, std::vector<int>& maxy);

PatternContainer distinguish8Points(const std::vector<cv::Point2f>& pointsIn, const float pR);

std::vector<int> crossCheck(std::vector<PatternContainer>& leftPoints, std::vector<PatternContainer>& rightPoints);

std::vector<cv::Point3f> uv2xyz(const std::vector<cv::Point2f>& lPts, const std::vector<cv::Point2f>& rPts,
	const cv::Mat& cameraMatrixl, const cv::Mat& distCoeffsl,
	const cv::Mat& cameraMatrixr, const cv::Mat& distCoeffsr, const cv::Mat& T2);

bool markerIdentify(const cv::Mat& leftSrc, const cv::Mat& rightSrc, std::vector<int>& markerId, std::vector<cv::Mat>& Tcam_marker);
