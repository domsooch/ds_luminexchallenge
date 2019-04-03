#pragma once

#include <iostream>
#include <algorithm>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


using namespace cv;
using namespace cv::ml;
using namespace std;


bool DEBUG = false;

void printRows(const Mat& M, int row_start, int row_end) {
	/*
	prints selected matrix rows.
	*/
	for (int i = row_start; i < row_end; i++) {
		Mat rc = M.row(i).clone();
		cout << "Row = " << i << " :  " << rc << endl;
	}
}

void printMatrix(const Mat& M) {
	/*
	prints all matrix rows.
	*/
	printRows(M, 0, M.rows);
}

void ShowImage(const Mat& M) {
	Mat test;
	resize(M, test, Size(), 10.0, 10.0, INTER_NEAREST);
	imwrite("sps_Image.jpg", test);
	imshow("Display window", test);                   // Show our image inside it.
	waitKey(0);
}

void ShowMatrix(const Mat& M, std::string name, bool show_image) {
	/*
	Image Data inspectoer
	*/
	const static vector<std::string> datatypes = { "#define CV_8U 0",
		"#define CV_8S 1",
		"#define CV_16U 2",
		"#define CV_16S 3",
		"#define CV_32S 4",
		"#define CV_32F 5",
		"#define CV_64F 6",
		"#define CV_USRTYPE1 7" };
	int depth = M.depth();
	printf("%s %i, %i depth: %s\n", name.c_str(), M.rows, M.cols, datatypes[depth].c_str());
	if (show_image) {
		ShowImage(M);
	}
}

void MakeSPixelArray(const Mat& M, vector<cv::Vec3f>& super_pixel_vector, float data_weight) {
	/*
	This creates a "super-pixel" made up of column normalized (x, y, 16bit_grayscale) vectors
	weight can be added to the 16bit channel in order to adjust segmentation.
	This is used to train the EM model.
	*/
	//Build SuperPixel
	int rows = M.rows;
	int cols = M.cols;
	int rc_rows = rows * cols;
	Mat reshaped_image = M.reshape(1, rc_rows);
	double min, max;
	cv::minMaxLoc(M, &min, &max);
	float fmax = (float)max;
	float val_sum = (float)cv::sum(reshaped_image)[0];
	int r; int c;
	float row_sum = (float)(rows*(rows + 1)) / 2.0;//Gauss's Trick
	float col_sum = (float)(rows*(rows + 1)) / 2.0;
	float val;
	/*MatIterator_<ushort> it, end;
	for (it=M.begin<ushort>, end=M.end<ushort>; it != end; ++it) {
		cout << "it" << *it << endl;
	}*/
	for (r = 0; r < rows; r++) {
		for (c = 0; c < cols; c++) {
			val = (float)(M.at<unsigned short>(r, c));
			cv::Vec3f x = { (float)r / row_sum , (float)c / col_sum, val / val_sum * data_weight };
			super_pixel_vector.push_back(x);
		}
	}
	//printf("MakeSPixelArray(Hack!): cv::sum(reshaped_image)[0] %f   %i rows sz out_array: %i\n", val_sum, rc_rows, super_pixel_vector.size());
}

void MakeMask(Mat& M, Mat& Mask, int search_value, uint8_t fill_value) {
	/*
	There is a better way to do this. 
	bitwise_and with a scaler is a better way to pull masks from the EM labels.
	*/
	Mask = Mat::zeros(M.rows, M.cols, CV_8UC1);
	MatIterator_<ushort> it, end;
	int x = 0; int val;
	for (int r = 0; r < M.rows; r++) {
		for (int c = 0; c < M.cols; c++) {
			val = (M.at<uchar>(r, c));
			if (val == search_value) {
				Mask.at<uchar>(r, c) = uchar(fill_value);
				x++;
			}
		}
	}
	//cout << "mask n: " << x << endl;
}

void FillMask(Mat& M, Mat& im_out) {
	/*
	This uses the and/or complementarity of the smallest seqgents to determine if tthere are two nuclei.
	None of the flood fill methods worked well, so I used a hack. 
	I flood the outer area of the segement 6 with 128
	Then I replace the inner pixels with 255
	Then I revert the 128 pixels to 0.
	In this way I generate a flood-filled n-1 inner segement
	Thios is used to determine if the the inner most segment is contained inside the n-1 segment
	*/
	//https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
	Mat im_th;
	threshold(M, im_th, 240, 255, THRESH_BINARY_INV);
	//if (DEBUG) ShowMatrix(im_th, "im_th", true);
	//printMatrix(im_th);
	uint8_t fillValue = 128;
	// Floodfill from point (0, 0)
	Mat im_floodfill = im_th.clone();
	floodFill(im_floodfill, cv::Point(31, 19), Scalar(120));
	Mat mask;
	inRange(im_floodfill, 0, 0, mask);//convert inner color to white
	im_floodfill.setTo(255, mask);
	mask = Mat::zeros(M.rows, M.cols, CV_8UC1);
	inRange(im_floodfill, 120, 120, mask);//convert inner color to white
	im_floodfill.setTo(0, mask);
	//if (DEBUG) ShowMatrix(im_floodfill, "im_floodfill", true);
	im_out = im_floodfill;
}

void SetColorScheme_RdYlGn(vector<Vec3b>& color_scheme) {
	/*
	Color Map
	*/
	color_scheme = { cv::Vec3b(165, 0, 38),
		cv::Vec3b(215, 48, 39),
		cv::Vec3b(244, 109, 67),
		cv::Vec3b(253, 174, 97),
		cv::Vec3b(254, 224, 139),
		cv::Vec3b(255, 255, 191),
		cv::Vec3b(217, 239, 139),
		cv::Vec3b(166, 217, 106),
		cv::Vec3b(102, 189, 99),
		cv::Vec3b(26, 152, 80),
		cv::Vec3b(0, 104, 55) };
}

class EMRunner {
	int img_rows;
	int img_cols;
	int num_em_clusters;
	Mat em_input;
	Mat labels_em, probs, log_likelihoods, means, label_2d;
	Ptr<EM> em_model;
public:
	EMRunner(int, int, int);
	void BuildEMModel();
	void Train(Mat&);
	void display();
	void ViewClusters(Mat&);
	std::string Classifier();
};
EMRunner::EMRunner(int nc, int r, int c) : num_em_clusters(nc), img_rows(r), img_cols(c) {}
void EMRunner::BuildEMModel() {
	em_model = EM::create();
	em_model->setClustersNumber(num_em_clusters);
	em_model->setCovarianceMatrixType(EM::COV_MAT_DIAGONAL);
	em_model->setTermCriteria(TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, 1e-6));
}
void EMRunner::Train(Mat& emInput) {
	em_input = emInput.clone();
	em_model->trainEM(emInput, log_likelihoods, labels_em, probs);
	means = em_model->getMeans();
	display();
}
void EMRunner::display() {
	//cout << "means: = " << means << endl;
	//cout << "labels: = " << labels_em << endl;
	if (DEBUG) ShowMatrix(probs, "probs", false);
	if (DEBUG) ShowMatrix(labels_em, "labels_em", false);
	if (DEBUG) ShowMatrix(means, "means", false);
}
void EMRunner::ViewClusters(Mat& rgb_image) {
	assert(num_em_clusters <= 10);//Color map is limited to 10 clusters
	vector<Vec3b> color_scheme;
	SetColorScheme_RdYlGn(color_scheme);
	rgb_image = Mat(img_rows, img_cols, CV_8UC3);
	Mat _label_2d = labels_em.reshape(1, (img_cols, img_rows));
	_label_2d.convertTo(label_2d, CV_8UC1);
	cout << "img_rows, img_cols" << img_rows << " , " << img_cols << endl;
	if (DEBUG) ShowMatrix(label_2d, "label_2d", false);
	MatIterator_<Vec3b> rgb_first = rgb_image.begin<Vec3b>();
	MatIterator_<Vec3b> rgb_last = rgb_image.end<Vec3b>();
	MatConstIterator_<uchar> label_first = label_2d.begin<uchar>();
	int x = 0;
	while (rgb_first != rgb_last) {
		//cout << x << " " << *label_first << color_scheme[*label_first] <<  endl;
		*rgb_first++ = color_scheme[*label_first++];
		//rgb_first++; label_first++;
		//if (x > img_sz.height*img_sz.width-1)break;//Hack there's a bug hewre
		x++;
	}
	if (DEBUG) ShowMatrix(rgb_image, "rgb_image", false);
	waitKey();
}

std::string  EMRunner::Classifier() {
	//Find most intense two label_ids
	struct idx_mean { int idx; float val; };
	std::vector<idx_mean> idx_meanLst;
	//std::cout << means << endl;
	for (int row = 0; row < means.rows; row++) {
		double val = means.at<double>(row, 2);
		//cout << "float val" << val << endl;
		idx_mean im = { row, (float)val };
		idx_meanLst.push_back(im);
	}
	sort(idx_meanLst.begin(), idx_meanLst.end(), [](idx_mean a, idx_mean b) {return a.val > b.val; });
	int inner_idx = idx_meanLst[0].idx;
	int second_idx = idx_meanLst[1].idx;
	//cout << "inner_idx" << inner_idx << " second_idx " << second_idx << endl;
	//Make masks
	uint fill_value = 255;
	Mat mask_inner, mask_second;// (img_rows, img_cols, CV_8UC1);
	MakeMask(label_2d, mask_inner, inner_idx, fill_value);
	MakeMask(label_2d, mask_second, second_idx, fill_value);

	if (DEBUG) ShowMatrix(mask_inner, "mask_inner", true);
	if (DEBUG) ShowMatrix(mask_second, "mask_second", true);

	Mat flood_mask_second;

	FillMask(mask_second, flood_mask_second);
	if (DEBUG) ShowMatrix(flood_mask_second, "flood_mask_second", true);

	Mat andffinner, orffinner;
	bitwise_and(flood_mask_second, mask_inner, andffinner);
	bitwise_or(flood_mask_second, mask_inner, orffinner);
	if (DEBUG) ShowMatrix(andffinner, "andffinner", true);
	if (DEBUG) ShowMatrix(orffinner, "orffinner", true);
	int count_and, count_or, count_inner, count_ff;
	count_ff = countNonZero(flood_mask_second);
	count_and = countNonZero(andffinner);
	count_or = countNonZero(orffinner);
	count_inner = countNonZero(mask_inner);
	float score = ((float)count_and) / ((float)count_or);
	std::string classifier_result;
	if (score < 0.11) {
		classifier_result = "C";
	}
	else if (count_inner < 30) {
		classifier_result = "A";
	}
	else {
		classifier_result = "B";
	}
	std::ostringstream ss;
	ss << count_ff << "," << count_inner << "," << count_and << "," << count_or << "," << score << "," << classifier_result;
	classifier_result = ss.str();
	return classifier_result;
}


