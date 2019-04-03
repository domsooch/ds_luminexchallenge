// ds_lumi.cpp : 
//This program classifies images into class A, B, and C
// Typical CommandLine: ds_lumi.exe --glob_path "Z:\tdata1\luminex_challenge\classA\*.tif"
//


#include <iostream>
#include <algorithm>
#include <fstream>
#include <experimental/filesystem>

#include <opencv2/core/utility.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "utils.hpp"



using namespace cv;
using namespace cv::ml;
using namespace std;


namespace fs = std::experimental::filesystem;


const char* keys =
{
	"{ b build | | print complete build info     }"
	"{ h help  | | print this help               }"
	"{ @p glob_path  |./test_pic.tif | input path glob wildcard }"
	"{ @n image_num |-1| image_num to process     }"
	"{ @c num_clusters|7| em clusters to make}"
};



int main(int argc, const char* argv[])
{
	cv::CommandLineParser parser(argc, argv, keys);
	std::string glob_path = "";
	int image_num = -1;
	int num_clusters = 7;
	if (parser.has("help"))
	{
		parser.printMessage();
	}
	else if (!parser.check())
	{
		parser.printErrors();
	}
	else if (parser.has("build"))
	{
		std::cout << cv::getBuildInformation() << std::endl;
	}
	else
	{
		std::cout << "Running OpenCV " << CV_VERSION << std::endl;
	}
	if (parser.has("image_num")) {
		image_num = parser.get<int>("image_num");
	}
	if (parser.has("num_clusters")) {
		num_clusters = parser.get<int>("num_clusters");
	}
	if (parser.has("glob_path")) {
		glob_path = parser.get<std::string>("glob_path");
		std::cout << "glob_path is " << glob_path << std::endl;
	}
	vector<String> filenames;
	vector<String> output;
	cv::glob(glob_path, filenames);
	
	//cv::Size img_sz; int imgDepth;
	for (size_t i = 0; i < filenames.size(); i++)
	{
		if ((image_num!=-1) && ((int)i!= image_num))continue;
		Mat img = imread(filenames[i], IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
		cv::Size img_sz = img.size();
		
		printf("processing file_path: %zi %s\n", i, filenames[i].c_str());
		//ShowMatrix(img, "input img", false);
		
		Mat thresh_img;
		threshold(img, thresh_img, 150, 2500, THRESH_TOZERO);

		Mat norm_image, cimg;
		thresh_img.convertTo(norm_image, CV_8UC3);

		vector<cv::Vec3f> super_pixel_vector;
		MakeSPixelArray(thresh_img, super_pixel_vector, 10000.0);
		Mat em_input = Mat(super_pixel_vector.size(), 3, CV_32F, super_pixel_vector.data());
		//ShowMatrix(em_input, "em_input", false);

		EMRunner EMR(num_clusters, img.rows, img.cols);
		EMR.BuildEMModel();
		EMR.Train(em_input);
		Mat rgb_image;
		EMR.ViewClusters(rgb_image);
		std::string ret = EMR.Classifier();
		output.push_back(filenames[i] + "," + ret);
		cout << output[output.size() - 1] << endl;
	}
	ofstream myfile;

	std::ofstream output_file("./cell_classifier_results.txt");
	std::ostream_iterator<std::string> output_iterator(output_file, "\n");
	std::copy(output.begin(), output.end(), output_iterator);
	myfile.close();
	return 0;
}
