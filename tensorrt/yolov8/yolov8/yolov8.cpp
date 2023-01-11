// xujing
//YOLOv8


#include <iostream>
#include <fstream>
#include <cstdio>
#include <stdio.h>
#include <math.h>
#include <string>

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"

#include <npp.h>
#include "logging.h"

using namespace sample;
using namespace std;
using namespace cv;

#define BATCH_SIZE 1
#define INPUT_W 640
#define INPUT_H 640
#define INPUT_SIZE 640
#define IsPadding 1


std::vector<std::string> class_names = { "person","cat","dog","horse" };


// 中点坐标宽高
struct Bbox {
	float x;
	float y;
	float w;
	float h;
	float score;
	int classes;
};

//前处理
void preprocess(cv::Mat& img, float data[]) {
	int w, h, x, y;
	float r_w = INPUT_W / (img.cols*1.0);
	float r_h = INPUT_H / (img.rows*1.0);
	if (r_h > r_w) {
		w = INPUT_W;
		h = r_w * img.rows;
		x = 0;
		y = (INPUT_H - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = INPUT_H;
		x = (INPUT_W - w) / 2;
		y = 0;
	}
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
	//cudaResize(img, re);
	cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

	int i = 0;
	for (int row = 0; row < INPUT_H; ++row) {
		uchar* uc_pixel = out.data + row * out.step;
		for (int col = 0; col < INPUT_W; ++col) {
			data[i] = (float)uc_pixel[2] / 255.0;
			data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
			data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;
			++i;
		}
	}
}


//只需把box映射回原图
std::vector<Bbox> rescale_box(std::vector<Bbox> &out, int width, int height) {
	float gain = 640.0 / std::max(width, height);
	float pad_x = (640.0 - width * gain) / 2;
	float pad_y = (640.0 - height * gain) / 2;

	std::vector<Bbox> boxs;
	Bbox box;
	for (int i = 0; i < (int)out.size(); i++) {
		box.x = (out[i].x - pad_x) / gain;
		box.y = (out[i].y - pad_y) / gain;
		box.w = out[i].w / gain;
		box.h = out[i].h / gain;
		box.score = out[i].score;
		box.classes = out[i].classes;

		boxs.push_back(box);
	}

	return boxs;

}

//可视化
cv::Mat renderBoundingBox(cv::Mat image, const std::vector<Bbox> &bboxes) {
	for (const auto &rect : bboxes)
	{

		cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
		cv::rectangle(image, rst, cv::Scalar(255, 204, 0), 2, cv::LINE_8, 0);
		//cv::rectangle(image, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2), cv::Point(rect.x + rect.w / 2, rect.y + rect.h / 2), cv::Scalar(255, 204,0), 3);

		int baseLine;
		std::string label = class_names[rect.classes] + ": " + std::to_string(rect.score * 100).substr(0, 4) + "%";

		cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_TRIPLEX, 0.5, 1, &baseLine);
		//int newY = std::max(rect.y, labelSize.height);
		rectangle(image, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - round(1.5*labelSize.height)),
			cv::Point(rect.x - rect.w / 2 + round(1.0*labelSize.width), rect.y - rect.h / 2 + baseLine), cv::Scalar(255, 204, 0), cv::FILLED);
		cv::putText(image, label, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 204, 255));


	}
	return image;
}


float h_input[INPUT_SIZE * INPUT_SIZE * 3];
int h_output_0[1];   //1
float h_output_1[1 * 20 * 4];   //1
float h_output_2[1 * 20];   //1
float h_output_3[1 * 20];   //1




int main() {

	Logger gLogger;
	//初始化插件，调用plugin必须初始化plugin respo
nvinfer1:initLibNvInferPlugins(&gLogger, "");


	nvinfer1::IRuntime* engine_runtime = nvinfer1::createInferRuntime(gLogger);
	std::string engine_filepath = "./model/yolov8s.plan";

	std::ifstream file;
	file.open(engine_filepath, std::ios::binary | std::ios::in);
	file.seekg(0, std::ios::end);
	int length = file.tellg();
	file.seekg(0, std::ios::beg);

	std::shared_ptr<char> data(new char[length], std::default_delete<char[]>());
	file.read(data.get(), length);
	file.close();

	//nvinfer1::ICudaEngine* engine_infer = engine_runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);
	nvinfer1::ICudaEngine* engine_infer = engine_runtime->deserializeCudaEngine(data.get(), length, nullptr);
	nvinfer1::IExecutionContext* engine_context = engine_infer->createExecutionContext();

	int input_index = engine_infer->getBindingIndex("images"); //1x3x640x640
	//std::string input_name = engine_infer->getBindingName(0)
	int output_index_1 = engine_infer->getBindingIndex("num_detections");  //1
	int output_index_2 = engine_infer->getBindingIndex("nmsed_boxes");   // 2
	int output_index_3 = engine_infer->getBindingIndex("nmsed_scores");  //3
	int output_index_4 = engine_infer->getBindingIndex("nmsed_classes"); //5


	std::cout << "输入的index: " << input_index << " 输出的num_detections-> " << output_index_1 << " 输出的nmsed_boxes-> " << output_index_2
		<< " 输出的nmsed_scores-> " << output_index_3 << " 输出的nmsed_classes-> " << output_index_4 << std::endl;

	if (engine_context == nullptr)
	{
		std::cerr << "Failed to create TensorRT Execution Context." << std::endl;
	}

	// cached_engine->destroy();
	std::cout << "loaded trt model , do inference" << std::endl;


	cv::String pattern = "./test_img/*.jpg";
	std::vector<cv::String> fn;
	cv::glob(pattern, fn, false);
	std::vector<cv::Mat> images;
	size_t count = fn.size(); //number of png files in images folde

	std::cout << count << std::endl;

	for (size_t i = 0; i < count; i++)
	{
		cv::Mat image = cv::imread(fn[i]);
		cv::Mat image_origin = image.clone();

		////cv2读图片
		//cv::Mat image;
		//image = cv::imread("./test.jpg", 1);
		std::cout << fn[i] << std::endl;

		preprocess(image, h_input);

		void* buffers[5];
		cudaMalloc(&buffers[0], INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float));  //<- input
		cudaMalloc(&buffers[1], 1 * sizeof(int)); //<- num_detections
		cudaMalloc(&buffers[2], 1 * 20 * 4 * sizeof(float)); //<- nmsed_boxes
		cudaMalloc(&buffers[3], 1 * 20 * sizeof(float)); //<- nmsed_scores
		cudaMalloc(&buffers[4], 1 * 20 * sizeof(float)); //<- nmsed_classes

		cudaMemcpy(buffers[0], h_input, INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float), cudaMemcpyHostToDevice);

		// -- do execute --------//
		engine_context->executeV2(buffers);

		cudaMemcpy(h_output_0, buffers[1], 1 * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_output_1, buffers[2], 1 * 20 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_output_2, buffers[3], 1 * 20 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_output_3, buffers[4], 1 * 20 * sizeof(float), cudaMemcpyDeviceToHost);

		std::cout << h_output_0[0] << std::endl;
		std::vector<Bbox> pred_box;
		for (int i = 0; i < h_output_0[0]; i++) {

			Bbox box;
			box.x = (h_output_1[i * 4 + 2] + h_output_1[i * 4]) / 2.0;
			box.y = (h_output_1[i * 4 + 3] + h_output_1[i * 4 + 1]) / 2.0;
			box.w = h_output_1[i * 4 + 2] - h_output_1[i * 4];
			box.h = h_output_1[i * 4 + 3] - h_output_1[i * 4 + 1];
			box.score = h_output_2[i];
			box.classes = (int)h_output_3[i];

			std::cout << box.classes << "," << box.score << std::endl;

			pred_box.push_back(box);


		}

		std::vector<Bbox> out = rescale_box(pred_box, image.cols, image.rows);

		cv::Mat img = renderBoundingBox(image, out);

		//cv::imwrite("final.jpg", img);

		cv::namedWindow("Image", 1);//创建窗口
		cv::imshow("Image", img);//显示图像

		cv::waitKey(0); //等待按键

		std::string x = fn[i].substr(10);
		std::cout << x << std::endl;
		std::string save_index = "./res/" + x;
		cv::imwrite(save_index, img);

		cudaFree(buffers[0]);
		cudaFree(buffers[1]);
		cudaFree(buffers[2]);
		cudaFree(buffers[3]);
		cudaFree(buffers[4]);



	}

	engine_runtime->destroy();
	engine_infer->destroy();

	return 0;
}