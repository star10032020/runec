#pragma once
#include "model.h"
#include "utils.h"
#include "process.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

class YoloTRT
{
public:
    YoloTRT(std::string user_wts_name = "model.wts", std::string user_engine_name = "model.engine", std::string user_class_file = "classes.txt", std::string user_sub_type = "n");
    ~YoloTRT();
    bool setConfig(std::string InputName = "images", std::string OutputName = "output",
                   int NumClass = 3, int BatchSize = 8,
                   int InputH = 640, int InputW = 640,
                   float NmsThresh = 0.45f, float ConfThresh = 0.5f,
                   int MaxInputImageSize = 3000 * 3000, int MaxNumOutputBbox = 1000);
    bool readyWork();

    void detect(const cv::Mat &image,std::vector<Detection>&res);
    Logger gLogger;

private:
    void serializeEngine(const int &kBatchSize, std::string &wts_name, std::string &engine_name, std::string &sub_type);
    std::string wts_name = "";
    std::string engine_name = "";
    std::string class_file = "";
    std::string sub_type = "";

    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;


    cudaStream_t stream;
    int kOutputSize;

    float *device_buffers[2];
    uint8_t *image_device = nullptr;
    float *output_buffer_host=nullptr;
    int inputIndex;
    int outputIndex;
    std::map<int, std::string> labels;

    std::string localInputName;
    std::string localOutputName;
};