#include "runec/YoloTRT.h"

void YoloTRT::detect(const cv::Mat &image,std::vector<Detection>&res)
{

    float scale = 1.0;
    int img_size = image.cols * image.rows * 3; // BGR
    cudaMemcpyAsync(image_device, image.data, img_size, cudaMemcpyHostToDevice, stream);
    preprocess(image_device, image.cols, image.rows, device_buffers[0], kInputW, kInputH, stream, scale);
    //   std::cout << "debug::prepare to enqueue...\n";
    context->enqueue(kBatchSize, (void **)device_buffers, stream, nullptr);
    cudaMemcpyAsync(output_buffer_host, device_buffers[1], kBatchSize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    //    std::cout << "debug::prepare to NMS...\n";
    res.clear();
    NMS(res, output_buffer_host, kConfThresh, kNmsThresh);
    /*�ع�*/
    for (Detection& result : res) {
        for (float& val : result.bbox)
        {
            val /= scale;
        }
    }
    
    //std::cout << "Debug:\n";
	//cv::Mat input2 = image.clone();
    //drawBbox(input2, res, scale, labels);
    //cv::imshow("Inference", input2);
    //cv::waitKey(2000);
}
YoloTRT::YoloTRT(std::string user_wts_name, std::string user_engine_name, std::string user_class_file, std::string user_sub_type)
{
    wts_name = user_wts_name;
    engine_name = user_engine_name;
    class_file = user_class_file;
    sub_type = user_sub_type;

    kOutputSize = -1;
    inputIndex = -1;
    outputIndex = -1;
}
YoloTRT::~YoloTRT()
{
    cudaStreamDestroy(stream);
    cudaFree(device_buffers[0]);
    cudaFree(device_buffers[1]);
    delete[] output_buffer_host;
    // Destroy the engine
    delete context;
    delete engine;
    delete runtime;
}
std::map<int,std::string>YoloTRT::getlabels(){
    return this->labels;
}
void YoloTRT::serializeEngine(const int &kBatchSize, std::string &wts_name, std::string &engine_name, std::string &sub_type)
{
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    nvinfer1::IHostMemory *serialized_engine = nullptr;

    if (sub_type == "n")
    {
        serialized_engine = buildEngineYolov8n(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    }
    else if (sub_type == "s")
    {
        serialized_engine = buildEngineYolov8s(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    }
    else if (sub_type == "m")
    {
        serialized_engine = buildEngineYolov8m(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    }
    else if (sub_type == "l")
    {
        serialized_engine = buildEngineYolov8l(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    }
    else if (sub_type == "x")
    {
        serialized_engine = buildEngineYolov8x(kBatchSize, builder, config, nvinfer1::DataType::kFLOAT, wts_name);
    }

    assert(serialized_engine);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p)
    {
        std::cout << "could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());

    delete builder;
    delete config;
    delete serialized_engine;
}

bool YoloTRT::setConfig(std::string InputName, std::string OutputName,
                        int NumClass, int BatchSize,
                        int InputH, int InputW,
                        float NmsThresh, float ConfThresh,
                        int MaxInputImageSize, int MaxNumOutputBbox)
{
    //std::cout << "InputName is:" << InputName << ",OutputName is:" << OutputName << "\n";
    localInputName = InputName, localOutputName = OutputName;

    kInputTensorName = localInputName.c_str();
    kOutputTensorName = localOutputName.c_str();

    kNumClass = NumClass;
    kBatchSize = BatchSize;

    kInputH = InputH;
    kInputW = InputW;

    kNmsThresh = NmsThresh;
    kConfThresh = ConfThresh;

    kMaxInputImageSize = MaxInputImageSize;
    kMaxNumOutputBbox = MaxNumOutputBbox;

    return true;
}
bool YoloTRT::readyWork()
{

    std::cout << "config showing ...\n";
    std::cout << kInputTensorName << " " << kOutputTensorName << "\n"
              << kNumClass << " " << kBatchSize << "\n";
    std::cout << kInputH << " " << kInputW << "\n"
              << kNmsThresh << " " << kConfThresh << "\n";
    std::cout << kMaxInputImageSize << " " << kMaxNumOutputBbox << "\n";

    std::cout << "config show done.\n";
    runtime = nullptr;
    engine = nullptr;
    context = nullptr;

    while (readEngineFile(engine_name, runtime, engine, context) == false)
    {
        serializeEngine(kBatchSize, wts_name, engine_name, sub_type);
    }
    cudaStreamCreate(&stream);
    kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

    image_device = nullptr;
    output_buffer_host = new float[kBatchSize * kOutputSize];
    assert(engine->getNbBindings() == 2);
    inputIndex = engine->getBindingIndex(kInputTensorName);
    outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    cudaMalloc((void **)&image_device, kMaxInputImageSize * 3);
    cudaMalloc((void **)&device_buffers[0], kBatchSize * 3 * kInputH * kInputW * sizeof(float));
    cudaMalloc((void **)&device_buffers[1], kBatchSize * kOutputSize * sizeof(float));

    labels.clear();
    readClassFile(class_file, labels);

    return true;
}