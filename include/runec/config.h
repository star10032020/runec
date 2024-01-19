#define USE_FP16
//#define USE_INT8
#pragma once
const static char* kInputTensorName = "images";
const static char* kOutputTensorName = "output";

static int kNumClass = 3;
static int kBatchSize = 8;

static int kInputH = 640;
static int kInputW = 640;

static float kNmsThresh = 0.45f;
static float kConfThresh = 0.5f;

static int kMaxInputImageSize = 3000*3000;
static int kMaxNumOutputBbox = 1000;

struct alignas(float) Detection {
  float bbox[4];
  float conf;
  float class_id;
};
