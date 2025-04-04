#include <gtest/gtest.h>

#include "ort_core/ort_core.h"
#include "tests/fps_counter.h"
#include "detection_2d_util/detection_2d_util.h"
#include "detection_2d_rf_detr/rf_detr.hpp"
#include "tests/image_drawer.h"

/**************************
****  trt core test ****
***************************/

using namespace inference_core;
using namespace detection_2d;

static
std::shared_ptr<BaseDetectionModel> CreateRFDetrModel()
{
  auto engine = CreateOrtInferCore("/workspace/models/rf-detr-base.onnx");
  auto preprocess_block = CreateCpuDetPreProcess({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}, true, true);
  auto model            = CreateRFDetrDetectionModel(engine, preprocess_block, 560, 560, 3, 90, 300);

  return model;
}

static
std::tuple<cv::Mat> ReadTestImage()
{
  auto left  = cv::imread("/workspace/test_data/persons.jpg");

  return {left};
}

TEST(rf_detr_test, ort_core_correctness)
{
  auto model = CreateRFDetrModel();
  auto [image] = ReadTestImage();

  std::vector<BBox2D> results;
  model->Detect(image, results, 0.4);

  ImageDrawHelper drawer(std::make_shared<cv::Mat>(image.clone()));
  for (const auto &res : results)
  {
    drawer.drawRect2D(res);
  }
  auto p_image = drawer.getImage();
  cv::imwrite("/workspace/test_data/rf_detr_detect_result.png", *p_image);
}


TEST(rf_detr_test, ort_core_speed)
{
  auto model   = CreateRFDetrModel();
  auto [image] = ReadTestImage();

  FPSCounter fps_counter;
  fps_counter.Start();
  for (size_t i = 0; i < 500; ++i)
  {
    std::vector<BBox2D> results;
    model->Detect(image, results, 0.4, false);
    fps_counter.Count(1);
    if (i % 100 == 0)
    {
      LOG(WARNING) << "average fps : " << fps_counter.GetFPS();
    }
  }
}