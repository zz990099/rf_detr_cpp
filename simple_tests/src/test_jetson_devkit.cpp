#include <gtest/gtest.h>
#include <glog/logging.h>

#include "trt_core/trt_core.h"
#include "tests/fps_counter.h"
#include "detection_2d_util/detection_2d_util.h"
#include "detection_2d_rf_detr/rf_detr.hpp"
#include "tests/image_drawer.h"

/**************************
****  trt core test ****
***************************/

using namespace inference_core;
using namespace detection_2d;

std::shared_ptr<BaseDetectionModel> CreateRFDetrModel()
{
  auto engine = CreateTrtInferCore("/workspace/models/rf-detr-base-fp16.engine");
  // auto preprocess_block = CreateCudaDetPreProcess();
  auto preprocess_block =
      CreateCpuDetPreProcess({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}, true, true);
  auto model = CreateRFDetrDetectionModel(engine, preprocess_block, 560, 560, 3, 90, 300);

  return model;
}

std::tuple<cv::Mat> ReadTestImage()
{
  auto left = cv::imread("/workspace/test_data/persons.jpg");

  return {left};
}

TEST(rf_detr_test, trt_core_correctness)
{
  auto model   = CreateRFDetrModel();
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

TEST(rf_detr_test, trt_core_speed)
{
  auto model   = CreateRFDetrModel();
  auto [image] = ReadTestImage();

  FPSCounter fps_counter;
  fps_counter.Start();
  for (size_t i = 0; i < 2000; ++i)
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

TEST(rf_detr_test, trt_core_pipeline_correctness)
{
  auto model = CreateRFDetrModel();
  model->InitPipeline();
  auto [image] = ReadTestImage();

  auto async_func = [&]() { return model->DetectAsync(image, 0.4); };

  auto thread_fut = std::async(std::launch::async, async_func);

  auto stereo_fut = thread_fut.get();

  CHECK(stereo_fut.valid());

  auto results = stereo_fut.get();

  ImageDrawHelper drawer(std::make_shared<cv::Mat>(image.clone()));
  for (const auto &res : results)
  {
    drawer.drawRect2D(res);
  }
  auto p_image = drawer.getImage();
  cv::imwrite("/workspace/test_data/rf_detr_detect_result.png", *p_image);
}

TEST(rf_detr_test, trt_core_pipeline_speed)
{
  auto model = CreateRFDetrModel();
  model->InitPipeline();
  auto [image] = ReadTestImage();

  deploy_core::BlockQueue<std::shared_ptr<std::future<std::vector<BBox2D>>>> future_bq(100);

  auto func_push_data = [&]() {
    int index = 0;
    while (index++ < 10000)
    {
      auto p_fut = std::make_shared<std::future<std::vector<BBox2D>>>(model->DetectAsync(image.clone(), 0.4));
      future_bq.BlockPush(p_fut);
    }
    future_bq.SetNoMoreInput();
  };

  FPSCounter fps_counter;
  auto       func_take_results = [&]() {
    int index = 0;
    fps_counter.Start();
    while (true)
    {
      auto output = future_bq.Take();
      if (!output.has_value())
        break;
      output.value()->get();
      fps_counter.Count(1);
      if (index++ % 100 == 0)
      {
        LOG(WARNING) << "average fps: " << fps_counter.GetFPS();
      }
    }
  };

  std::thread t_push(func_push_data);
  std::thread t_take(func_take_results);

  t_push.join();
  model->StopPipeline();
  t_take.join();
  model->ClosePipeline();

  LOG(WARNING) << "average fps: " << fps_counter.GetFPS();
}
