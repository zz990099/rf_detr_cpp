#include "detection_2d_rf_detr/rf_detr.hpp"

namespace detection_2d {

class RFDetrDetection : public BaseDetectionModel {
public:
  RFDetrDetection(const std::shared_ptr<inference_core::BaseInferCore> &infer_core,
                  const std::shared_ptr<IDetectionPreProcess>          &preprocess_block,
                  const int                                             input_height,
                  const int                                             input_width,
                  const int                                             input_channel,
                  const int                                             cls_number,
                  const int                                             candidates_num,
                  const int                                             select_topk,
                  const std::vector<std::string>                       &input_blobs_name,
                  const std::vector<std::string>                       &output_blobs_name);

  ~RFDetrDetection() = default;

private:
  bool PreProcess(std::shared_ptr<async_pipeline::IPipelinePackage> pipeline_unit) override;

  bool PostProcess(std::shared_ptr<async_pipeline::IPipelinePackage> pipeline_unit) override;

private:
  const std::vector<std::string> input_blobs_name_;
  const std::vector<std::string> output_blobs_name_;
  const int                      input_height_;
  const int                      input_width_;
  const int                      input_channel_;
  const int                      cls_number_;
  const int                      candidates_num_;
  const int                      select_topk_;

  const std::shared_ptr<inference_core::BaseInferCore> infer_core_;
  std::shared_ptr<IDetectionPreProcess>                preprocess_block_;
};

RFDetrDetection::RFDetrDetection(const std::shared_ptr<inference_core::BaseInferCore> &infer_core,
                                 const std::shared_ptr<IDetectionPreProcess> &preprocess_block,
                                 const int                                    input_height,
                                 const int                                    input_width,
                                 const int                                    input_channel,
                                 const int                                    cls_number,
                                 const int                                    candidates_num,
                                 const int                                    select_topk,
                                 const std::vector<std::string>              &input_blobs_name,
                                 const std::vector<std::string>              &output_blobs_name)
    : BaseDetectionModel(infer_core),
      input_blobs_name_(input_blobs_name),
      output_blobs_name_(output_blobs_name),
      input_height_(input_height),
      input_width_(input_width),
      input_channel_(input_channel),
      cls_number_(cls_number),
      candidates_num_(candidates_num),
      select_topk_(select_topk),
      infer_core_(infer_core),
      preprocess_block_(preprocess_block)
{
  // 创建并获取一个缓存句柄，用于校验模型与算法的一致性
  auto p_map_buffer2ptr = infer_core_->AllocBlobsBuffer();
  if (p_map_buffer2ptr->Size() != input_blobs_name_.size() + output_blobs_name_.size())
  {
    LOG(ERROR) << "[RFDetrDetection] Infer core should has {"
               << input_blobs_name_.size() + output_blobs_name_.size() << "} blobs !"
               << " but got " << p_map_buffer2ptr->Size() << " blobs";
    throw std::runtime_error(
        "[RFDetrDetection] Construction Failed!!! Got invalid blobs_num size!!!");
  }

  for (const std::string &input_blob_name : input_blobs_name)
  {
    if (p_map_buffer2ptr->GetOuterBlobBuffer(input_blob_name).first == nullptr)
    {
      LOG(ERROR) << "[RFDetrDetection] Input_blob_name_ {" << input_blob_name
                 << "input blob name does not match `infer_core_` !";
      throw std::runtime_error(
          "[RFDetrDetection] Construction Failed!!! Got invalid input_blob_name!!!");
    }
  }

  for (const std::string &output_blob_name : output_blobs_name)
  {
    if (p_map_buffer2ptr->GetOuterBlobBuffer(output_blob_name).first == nullptr)
    {
      LOG(ERROR) << "[RFDetrDetection] Output_blob_name_ {" << output_blob_name
                 << "output blob name does not match `infer_core_` !";
      throw std::runtime_error(
          "[RFDetrDetection] Construction Failed!!! Got invalid output_blob_name!!!");
    }
  }
}

bool RFDetrDetection::PreProcess(std::shared_ptr<async_pipeline::IPipelinePackage> _package)
{
  auto package = std::dynamic_pointer_cast<DetectionPipelinePackage>(_package);
  CHECK_STATE(package != nullptr,
              "[RFDetrDetection] PreProcess the `_package` instance does not belong to "
              "`DetectionPipelinePackage`");

  const auto &blobs_buffer = package->GetInferBuffer();
  float       scale        = preprocess_block_->Preprocess(package->input_image_data, blobs_buffer,
                                                           input_blobs_name_[0], input_height_, input_width_);
  package->transform_scale = scale;
  return true;
}

bool RFDetrDetection::PostProcess(std::shared_ptr<async_pipeline::IPipelinePackage> _package)
{
  auto package = std::dynamic_pointer_cast<DetectionPipelinePackage>(_package);
  CHECK_STATE(package != nullptr,
              "[RFDetrDetection] PostProcess the `_package` instance does not belong to "
              "`DetectionPipelinePackage`");

  const auto &blobs_buffer = package->GetInferBuffer();

  // RFDetrDetection outputs: dets <float32> (300, 4); labels (300, cls_num + 1);

  const float *dets_ptr =
      reinterpret_cast<float *>(blobs_buffer->GetOuterBlobBuffer(output_blobs_name_[0]).first);
  const float *labels_ptr =
      reinterpret_cast<float *>(blobs_buffer->GetOuterBlobBuffer(output_blobs_name_[1]).first);

  const float conf_thresh  = package->conf_thresh;
  const float transf_scale = package->transform_scale;

  std::vector<std::pair<float, int>> container(candidates_num_ * (cls_number_ + 1));

  for (int i = 0; i < candidates_num_; ++i)
  {
    int offset = i * (cls_number_ + 1);
    for (int j = 0; j < cls_number_ + 1; ++j)
    {
      container[offset + j] = {labels_ptr[offset + j], offset + j};
    }
  }

  std::sort(container.begin(), container.end(), std::greater<std::pair<float, int>>());

  for (int i = 0; i < select_topk_; ++i)
  {
    float score = 1 / (1 + exp(-container[i].first));
    if (score < conf_thresh)
      break;

    int score_index = container[i].second;

    int cls       = score_index % (cls_number_ + 1);
    int box_index = score_index / (cls_number_ + 1);

    const float *box_ptr = dets_ptr + box_index * 4;
    BBox2D       box;
    box.x    = box_ptr[0] * input_width_ / transf_scale;
    box.y    = box_ptr[1] * input_height_ / transf_scale;
    box.w    = box_ptr[2] * input_width_ / transf_scale;
    box.h    = box_ptr[3] * input_height_ / transf_scale;
    box.cls  = cls;
    box.conf = score;
    package->results.push_back(box);
  }

  return true;
}

std::shared_ptr<BaseDetectionModel> CreateRFDetrDetectionModel(
    const std::shared_ptr<inference_core::BaseInferCore> &infer_core,
    const std::shared_ptr<IDetectionPreProcess>          &preprocess_block,
    const int                                             input_height,
    const int                                             input_width,
    const int                                             input_channel,
    const int                                             cls_number,
    const int                                             candidates_num,
    const int                                             select_topk,
    const std::vector<std::string>                       &input_blobs_name,
    const std::vector<std::string>                       &output_blobs_name)
{
  return std::make_shared<RFDetrDetection>(infer_core, preprocess_block, input_height, input_width,
                                           input_channel, cls_number, candidates_num, select_topk,
                                           input_blobs_name, output_blobs_name);
}

} // namespace detection_2d