#pragma once

#include "deploy_core/base_detection.h"

namespace detection_2d {

std::shared_ptr<BaseDetectionModel> CreateRFDetrDetectionModel(
    const std::shared_ptr<inference_core::BaseInferCore> &infer_core,
    const std::shared_ptr<IDetectionPreProcess>          &preprocess_block,
    const int                                             input_height,
    const int                                             input_width,
    const int                                             input_channel,
    const int                                             cls_number,
    const int                                             candidates_num    = 300,
    const int                                             select_topk       = 100,
    const std::vector<std::string>                       &input_blobs_name  = {"input"},
    const std::vector<std::string>                       &output_blobs_name = {"dets", "labels"});

} // namespace detection_2d
