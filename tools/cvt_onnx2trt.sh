#!/bin/bash

/usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/rf-detr-base.onnx \
                              --saveEngine=/workspace/models/rf-detr-base-fp16.engine \
                              --fp16