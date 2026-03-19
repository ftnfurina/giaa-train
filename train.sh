#!/bin/bash

# 获取命令行参数 type = det | rec
type=$1

# 检查参数是否正常
if [ "$type" != "det" ] && [ "$type" != "rec" ]; then
    echo "Usage: $0 det|rec"
    exit 1
fi

echo "Start train ${type} model"

# 训练模型 rec.yml -> type.yml
uv run PaddleOCR/tools/train.py -c configs/PP-OCRv5_mobile_${type}.yml # -o Global.use_gpu=False

if [ $? -ne 0 ]; then
    echo "Train failed"
    exit 1
fi

echo "Train ${type} model success"
echo "Model path: output/PP-OCRv5_mobile_giaa_${type}/"

echo "Start evaluate ${type} model"

# 评估模型
uv run PaddleOCR/tools/eval.py -c configs/PP-OCRv5_mobile_${type}.yml \
 -o Global.pretrained_model=output/PP-OCRv5_mobile_giaa_${type}/best_accuracy.pdparams

if [ $? -ne 0 ]; then
  echo "Evaluate failed"
  exit 1
fi

echo "Evaluate ${type} model success"

echo "Start export ${type} model"

# 导出模型
uv run PaddleOCR/tools/export_model.py -c configs/PP-OCRv5_mobile_${type}.yml \
 -o Global.pretrained_model=output/PP-OCRv5_mobile_giaa_${type}/best_accuracy.pdparams \
    Global.save_inference_dir=output/PP-OCRv5_mobile_giaa_${type}_infer/

if [ $? -ne 0 ]; then
    echo "Export failed"
    exit 1
fi

echo "Export ${type} model success"
echo "Model path: output/PP-OCRv5_mobile_giaa_${type}_infer/"

echo "Start convert ${type} model to onnx"

# 模型转换
uv run paddle2onnx -m output/PP-OCRv5_mobile_giaa_${type}_infer/ \
 -mf inference.json -pf inference.pdiparams -s onnx/PP-OCRv5_mobile_giaa_${type}.onnx

if [ $? -ne 0 ]; then
    echo "Convert to onnx failed"
    exit 1
fi

echo "Convert ${type} model to onnx success"
echo "Model path: onnx/PP-OCRv5_mobile_giaa_${type}.onnx"
