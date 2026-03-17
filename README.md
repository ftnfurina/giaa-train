# GIAA OCR 模型炼丹

此项目为 [GIAA](https://github.com/ftnfurina/giaa) 项目的 OCR 模型炼丹，基于 PaddleOCR PP-OCRv5 版本微调。

## 环境搭建

### 克隆项目

```bash
git clone --recurse-submodules https://github.com/ftnfurina/giaa-train.git
```

### 安装 uv

```bash
pip install uv
```

### 确认 CUDA 版本

通过 `nvitop` 查看 CUDA 版本, 将 [pyproject.toml](./pyproject.toml) 中的 `paddlepaddle-gpu` `url` 改为对应版本的链接。

```bash
pip install nvitop
nvitop
```

### 安装依赖

```bash
uv sync --extra gpu # cpu
```

### 构建数据集

```bash
uv run build_dataset.py
```

### 下载训练模型到 `configs` 目录

+ [PP-OCRv5_mobile_det_pretrained.pdparams](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_det_pretrained.pdparams)
+ [PP-OCRv5_mobile_rec_pretrained.pdparams](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams)

### 安装 ccache 和 screen

```bash
sudo apt-get install ccache # 加速训练
sudo apt-get install screen # 后台运行

screen -S giaa-train # 新建一个 screen 窗口
# ... # 执行训练命令
screen -r giaa-train # 进入 screen 窗口
```

### 开始训练

```bash
uv run PaddleOCR/tools/train.py -c configs/PP-OCRv5_mobile_det.yml # -o Global.use_gpu=False
uv run PaddleOCR/tools/train.py -c configs/PP-OCRv5_mobile_rec.yml # -o Global.use_gpu=False
```

### 评估模型

```bash
uv run PaddleOCR/tools/eval.py -c configs/PP-OCRv5_mobile_det.yml -o Global.pretrained_model=output/PP-OCRv5_mobile_giaa_det/best_accuracy.pdparams
uv run PaddleOCR/tools/eval.py -c configs/PP-OCRv5_mobile_rec.yml -o Global.pretrained_model=output/PP-OCRv5_mobile_giaa_rec/best_accuracy.pdparams
```

### 导出模型

```bash
uv run PaddleOCR/tools/export_model.py -c configs/PP-OCRv5_mobile_det.yml -o Global.pretrained_model=output/PP-OCRv5_mobile_giaa_det/best_accuracy.pdparams Global.save_inference_dir=output/PP-OCRv5_mobile_giaa_det_infer/
uv run PaddleOCR/tools/export_model.py -c configs/PP-OCRv5_mobile_rec.yml -o Global.pretrained_model=output/PP-OCRv5_mobile_giaa_rec/best_accuracy.pdparams Global.save_inference_dir=output/PP-OCRv5_mobile_giaa_rec_infer/
```

### 模型转换

```bash
uv run paddle2onnx -m output/PP-OCRv5_mobile_giaa_det_infer/ -mf inference.json -pf inference.pdiparams -s onnx/PP-OCRv5_mobile_giaa_det.onnx
uv run paddle2onnx -m output/PP-OCRv5_mobile_giaa_rec_infer/ -mf inference.json -pf inference.pdiparams -s onnx/PP-OCRv5_mobile_giaa_rec.onnx
```

## 相关链接

+ PaddlePaddle 安装指南: https://www.paddlepaddle.org.cn/install/quick
+ PaddleOCR 文档: https://www.paddleocr.ai/main/index.html
+ PaddleOCR GitHub: https://github.com/PaddlePaddle/PaddleOCR
+ Genshin Impact 数据库: https://github.com/theBowja/genshin-db
+ uv 文档: https://docs.astral.sh/uv/
+ GIAA: https://github.com/ftnfurina/giaa
