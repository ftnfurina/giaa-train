# GIAA OCR 模型炼丹

此项目为 [GIAA](https://github.com/ftnfurina/giaa) 项目的 OCR 模型炼丹，基于 PaddleOCR PP-OCRv5 版本微调。

## 环境搭建

### 克隆项目

```bash
git clone --recurse-submodules https://github.com/ftnfurina/giaa-train.git
```

### 确认 CUDA 版本

通过 `nvitop` 查看 CUDA 版本, 将 [pyproject.toml](./pyproject.toml) 中的 `paddlepaddle-gpu` `url` 改为对应版本的链接。

```bash
pip install nvitop
nvitop
```

### 初始化项目

```bash
bash ./init.sh gpu
```

### 下载训练模型到 `configs` 目录

+ [PP-OCRv5_mobile_det_pretrained.pdparams](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_det_pretrained.pdparams)
+ [PP-OCRv5_mobile_rec_pretrained.pdparams](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams)

### 开始训练

```bash
bash ./train.sh det
bash ./train.sh rec
```

### 验证结果

将截图放入 `images` 目录，运行 `validate.py` 脚本，验证结果将保存至 `images/inference/` 目录。

```bash
uv run validate.py
```

## 相关链接

+ PaddlePaddle 安装指南: https://www.paddlepaddle.org.cn/install/quick
+ PaddleOCR 文档: https://www.paddleocr.ai/main/index.html
+ PaddleOCR GitHub: https://github.com/PaddlePaddle/PaddleOCR
+ Genshin Impact 数据库: https://github.com/theBowja/genshin-db
+ uv 文档: https://docs.astral.sh/uv/
+ GIAA: https://github.com/ftnfurina/giaa
