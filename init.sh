#!/bin/bash

# 获取命令行参数 type = gpu | cpu 默认 cpu
if [ $# -eq 0 ]; then
    type="cpu"
else
    type=$1
fi

echo "Start install uv"

# 安装 uv
pip install uv

if [ $? -ne 0 ]; then
    echo "Failed to install uv"
    exit 1
fi

echo "Start install ccache"

# 安装 ccache
if [ -f /etc/debian_version ]; then
    sudo apt-get install ccache
elif [ -f /etc/redhat-release ]; then
    sudo yum install ccache
elif [ -f /etc/fedora-release ]; then
    sudo dnf install ccache
else
    echo "Unsupported Linux distribution"
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "Failed to install ccache"
    exit 1
fi

echo "Start initialize uv environment"

# 初始化 uv 环境
uv sync --extra ${type}

if [ $? -ne 0 ]; then
    echo "Failed to initialize uv environment"
    exit 1
fi

echo "Start build dataset"

# 构建数据集
uv run dataset.py

if [ $? -ne 0 ]; then
    echo "Failed to build dataset"
    exit 1
fi

echo "All dependencies installed successfully"