#!/bin/bash
#
# DINOv3 精度对比简化脚本
# 直接使用配置文件进行测试
#

PYTHON_CMD=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "python3")

# 配置文件路径
CONFIG_FILE=${1:-"test_tipc/configs/DINOv3/DINOv3_vits16_hf_precision.txt"}

echo "========================================"
echo "DINOv3 精度对比测试 (简化版)"
echo "配置文件: ${CONFIG_FILE}"
echo "========================================"
echo ""

# 设置 PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# 执行测试
${PYTHON_CMD} test_tipc/compare_hf_precision.py --config ${CONFIG_FILE}

echo ""
echo "========================================"
