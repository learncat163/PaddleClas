#!/bin/bash
#
# DINOv3 精度对比测试脚本
# 用于对比 HuggingFace DINOv3 和 PaddleClas DINOv3 的输出精度
#

source test_tipc/common_func.sh

FILENAME=$1
MODE=${2:-"lite_train_lite_infer"}

dataline=$(cat ${FILENAME})
IFS=$'\n'
lines=(${dataline})

# 解析基础参数
model_name=$(func_parser_value "${lines[1]}")
python=$(func_parser_value "${lines[2]}")

# 创建日志目录
CLS_ROOT_PATH=$(pwd)
LOG_PATH="${CLS_ROOT_PATH}/test_tipc/output/${model_name}/${MODE}"
mkdir -p ${LOG_PATH}

echo "========================================"
echo "DINOv3 精度对比测试"
echo "模型: ${model_name}"
echo "模式: ${MODE}"
echo "========================================"

# 检测系统可用的 Python 版本
PYTHON_CMD=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "python3")

# 从配置文件中解析 HF 精度对比参数
function func_get_hf_param() {
    local key=$1
    local line=$(echo "${dataline}" | grep "^${key}:")
    if [ -n "$line" ]; then
        echo "${line##${key}:}"
    else
        echo ""
    fi
}

hf_model_path=$(func_get_hf_param "hf_model_path")
hf_atol=$(func_get_hf_param "hf_precision_atol")
hf_rtol=$(func_get_hf_param "hf_precision_rtol")
test_image=$(func_get_hf_param "test_image_path")

# 设置默认值
hf_atol=${hf_atol:-1e-5}
hf_rtol=${hf_rtol:-1e-5}
test_image=${test_image:-"/home/cao/code/self/paddle/dinov3/000000039769.jpg"}

# 根据模型名称设置 Paddle 模型类
case "${model_name}" in
    "DINOv3_vits16")
        paddle_class="ppcls.arch.backbone.model_zoo.dinov3.DINOv3_vits16"
        pdparams_path="/tmp/dinov3-vits16.pdparams"
        ;;
    "DINOv3_vitb16")
        paddle_class="ppcls.arch.backbone.model_zoo.dinov3.DINOv3_vitb16"
        pdparams_path="/tmp/dinov3-vitb16.pdparams"
        ;;
    "DINOv3_vitl16")
        paddle_class="ppcls.arch.backbone.model_zoo.dinov3.DINOv3_vitl16"
        pdparams_path="/tmp/dinov3-vitl16.pdparams"
        ;;
    *)
        echo "错误: 未知的模型名称 ${model_name}"
        exit 1
        ;;
esac

echo "HF 模型路径: ${hf_model_path}"
echo "Paddle 权重: ${pdparams_path}"
echo "测试图片: ${test_image}"
echo ""

# 检查依赖
echo "检查依赖..."
if ! ${PYTHON_CMD} -c "import torch; import transformers; import paddle" 2>/dev/null; then
    echo "警告: 缺少必要的依赖包 (torch/transformers/paddle)"
    echo "请安装: pip install torch transformers paddlepaddle-gpu"
fi

# 执行精度对比测试
echo "开始精度对比测试..."
precision_log="${LOG_PATH}/hf_precision.log"

cmd="PYTHONPATH=$(pwd):$PYTHONPATH ${PYTHON_CMD} test_tipc/compare_hf_precision.py \
    --model_name ${model_name} \
    --hf_model_path ${hf_model_path} \
    --pdparams_path ${pdparams_path} \
    --image_path ${test_image} \
    --atol ${hf_atol} \
    --rtol ${hf_rtol} \
    --paddle_model_class ${paddle_class} > ${precision_log} 2>&1"

echo "执行命令: ${cmd}"
eval $cmd
exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "✅ 精度对比测试完成"
    cat ${precision_log}
else
    echo "❌ 精度对比测试失败"
    cat ${precision_log}
    exit 1
fi

# 检查精度是否通过
if grep -q '"passed": true' ${precision_log}; then
    echo ""
    echo "========================================"
    echo "✅ 精度验证通过！"
    echo "========================================"
    exit 0
elif grep -q '"passed": false' ${precision_log}; then
    echo ""
    echo "========================================"
    echo "❌ 精度验证未通过！"
    echo "========================================"
    exit 1
else
    echo ""
    echo "⚠️ 无法解析精度验证结果"
    exit 1
fi
