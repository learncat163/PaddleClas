#!/bin/bash
# MViTv2 训练测试脚本
# 用于测试 MViTv2_base 模型的完整训练流程

set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 日志文件
LOG_FILE="$LOG_DIR/train_test_${TIMESTAMP}.log"

# 模型配置
MODEL_NAME="MViTv2_base"
CONFIG_FILE="$PROJECT_ROOT/ppcls/configs/ImageNet/MViTv2/MViTv2_base.yaml"
OUTPUT_DIR="$PROJECT_ROOT/output/MViTv2_base"

# 训练参数
EPOCHS=2
BATCH_SIZE=8
NUM_GPU=1
GPU_ID=0

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 未找到，请先安装"
        return 1
    fi
    return 0
}

# 检查环境
check_environment() {
    log_info "检查环境..."
    
    # 检查 CUDA
    if command -v nvidia-smi &> /dev/null; then
        log_info "CUDA 版本: $(nvidia-smi | grep "CUDA Version" | awk '{print $9}')"
        log_info "GPU 信息:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG_FILE"
    else
        log_warn "未检测到 nvidia-smi，可能使用 CPU 训练"
    fi
    
    # 检查 PaddlePaddle
    if python -c "import paddle; print(f'PaddlePaddle 版本: {paddle.__version__}')" 2>&1 | tee -a "$LOG_FILE"; then
        log_info "PaddlePaddle 环境正常"
    else
        log_error "PaddlePaddle 导入失败"
        return 1
    fi
    
    # 检查 CUDA 可用性
    python << EOF | tee -a "$LOG_FILE"
import paddle
if paddle.is_compiled_with_cuda():
    print(f"CUDA 可用: True")
    print(f"CUDA 设备数量: {paddle.device.cuda.device_count()}")
else:
    print(f"CUDA 可用: False")
EOF
    
    return 0
}

# 检查数据集
check_dataset() {
    log_info "检查数据集..."
    
    DATASET_ROOT="$PROJECT_ROOT/dataset/ILSVRC2012"
    
    if [ ! -d "$DATASET_ROOT" ]; then
        log_warn "数据集目录不存在: $DATASET_ROOT"
        log_warn "请先准备 ImageNet 数据集"
        log_info "可以使用 prepare.sh 脚本准备测试数据集"
        return 1
    fi
    
    # 检查训练数据
    if [ -f "$DATASET_ROOT/train_list.txt" ]; then
        TRAIN_SAMPLES=$(wc -l < "$DATASET_ROOT/train_list.txt")
        log_info "训练样本数: $TRAIN_SAMPLES"
    else
        log_warn "训练列表文件不存在: $DATASET_ROOT/train_list.txt"
    fi
    
    # 检查验证数据
    if [ -f "$DATASET_ROOT/val_list.txt" ]; then
        VAL_SAMPLES=$(wc -l < "$DATASET_ROOT/val_list.txt")
        log_info "验证样本数: $VAL_SAMPLES"
    else
        log_warn "验证列表文件不存在: $DATASET_ROOT/val_list.txt"
    fi
    
    return 0
}

# 训练模型
train_model() {
    log_info "开始训练 $MODEL_NAME 模型..."
    log_info "配置文件: $CONFIG_FILE"
    log_info "输出目录: $OUTPUT_DIR"
    log_info "训练轮数: $EPOCHS"
    log_info "批大小: $BATCH_SIZE"
    
    cd "$PROJECT_ROOT"
    
    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    export FLAGS_cudnn_deterministic=True
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # 训练命令
    log_info "执行训练命令..."
    
    python tools/train.py \
        -c "$CONFIG_FILE" \
        -o Global.epochs=$EPOCHS \
        -o Global.output_dir="$OUTPUT_DIR" \
        -o DataLoader.Train.sampler.batch_size=$BATCH_SIZE \
        -o DataLoader.Train.loader.num_workers=0 \
        -o DataLoader.Train.loader.use_shared_memory=False \
        -o Global.eval_during_train=True \
        -o Global.eval_interval=1 \
        -o Global.print_batch_step=10 \
        2>&1 | tee -a "$LOG_FILE"
    
    # 检查训练是否成功
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_info "训练成功完成"
        
        # 列出生成的模型文件
        if [ -d "$OUTPUT_DIR" ]; then
            log_info "生成的模型文件:"
            find "$OUTPUT_DIR" -name "*.pdparams" -o -name "*.pdopt" | tee -a "$LOG_FILE"
        fi
    else
        log_error "训练失败"
        return 1
    fi
    
    return 0
}

# 评估模型
eval_model() {
    log_info "开始评估模型..."
    
    # 查找最新的模型文件
    LATEST_MODEL=$(find "$OUTPUT_DIR" -name "*.pdparams" | sort | tail -n 1)
    
    if [ -z "$LATEST_MODEL" ]; then
        log_error "未找到模型文件: $OUTPUT_DIR"
        return 1
    fi
    
    log_info "使用模型: $LATEST_MODEL"
    
    cd "$PROJECT_ROOT"
    
    # 评估命令
    python tools/eval.py \
        -c "$CONFIG_FILE" \
        -o Global.pretrained_model="${LATEST_MODEL%.pdparams}" \
        2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_info "评估成功完成"
    else
        log_error "评估失败"
        return 1
    fi
    
    return 0
}

# 导出模型
export_model() {
    log_info "开始导出模型..."
    
    # 查找最新的模型文件
    LATEST_MODEL=$(find "$OUTPUT_DIR" -name "*.pdparams" | sort | tail -n 1)
    
    if [ -z "$LATEST_MODEL" ]; then
        log_error "未找到模型文件: $OUTPUT_DIR"
        return 1
    fi
    
    log_info "使用模型: $LATEST_MODEL"
    
    INFER_DIR="$PROJECT_ROOT/inference/MViTv2_base"
    
    cd "$PROJECT_ROOT"
    
    # 导出命令
    python tools/export_model.py \
        -c "$CONFIG_FILE" \
        -o Global.pretrained_model="${LATEST_MODEL%.pdparams}" \
        -o Global.save_inference_dir="$INFER_DIR" \
        2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_info "导出成功完成"
        log_info "推理模型保存在: $INFER_DIR"
    else
        log_error "导出失败"
        return 1
    fi
    
    return 0
}

# 主函数
main() {
    log_info "======================================"
    log_info "MViTv2 训练测试脚本"
    log_info "======================================"
    log_info "时间: $(date)"
    log_info "项目根目录: $PROJECT_ROOT"
    log_info ""
    
    # 检查环境
    if ! check_environment; then
        log_error "环境检查失败"
        exit 1
    fi
    
    echo ""
    
    # 检查数据集
    if ! check_dataset; then
        log_warn "数据集检查失败，继续尝试训练..."
    fi
    
    echo ""
    
    # 训练模型
    if ! train_model; then
        log_error "训练失败，退出"
        exit 1
    fi
    
    echo ""
    
    # 评估模型
    if ! eval_model; then
        log_warn "评估失败，但训练已完成"
    fi
    
    echo ""
    
    # 导出模型
    if ! export_model; then
        log_warn "导出失败，但训练已完成"
    fi
    
    echo ""
    log_info "======================================"
    log_info "训练测试完成！"
    log_info "======================================"
    log_info "日志文件: $LOG_FILE"
    log_info "输出目录: $OUTPUT_DIR"
}

# 执行主函数
main "$@"
