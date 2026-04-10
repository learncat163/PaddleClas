#!/bin/bash
# MViTv2 快速验证脚本
# 用于快速验证 MViTv2_base 模型训练流程是否正常

set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 日志文件
LOG_FILE="$LOG_DIR/quick_train_test_${TIMESTAMP}.log"

# 模型配置
MODEL_NAME="MViTv2_base"
CONFIG_FILE="$PROJECT_ROOT/ppcls/configs/ImageNet/MViTv2/MViTv2_base.yaml"
OUTPUT_DIR="$PROJECT_ROOT/output/MViTv2_base_quick"

# 快速验证参数 - 使用最小配置
EPOCHS=1
BATCH_SIZE=2
NUM_GPU=1
GPU_ID=0

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1" | tee -a "$LOG_FILE"
}

# 检查环境
quick_check_environment() {
    log_step "1/4 快速环境检查..."
    
    # 检查 PaddlePaddle
    if python -c "import paddle; print(f'PaddlePaddle {paddle.__version__}')" 2>&1 | tee -a "$LOG_FILE"; then
        :
    else
        log_error "PaddlePaddle 导入失败"
        return 1
    fi
    
    # 检查 CUDA
    python << EOF | tee -a "$LOG_FILE"
import paddle
if paddle.is_compiled_with_cuda():
    print(f"CUDA 可用，GPU 数量: {paddle.device.cuda.device_count()}")
else:
    print("使用 CPU 模式")
EOF
    
    log_info "环境检查完成"
    return 0
}

# 创建最小测试数据集
create_test_data() {
    log_step "2/4 创建测试数据集..."
    
    TEST_DATA_DIR="$PROJECT_ROOT/dataset/ILSVRC2012_test"
    mkdir -p "$TEST_DATA_DIR/train"
    mkdir -p "$TEST_DATA_DIR/val"
    
    # 创建简单的测试图片（如果不存在）
    if [ ! -f "$TEST_DATA_DIR/train/test.jpg" ]; then
        log_info "创建测试图片..."
        python << EOF | tee -a "$LOG_FILE"
from PIL import Image
import os

# 创建随机图片
img = Image.new('RGB', (224, 224), color='red')
img.save(os.path.join('$TEST_DATA_DIR', 'train', 'test.jpg'))
img.save(os.path.join('$TEST_DATA_DIR', 'val', 'test.jpg'))
print("测试图片创建完成")
EOF
    fi
    
    # 创建训练列表文件
    cat > "$TEST_DATA_DIR/train_list.txt" << EOF
train/test.jpg 0
train/test.jpg 1
train/test.jpg 2
train/test.jpg 3
EOF
    
    # 创建验证列表文件
    cat > "$TEST_DATA_DIR/val_list.txt" << EOF
val/test.jpg 0
val/test.jpg 1
val/test.jpg 2
val/test.jpg 3
EOF
    
    log_info "测试数据集创建完成: $TEST_DATA_DIR"
    return 0
}

# 快速训练测试
quick_train_test() {
    log_step "3/4 快速训练测试..."
    log_info "配置文件: $CONFIG_FILE"
    log_info "训练轮数: $EPOCHS (快速测试)"
    log_info "批大小: $BATCH_SIZE (最小配置)"
    
    cd "$PROJECT_ROOT"
    
    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    export FLAGS_cudnn_deterministic=True
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # 训练命令 - 使用测试数据集
    log_info "开始训练..."
    
    python tools/train.py \
        -c "$CONFIG_FILE" \
        -o Global.epochs=$EPOCHS \
        -o Global.output_dir="$OUTPUT_DIR" \
        -o DataLoader.Train.dataset.image_root="$TEST_DATA_DIR/" \
        -o DataLoader.Train.dataset.cls_label_path="$TEST_DATA_DIR/train_list.txt" \
        -o DataLoader.Eval.dataset.image_root="$TEST_DATA_DIR/" \
        -o DataLoader.Eval.dataset.cls_label_path="$TEST_DATA_DIR/val_list.txt" \
        -o DataLoader.Train.sampler.batch_size=$BATCH_SIZE \
        -o DataLoader.Eval.sampler.batch_size=$BATCH_SIZE \
        -o DataLoader.Train.loader.num_workers=0 \
        -o DataLoader.Eval.loader.num_workers=0 \
        -o DataLoader.Train.loader.use_shared_memory=False \
        -o DataLoader.Eval.loader.use_shared_memory=False \
        -o Global.eval_during_train=True \
        -o Global.eval_interval=1 \
        -o Global.print_batch_step=1 \
        2>&1 | tee -a "$LOG_FILE"
    
    # 检查训练是否成功
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_info "快速训练测试成功！"
        
        # 列出生成的模型文件
        if [ -d "$OUTPUT_DIR" ]; then
            log_info "生成的模型文件:"
            find "$OUTPUT_DIR" -name "*.pdparams" | head -5 | tee -a "$LOG_FILE"
        fi
    else
        log_error "快速训练测试失败"
        return 1
    fi
    
    return 0
}

# 快速评估测试
quick_eval_test() {
    log_step "4/4 快速评估测试..."
    
    # 查找最新的模型文件
    LATEST_MODEL=$(find "$OUTPUT_DIR" -name "*.pdparams" | sort | tail -n 1)
    
    if [ -z "$LATEST_MODEL" ]; then
        log_warn "未找到模型文件，跳过评估"
        return 0
    fi
    
    log_info "使用模型: $LATEST_MODEL"
    
    cd "$PROJECT_ROOT"
    
    # 评估命令
    python tools/eval.py \
        -c "$CONFIG_FILE" \
        -o Global.pretrained_model="${LATEST_MODEL%.pdparams}" \
        -o DataLoader.Eval.dataset.image_root="$TEST_DATA_DIR/" \
        -o DataLoader.Eval.dataset.cls_label_path="$TEST_DATA_DIR/val_list.txt" \
        -o DataLoader.Eval.sampler.batch_size=$BATCH_SIZE \
        -o DataLoader.Eval.loader.num_workers=0 \
        -o DataLoader.Eval.loader.use_shared_memory=False \
        2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_info "快速评估测试成功！"
    else
        log_warn "快速评估测试失败"
    fi
    
    return 0
}

# 主函数
main() {
    log_info "======================================"
    log_info "MViTv2 快速验证脚本"
    log_info "======================================"
    log_info "时间: $(date)"
    log_info "项目根目录: $PROJECT_ROOT"
    log_info ""
    
    # 快速环境检查
    if ! quick_check_environment; then
        log_error "环境检查失败"
        exit 1
    fi
    
    echo ""
    
    # 创建测试数据
    if ! create_test_data; then
        log_error "创建测试数据失败"
        exit 1
    fi
    
    echo ""
    
    # 快速训练测试
    if ! quick_train_test; then
        log_error "快速训练测试失败"
        exit 1
    fi
    
    echo ""
    
    # 快速评估测试
    quick_eval_test
    
    echo ""
    log_info "======================================"
    log_info "快速验证完成！"
    log_info "======================================"
    log_info "日志文件: $LOG_FILE"
    log_info "输出目录: $OUTPUT_DIR"
    log_info ""
    log_info "如果快速验证通过，可以运行完整训练测试："
    log_info "bash tests/scripts/train_mvitv2.sh"
}

# 执行主函数
main "$@"
