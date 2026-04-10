#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MViTv2 训练测试脚本 (Python 版本)
用于测试 MViTv2_base 模型的完整训练流程
"""

import os
import sys
import subprocess
import logging
import argparse
import datetime
from pathlib import Path
from typing import Optional, Tuple


class TrainTestConfig:
    """训练测试配置"""
    
    def __init__(self):
        # 获取项目根目录
        self.script_dir = Path(__file__).parent.absolute()
        self.project_root = self.script_dir.parent.parent.parent
        self.log_dir = self.script_dir.parent / "logs"
        
        # 模型配置
        self.model_name = "MViTv2_base"
        self.config_file = self.project_root / "ppcls/configs/ImageNet/MViTv2/MViTv2_base.yaml"
        self.output_dir = self.project_root / "output" / "MViTv2_base"
        
        # 训练参数
        self.epochs = 2
        self.batch_size = 8
        self.num_gpu = 1
        self.gpu_id = "0"
        
        # 确保日志目录存在
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 日志文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"train_test_{timestamp}.log"


class TrainTestRunner:
    """训练测试运行器"""
    
    def __init__(self, config: TrainTestConfig):
        self.config = config
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志"""
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_environment(self) -> bool:
        """检查环境"""
        self.logger.info("检查环境...")
        
        try:
            # 检查 PaddlePaddle
            result = subprocess.run(
                [sys.executable, "-c", "import paddle; print(f'PaddlePaddle {paddle.__version__}')"],
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.info(result.stdout.strip())
            
            # 检查 CUDA 可用性
            check_cuda = """
import paddle
if paddle.is_compiled_with_cuda():
    print(f"CUDA 可用: True")
    print(f"CUDA 设备数量: {paddle.device.cuda.device_count()}")
else:
    print("CUDA 可用: False")
"""
            result = subprocess.run(
                [sys.executable, "-c", check_cuda],
                capture_output=True,
                text=True,
                check=True
            )
            self.logger.info(result.stdout.strip())
            
            self.logger.info("环境检查完成")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"环境检查失败: {e}")
            return False
    
    def check_dataset(self) -> bool:
        """检查数据集"""
        self.logger.info("检查数据集...")
        
        dataset_root = self.config.project_root / "dataset" / "ILSVRC2012"
        
        if not dataset_root.exists():
            self.logger.warning(f"数据集目录不存在: {dataset_root}")
            self.logger.warning("请先准备 ImageNet 数据集")
            return False
        
        # 检查训练数据
        train_list = dataset_root / "train_list.txt"
        if train_list.exists():
            train_samples = sum(1 for _ in open(train_list))
            self.logger.info(f"训练样本数: {train_samples}")
        else:
            self.logger.warning(f"训练列表文件不存在: {train_list}")
        
        # 检查验证数据
        val_list = dataset_root / "val_list.txt"
        if val_list.exists():
            val_samples = sum(1 for _ in open(val_list))
            self.logger.info(f"验证样本数: {val_samples}")
        else:
            self.logger.warning(f"验证列表文件不存在: {val_list}")
        
        return True
    
    def train_model(self) -> bool:
        """训练模型"""
        self.logger.info("开始训练 MViTv2_base 模型...")
        self.logger.info(f"配置文件: {self.config.config_file}")
        self.logger.info(f"输出目录: {self.config.output_dir}")
        self.logger.info(f"训练轮数: {self.config.epochs}")
        self.logger.info(f"批大小: {self.config.batch_size}")
        
        # 切换到项目目录
        os.chdir(self.config.project_root)
        
        # 设置环境变量
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.config.gpu_id
        env["FLAGS_cudnn_deterministic"] = "True"
        env["PYTHONPATH"] = str(self.config.project_root) + ":" + env.get("PYTHONPATH", "")
        
        # 训练命令
        cmd = [
            sys.executable, "tools/train.py",
            "-c", str(self.config.config_file),
            "-o", f"Global.epochs={self.config.epochs}",
            "-o", f"Global.output_dir={self.config.output_dir}",
            "-o", f"DataLoader.Train.sampler.batch_size={self.config.batch_size}",
            "-o", "DataLoader.Train.loader.num_workers=0",
            "-o", "DataLoader.Train.loader.use_shared_memory=False",
            "-o", "Global.eval_during_train=True",
            "-o", "Global.eval_interval=1",
            "-o", "Global.print_batch_step=10",
        ]
        
        self.logger.info(f"执行训练命令...")
        self.logger.debug(f"命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                env=env,
                check=True,
                text=True,
                capture_output=True
            )
            self.logger.info(result.stdout)
            
            # 检查模型文件
            if self.config.output_dir.exists():
                pdparams_files = list(self.config.output_dir.rglob("*.pdparams"))
                if pdparams_files:
                    self.logger.info("生成的模型文件:")
                    for f in pdparams_files[:5]:  # 只显示前5个
                        self.logger.info(f"  - {f}")
            
            self.logger.info("训练成功完成")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"训练失败: {e}")
            if e.stderr:
                self.logger.error(e.stderr)
            return False
    
    def eval_model(self) -> bool:
        """评估模型"""
        self.logger.info("开始评估模型...")
        
        # 查找最新的模型文件
        pdparams_files = list(self.config.output_dir.rglob("*.pdparams"))
        if not pdparams_files:
            self.logger.error(f"未找到模型文件: {self.config.output_dir}")
            return False
        
        latest_model = sorted(pdparams_files)[-1]
        self.logger.info(f"使用模型: {latest_model}")
        
        # 切换到项目目录
        os.chdir(self.config.project_root)
        
        # 设置环境变量
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.config.gpu_id
        env["PYTHONPATH"] = str(self.config.project_root) + ":" + env.get("PYTHONPATH", "")
        
        # 评估命令
        pretrained_model = str(latest_model.with_suffix(''))
        cmd = [
            sys.executable, "tools/eval.py",
            "-c", str(self.config.config_file),
            "-o", f"Global.pretrained_model={pretrained_model}",
        ]
        
        try:
            result = subprocess.run(
                cmd,
                env=env,
                check=True,
                text=True,
                capture_output=True
            )
            self.logger.info(result.stdout)
            self.logger.info("评估成功完成")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"评估失败: {e}")
            if e.stderr:
                self.logger.error(e.stderr)
            return False
    
    def export_model(self) -> bool:
        """导出模型"""
        self.logger.info("开始导出模型...")
        
        # 查找最新的模型文件
        pdparams_files = list(self.config.output_dir.rglob("*.pdparams"))
        if not pdparams_files:
            self.logger.error(f"未找到模型文件: {self.config.output_dir}")
            return False
        
        latest_model = sorted(pdparams_files)[-1]
        self.logger.info(f"使用模型: {latest_model}")
        
        # 切换到项目目录
        os.chdir(self.config.project_root)
        
        # 设置环境变量
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.config.project_root) + ":" + env.get("PYTHONPATH", "")
        
        infer_dir = self.config.project_root / "inference" / "MViTv2_base"
        
        # 导出命令
        pretrained_model = str(latest_model.with_suffix(''))
        cmd = [
            sys.executable, "tools/export_model.py",
            "-c", str(self.config.config_file),
            "-o", f"Global.pretrained_model={pretrained_model}",
            "-o", f"Global.save_inference_dir={infer_dir}",
        ]
        
        try:
            result = subprocess.run(
                cmd,
                env=env,
                check=True,
                text=True,
                capture_output=True
            )
            self.logger.info(result.stdout)
            self.logger.info(f"推理模型保存在: {infer_dir}")
            self.logger.info("导出成功完成")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"导出失败: {e}")
            if e.stderr:
                self.logger.error(e.stderr)
            return False
    
    def run(self, skip_eval: bool = False, skip_export: bool = False) -> int:
        """运行完整的训练测试流程"""
        self.logger.info("=" * 60)
        self.logger.info("MViTv2 训练测试脚本 (Python 版本)")
        self.logger.info("=" * 60)
        self.logger.info(f"时间: {datetime.datetime.now()}")
        self.logger.info(f"项目根目录: {self.config.project_root}")
        self.logger.info("")
        
        # 检查环境
        if not self.check_environment():
            self.logger.error("环境检查失败")
            return 1
        
        self.logger.info("")
        
        # 检查数据集
        self.check_dataset()
        
        self.logger.info("")
        
        # 训练模型
        if not self.train_model():
            self.logger.error("训练失败，退出")
            return 1
        
        self.logger.info("")
        
        # 评估模型
        if not skip_eval:
            if not self.eval_model():
                self.logger.warning("评估失败，但训练已完成")
        else:
            self.logger.info("跳过评估")
        
        self.logger.info("")
        
        # 导出模型
        if not skip_export:
            if not self.export_model():
                self.logger.warning("导出失败，但训练已完成")
        else:
            self.logger.info("跳过导出")
        
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("训练测试完成！")
        self.logger.info("=" * 60)
        self.logger.info(f"日志文件: {self.config.log_file}")
        self.logger.info(f"输出目录: {self.config.output_dir}")
        
        return 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MViTv2 训练测试脚本")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=8, help="批大小")
    parser.add_argument("--gpu-id", type=str, default="0", help="GPU ID")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评估")
    parser.add_argument("--skip-export", action="store_true", help="跳过导出")
    
    args = parser.parse_args()
    
    # 创建配置
    config = TrainTestConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.gpu_id = args.gpu_id
    
    # 创建运行器并运行
    runner = TrainTestRunner(config)
    return runner.run(skip_eval=args.skip_eval, skip_export=args.skip_export)


if __name__ == "__main__":
    sys.exit(main())
