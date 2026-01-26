#!/usr/bin/env python
import os
import argparse
import paddle
import traceback
def load_pytorch_weights(model_name):
    try:
        import torch
        import timm
    except ImportError:
        raise ImportError("需要安装 torch 和 timm: pip install torch timm")

    print(f"从 timm 加载 {model_name} 模型...")
    pt_model = timm.create_model(model_name, pretrained=True)
    pt_state_dict = pt_model.state_dict()

    print(f"PyTorch 模型参数数量: {len(pt_state_dict)}")
    return pt_state_dict


def convert_name(pt_name):
    if 'num_batches_tracked' in pt_name:
        return None

    name = pt_name.replace('running_mean', '_mean')
    name = name.replace('running_var', '_variance')
    
    return name


def debug_weight_mapping(pt_state_dict, model_name):
    print(f"\n=== 调试模式: {model_name} ===")
    
    aggreg_keys = [k for k in pt_state_dict.keys() if 'aggreg' in k]
    if aggreg_keys:
        print(f"发现 {len(aggreg_keys)} 个aggreg相关参数:")
        for key in aggreg_keys[:10]:  
            print(f"  {key}")
        if len(aggreg_keys) > 10:
            print(f"  ... 还有 {len(aggreg_keys)-10} 个")
    
    attention_keys = [k for k in pt_state_dict.keys() if any(x in k for x in ['qkv', 'proj', 'kernel_func'])]
    if attention_keys:
        print(f"发现 {len(attention_keys)} 个attention相关参数:")
        for key in attention_keys[:10]:
            print(f"  {key}")
        if len(attention_keys) > 10:
            print(f"  ... 还有 {len(attention_keys)-10} 个")
    
    print(f"总参数数量: {len(pt_state_dict)}")
    print("=" * 50)


def is_large_model(model_name):
    return 'efficientvit_l' in model_name.lower()


def convert_param(pt_name, pt_param):
    param = pt_param.cpu().numpy()

    if 'weight' in pt_name and param.ndim == 2:
        if 'conv' not in pt_name.lower() and 'norm' not in pt_name.lower():
            param = param.T
    return param


def convert_state_dict(pt_state_dict, model_name=None, debug=False):
    pd_state_dict = {}
    skipped = []
    
    is_large = is_large_model(model_name) if model_name else False
    
    if debug or is_large:
        debug_weight_mapping(pt_state_dict, model_name)

    for pt_name, pt_param in pt_state_dict.items():
        pd_name = convert_name(pt_name)
        if pd_name is None:
            skipped.append(pt_name)
            continue

        pd_param = convert_param(pt_name, pt_param)
        pd_state_dict[pd_name] = pd_param

    print(f"转换完成: {len(pd_state_dict)} 个参数")
    print(f"模型类型: {'Large系列' if is_large else 'Base系列'}")
    if skipped:
        print(f"跳过的参数 ({len(skipped)}): {skipped[:5]}...")

    return pd_state_dict


def save_weights(pd_state_dict, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    paddle.save(pd_state_dict, output_path)
    print(f"\n权重已保存到: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


model_keys = [
    'efficientvit_b0.r224_in1k',
    'efficientvit_b1.r224_in1k',
    'efficientvit_b1.r256_in1k',
    'efficientvit_b1.r288_in1k',
    'efficientvit_b2.r224_in1k',
    'efficientvit_b2.r256_in1k',
    'efficientvit_b2.r288_in1k',
    'efficientvit_b3.r224_in1k',
    'efficientvit_b3.r256_in1k',
    'efficientvit_b3.r288_in1k',
    'efficientvit_l1.r224_in1k',
    'efficientvit_l2.r224_in1k',
    'efficientvit_l2.r256_in1k',
    'efficientvit_l2.r288_in1k',
    'efficientvit_l2.r384_in1k',
    'efficientvit_l3.r224_in1k',
    'efficientvit_l3.r256_in1k',
    'efficientvit_l3.r320_in1k',
    'efficientvit_l3.r384_in1k'
]

def main():
    parser = argparse.ArgumentParser(description='转换 timm EfficientViT 权重到 PaddlePaddle')
    parser.add_argument('--model', type=str, required=False,
                        help='模型名称，如果不指定则转换所有支持的模型')
    parser.add_argument('--output', type=str, default='weights/',
                        help='输出目录')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式，显示详细的权重映射信息')

    args = parser.parse_args()

    # 如果没有指定模型，则遍历所有支持的模型
    if args.model is None or args.model == "":
        for model_name in model_keys:
            print(f"\n正在转换模型: {model_name}")
            try:
                pt_state_dict = load_pytorch_weights(model_name)
                pd_state_dict = convert_state_dict(pt_state_dict, model_name, args.debug)
                output_path = os.path.join(args.output, f'{model_name}.pdparams')
                save_weights(pd_state_dict, output_path)
            except Exception as e:
                print(f"转换 {model_name} 时出错: {str(e)}")
                print(f"详细堆栈信息:\n{traceback.format_exc()}")
                continue
    else:
        # 单个模型转换
        if args.model not in model_keys:
            print(f"错误: 模型 {args.model} 不在支持的模型列表中")
            print(f"支持的模型: {model_keys}")
            return

        pt_state_dict = load_pytorch_weights(args.model)
        pd_state_dict = convert_state_dict(pt_state_dict, args.model, args.debug)
        output_path = os.path.join(args.output, f'{args.model}.pdparams')
        save_weights(pd_state_dict, output_path)

    print("\n" + "=" * 60)
    print("转换完成")
    print("\n使用方法:")
    print(f"  model = efficientvit_b0()")
    print(f"  state_dict = paddle.load('{output_path}')")
    print(f"  model.set_state_dict(state_dict)")
    print("=" * 60)


if __name__ == '__main__':
    main()
