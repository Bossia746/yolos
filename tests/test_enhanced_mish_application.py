# -*- coding: utf-8 -*-
"""
增强版Mish激活函数应用测试
验证Mish激活函数在各个模块中的集成效果和性能提升
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Any
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models.enhanced_mish_activation import (
    EnhancedMish, MishVariants, ActivationReplacer, 
    MishOptimizer, create_mish, replace_activations_with_mish
)
from src.models.adaptive_roi_application import AdaptiveROIApplication
from src.models.c3ghost_module import C3GhostModule
from src.models.advanced_yolo_optimizations import IGDModule, SimAMAttention

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MishApplicationTester:
    """Mish激活函数应用测试器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 测试配置
        self.test_configs = {
            'batch_size': 4,
            'input_size': (3, 320, 320),
            'num_classes': 10,
            'test_iterations': 100
        }
    
    def test_mish_variants(self) -> Dict[str, Any]:
        """测试不同Mish变体的性能"""
        logger.info("开始测试Mish变体性能...")
        
        variants = {
            'standard': MishVariants.standard_mish(),
            'fast': MishVariants.fast_mish(),
            'adaptive': MishVariants.adaptive_mish(learnable=True),
            'quantized': MishVariants.quantized_mish()
        }
        
        results = {}
        test_input = torch.randn(32, 256, 64, 64).to(self.device)
        
        for name, mish_func in variants.items():
            mish_func = mish_func.to(self.device)
            
            # 预热
            for _ in range(10):
                with torch.no_grad():
                    _ = mish_func(test_input)
            
            # 性能测试
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            start_time = time.time()
            
            for _ in range(self.test_configs['test_iterations']):
                with torch.no_grad():
                    output = mish_func(test_input)
            
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / self.test_configs['test_iterations']
            
            # 计算输出统计
            output_mean = output.mean().item()
            output_std = output.std().item()
            output_max = output.max().item()
            output_min = output.min().item()
            
            results[name] = {
                'avg_time_ms': avg_time * 1000,
                'output_stats': {
                    'mean': output_mean,
                    'std': output_std,
                    'max': output_max,
                    'min': output_min
                },
                'memory_usage_mb': torch.cuda.memory_allocated() / 1024**2 if self.device.type == 'cuda' else 0
            }
            
            logger.info(f"{name} Mish - 平均时间: {avg_time*1000:.3f}ms, 输出均值: {output_mean:.4f}")
        
        return results
    
    def test_adaptive_roi_with_mish(self) -> Dict[str, Any]:
        """测试集成Mish的自适应ROI应用"""
        logger.info("测试自适应ROI应用中的Mish集成...")
        
        try:
            # 创建自适应ROI应用实例
            roi_app = AdaptiveROIApplication(
                input_channels=3,
                num_classes=self.test_configs['num_classes'],
                roi_size=32
            ).to(self.device)
            
            # 测试输入
            batch_size = self.test_configs['batch_size']
            input_tensor = torch.randn(batch_size, *self.test_configs['input_size']).to(self.device)
            
            # 前向传播测试
            roi_app.eval()
            with torch.no_grad():
                start_time = time.time()
                
                for _ in range(50):
                    outputs = roi_app(input_tensor)
                
                end_time = time.time()
            
            avg_inference_time = (end_time - start_time) / 50
            
            # 检查输出
            roi_predictions = outputs['roi_predictions']
            confidence_scores = outputs['confidence_scores']
            
            results = {
                'success': True,
                'avg_inference_time_ms': avg_inference_time * 1000,
                'roi_predictions_shape': list(roi_predictions.shape),
                'confidence_scores_shape': list(confidence_scores.shape),
                'roi_predictions_stats': {
                    'mean': roi_predictions.mean().item(),
                    'std': roi_predictions.std().item(),
                    'min': roi_predictions.min().item(),
                    'max': roi_predictions.max().item()
                },
                'confidence_stats': {
                    'mean': confidence_scores.mean().item(),
                    'std': confidence_scores.std().item(),
                    'min': confidence_scores.min().item(),
                    'max': confidence_scores.max().item()
                },
                'memory_usage_mb': torch.cuda.memory_allocated() / 1024**2 if self.device.type == 'cuda' else 0
            }
            
            logger.info(f"自适应ROI测试成功 - 推理时间: {avg_inference_time*1000:.2f}ms")
            
        except Exception as e:
            logger.error(f"自适应ROI测试失败: {str(e)}")
            results = {
                'success': False,
                'error': str(e)
            }
        
        return results
    
    def test_c3ghost_with_mish(self) -> Dict[str, Any]:
        """测试集成Mish的C3Ghost模块"""
        logger.info("测试C3Ghost模块中的Mish集成...")
        
        try:
            # 创建C3Ghost模块实例
            c3ghost = C3GhostModule(
                c1=128,
                c2=256,
                n=3,
                shortcut=True,
                g=1,
                e=0.5
            ).to(self.device)
            
            # 测试输入
            test_input = torch.randn(4, 128, 64, 64).to(self.device)
            
            # 前向传播测试
            c3ghost.eval()
            with torch.no_grad():
                start_time = time.time()
                
                for _ in range(100):
                    output = c3ghost(test_input)
                
                end_time = time.time()
            
            avg_inference_time = (end_time - start_time) / 100
            
            results = {
                'success': True,
                'avg_inference_time_ms': avg_inference_time * 1000,
                'input_shape': list(test_input.shape),
                'output_shape': list(output.shape),
                'output_stats': {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item()
                },
                'memory_usage_mb': torch.cuda.memory_allocated() / 1024**2 if self.device.type == 'cuda' else 0
            }
            
            logger.info(f"C3Ghost测试成功 - 推理时间: {avg_inference_time*1000:.2f}ms")
            
        except Exception as e:
            logger.error(f"C3Ghost测试失败: {str(e)}")
            results = {
                'success': False,
                'error': str(e)
            }
        
        return results
    
    def test_igd_module_with_mish(self) -> Dict[str, Any]:
        """测试集成Mish的IGD模块"""
        logger.info("测试IGD模块中的Mish集成...")
        
        try:
            # 创建IGD模块实例
            igd_module = IGDModule(
                in_channels=[128, 256, 512],
                out_channels=256,
                enhanced_fusion=True
            ).to(self.device)
            
            # 测试输入 - 多尺度特征
            features = [
                torch.randn(2, 128, 80, 80).to(self.device),
                torch.randn(2, 256, 40, 40).to(self.device),
                torch.randn(2, 512, 20, 20).to(self.device)
            ]
            
            # 前向传播测试
            igd_module.eval()
            with torch.no_grad():
                start_time = time.time()
                
                for _ in range(50):
                    outputs = igd_module(features)
                
                end_time = time.time()
            
            avg_inference_time = (end_time - start_time) / 50
            
            results = {
                'success': True,
                'avg_inference_time_ms': avg_inference_time * 1000,
                'input_shapes': [list(f.shape) for f in features],
                'output_shapes': [list(o.shape) for o in outputs],
                'output_stats': {
                    'means': [o.mean().item() for o in outputs],
                    'stds': [o.std().item() for o in outputs],
                    'mins': [o.min().item() for o in outputs],
                    'maxs': [o.max().item() for o in outputs]
                },
                'memory_usage_mb': torch.cuda.memory_allocated() / 1024**2 if self.device.type == 'cuda' else 0
            }
            
            logger.info(f"IGD模块测试成功 - 推理时间: {avg_inference_time*1000:.2f}ms")
            
        except Exception as e:
            logger.error(f"IGD模块测试失败: {str(e)}")
            results = {
                'success': False,
                'error': str(e)
            }
        
        return results
    
    def test_activation_replacement(self) -> Dict[str, Any]:
        """测试激活函数替换功能"""
        logger.info("测试激活函数替换功能...")
        
        try:
            # 创建一个包含多种激活函数的测试模型
            class TestModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                    self.relu1 = nn.ReLU(inplace=True)
                    self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                    self.silu1 = nn.SiLU()
                    self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                    self.gelu1 = nn.GELU()
                    
                def forward(self, x):
                    x = self.relu1(self.conv1(x))
                    x = self.silu1(self.conv2(x))
                    x = self.gelu1(self.conv3(x))
                    return x
            
            # 创建原始模型
            original_model = TestModel().to(self.device)
            
            # 使用激活函数替换器
            replacer = ActivationReplacer(mish_variant='fast')
            replaced_model = replacer.replace_activations(original_model)
            
            # 测试输入
            test_input = torch.randn(2, 3, 64, 64).to(self.device)
            
            # 比较性能
            models = {'original': original_model, 'mish_replaced': replaced_model}
            results = {}
            
            for name, model in models.items():
                model.eval()
                with torch.no_grad():
                    # 预热
                    for _ in range(10):
                        _ = model(test_input)
                    
                    # 性能测试
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    start_time = time.time()
                    
                    for _ in range(100):
                        output = model(test_input)
                    
                    torch.cuda.synchronize() if self.device.type == 'cuda' else None
                    end_time = time.time()
                
                avg_time = (end_time - start_time) / 100
                
                results[name] = {
                    'avg_inference_time_ms': avg_time * 1000,
                    'output_shape': list(output.shape),
                    'output_stats': {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item()
                    }
                }
            
            # 计算性能差异
            time_diff = results['mish_replaced']['avg_inference_time_ms'] - results['original']['avg_inference_time_ms']
            time_ratio = results['mish_replaced']['avg_inference_time_ms'] / results['original']['avg_inference_time_ms']
            
            results['comparison'] = {
                'time_difference_ms': time_diff,
                'time_ratio': time_ratio,
                'performance_change': 'improved' if time_diff < 0 else 'degraded' if time_diff > 0 else 'unchanged'
            }
            
            logger.info(f"激活函数替换测试完成 - 时间差异: {time_diff:.2f}ms, 比率: {time_ratio:.3f}")
            
        except Exception as e:
            logger.error(f"激活函数替换测试失败: {str(e)}")
            results = {
                'success': False,
                'error': str(e)
            }
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行综合测试"""
        logger.info("开始运行Mish激活函数综合测试...")
        
        comprehensive_results = {
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(self.device),
            'test_configs': self.test_configs
        }
        
        # 1. Mish变体性能测试
        logger.info("\n=== 1. Mish变体性能测试 ===")
        comprehensive_results['mish_variants'] = self.test_mish_variants()
        
        # 2. 自适应ROI应用测试
        logger.info("\n=== 2. 自适应ROI应用测试 ===")
        comprehensive_results['adaptive_roi'] = self.test_adaptive_roi_with_mish()
        
        # 3. C3Ghost模块测试
        logger.info("\n=== 3. C3Ghost模块测试 ===")
        comprehensive_results['c3ghost'] = self.test_c3ghost_with_mish()
        
        # 4. IGD模块测试
        logger.info("\n=== 4. IGD模块测试 ===")
        comprehensive_results['igd_module'] = self.test_igd_module_with_mish()
        
        # 5. 激活函数替换测试
        logger.info("\n=== 5. 激活函数替换测试 ===")
        comprehensive_results['activation_replacement'] = self.test_activation_replacement()
        
        return comprehensive_results
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """生成测试报告"""
        report = []
        report.append("=" * 80)
        report.append("增强版Mish激活函数应用测试报告")
        report.append("=" * 80)
        report.append(f"测试时间: {results['test_timestamp']}")
        report.append(f"测试设备: {results['device']}")
        report.append("")
        
        # Mish变体性能报告
        if 'mish_variants' in results:
            report.append("1. Mish变体性能测试结果:")
            report.append("-" * 40)
            for variant, data in results['mish_variants'].items():
                report.append(f"  {variant.upper()} Mish:")
                report.append(f"    平均推理时间: {data['avg_time_ms']:.3f}ms")
                report.append(f"    输出统计: 均值={data['output_stats']['mean']:.4f}, 标准差={data['output_stats']['std']:.4f}")
                if data['memory_usage_mb'] > 0:
                    report.append(f"    显存使用: {data['memory_usage_mb']:.1f}MB")
                report.append("")
        
        # 自适应ROI测试报告
        if 'adaptive_roi' in results:
            report.append("2. 自适应ROI应用测试结果:")
            report.append("-" * 40)
            roi_data = results['adaptive_roi']
            if roi_data.get('success', False):
                report.append(f"  测试状态: 成功")
                report.append(f"  平均推理时间: {roi_data['avg_inference_time_ms']:.2f}ms")
                report.append(f"  ROI预测形状: {roi_data['roi_predictions_shape']}")
                report.append(f"  置信度分数形状: {roi_data['confidence_scores_shape']}")
            else:
                report.append(f"  测试状态: 失败 - {roi_data.get('error', '未知错误')}")
            report.append("")
        
        # C3Ghost测试报告
        if 'c3ghost' in results:
            report.append("3. C3Ghost模块测试结果:")
            report.append("-" * 40)
            c3_data = results['c3ghost']
            if c3_data.get('success', False):
                report.append(f"  测试状态: 成功")
                report.append(f"  平均推理时间: {c3_data['avg_inference_time_ms']:.2f}ms")
                report.append(f"  输入形状: {c3_data['input_shape']} -> 输出形状: {c3_data['output_shape']}")
            else:
                report.append(f"  测试状态: 失败 - {c3_data.get('error', '未知错误')}")
            report.append("")
        
        # IGD模块测试报告
        if 'igd_module' in results:
            report.append("4. IGD模块测试结果:")
            report.append("-" * 40)
            igd_data = results['igd_module']
            if igd_data.get('success', False):
                report.append(f"  测试状态: 成功")
                report.append(f"  平均推理时间: {igd_data['avg_inference_time_ms']:.2f}ms")
                report.append(f"  多尺度输入处理: {len(igd_data['input_shapes'])}个尺度")
            else:
                report.append(f"  测试状态: 失败 - {igd_data.get('error', '未知错误')}")
            report.append("")
        
        # 激活函数替换测试报告
        if 'activation_replacement' in results:
            report.append("5. 激活函数替换测试结果:")
            report.append("-" * 40)
            repl_data = results['activation_replacement']
            if 'comparison' in repl_data:
                comp = repl_data['comparison']
                report.append(f"  性能变化: {comp['performance_change']}")
                report.append(f"  时间差异: {comp['time_difference_ms']:.2f}ms")
                report.append(f"  性能比率: {comp['time_ratio']:.3f}")
            report.append("")
        
        report.append("=" * 80)
        report.append("测试完成")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """主函数"""
    try:
        # 创建测试器
        tester = MishApplicationTester()
        
        # 运行综合测试
        results = tester.run_comprehensive_test()
        
        # 生成并打印报告
        report = tester.generate_test_report(results)
        print(report)
        
        # 保存结果到文件
        results_file = Path('mish_application_test_results.json')
        import json
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"测试结果已保存到: {results_file}")
        
        # 计算总体成功率
        success_count = 0
        total_tests = 0
        
        test_modules = ['adaptive_roi', 'c3ghost', 'igd_module']
        for module in test_modules:
            if module in results:
                total_tests += 1
                if results[module].get('success', False):
                    success_count += 1
        
        success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0
        logger.info(f"\n总体测试成功率: {success_rate:.1f}% ({success_count}/{total_tests})")
        
        return success_rate >= 80  # 80%以上成功率认为测试通过
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)