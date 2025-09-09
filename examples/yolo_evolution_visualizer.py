#!/usr/bin/env python3
"""
YOLO算法演进可视化工具
展示YOLO系列算法的发展历程和性能对比
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from core.logger import get_logger
except ImportError:
    # 简单的日志记录
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
else:
    logger = get_logger(__name__)

class YOLOEvolutionVisualizer:
    """YOLO算法演进可视化器"""
    
    def __init__(self):
        self.yolo_data = self._load_yolo_evolution_data()
        self.output_dir = Path('docs/visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_yolo_evolution_data(self):
        """加载YOLO演进数据"""
        return {
            'YOLOv1': {
                'year': 2016,
                'mAP_coco': 63.4,
                'fps': 45,
                'params_m': 235,
                'flops_g': 68.2,
                'key_features': ['单阶段检测', '全卷积网络', '端到端训练'],
                'innovations': ['统一检测框架', '实时检测']
            },
            'YOLOv2': {
                'year': 2017,
                'mAP_coco': 76.8,
                'fps': 67,
                'params_m': 50.7,
                'flops_g': 34.9,
                'key_features': ['Anchor boxes', 'Batch normalization', 'High resolution classifier'],
                'innovations': ['维度聚类', '直接位置预测', '细粒度特征']
            },
            'YOLOv3': {
                'year': 2018,
                'mAP_coco': 55.3,
                'fps': 20,
                'params_m': 61.9,
                'flops_g': 65.9,
                'key_features': ['多尺度预测', 'Darknet-53', '特征金字塔'],
                'innovations': ['三个尺度检测', '残差连接', '更好的小目标检测']
            },
            'YOLOv4': {
                'year': 2020,
                'mAP_coco': 65.7,
                'fps': 65,
                'params_m': 64.0,
                'flops_g': 59.6,
                'key_features': ['CSPDarknet53', 'PANet', 'SPP'],
                'innovations': ['Bag of Freebies', 'Bag of Specials', 'Mosaic数据增强']
            },
            'YOLOv5': {
                'year': 2020,
                'mAP_coco': 56.8,
                'fps': 140,
                'params_m': 46.5,
                'flops_g': 50.7,
                'key_features': ['Focus层', 'CSP结构', 'Auto-anchor'],
                'innovations': ['PyTorch实现', '模型缩放', '自动超参数优化']
            },
            'YOLOv6': {
                'year': 2022,
                'mAP_coco': 57.2,
                'fps': 1234,
                'params_m': 18.5,
                'flops_g': 45.3,
                'key_features': ['EfficientRep', 'RepVGG', 'SimOTA'],
                'innovations': ['重参数化', '高效训练', '标签分配优化']
            },
            'YOLOv7': {
                'year': 2022,
                'mAP_coco': 56.8,
                'fps': 161,
                'params_m': 36.9,
                'flops_g': 104.7,
                'key_features': ['E-ELAN', '模型缩放', '复合缩放'],
                'innovations': ['可训练的bag-of-freebies', '扩展高效层聚合网络']
            },
            'YOLOv8': {
                'year': 2023,
                'mAP_coco': 53.9,
                'fps': 280,
                'params_m': 25.9,
                'flops_g': 78.9,
                'key_features': ['Anchor-free', 'C2f模块', '解耦头'],
                'innovations': ['无锚点设计', '改进的特征融合', '统一的检测头']
            },
            'YOLOv9': {
                'year': 2024,
                'mAP_coco': 53.0,
                'fps': 227,
                'params_m': 25.3,
                'flops_g': 76.8,
                'key_features': ['PGI', 'GELAN', '可逆函数'],
                'innovations': ['可编程梯度信息', '广义高效层聚合网络']
            },
            'YOLOv10': {
                'year': 2024,
                'mAP_coco': 54.4,
                'fps': 300,
                'params_m': 24.4,
                'flops_g': 70.5,
                'key_features': ['NMS-free', '双重标签分配', 'Consistent dual assignments'],
                'innovations': ['无NMS训练', '效率优化', '一致性双重分配']
            },
            'YOLOv11': {
                'year': 2024,
                'mAP_coco': 55.2,
                'fps': 320,
                'params_m': 20.1,
                'flops_g': 65.2,
                'key_features': ['C3k2', '改进的SPPF', '增强的特征融合'],
                'innovations': ['更高效的架构', '改进的多尺度特征融合', '优化的计算效率']
            }
        }
        
    def plot_performance_evolution(self):
        """绘制性能演进图"""
        print("📊 生成性能演进图...")
        
        # 准备数据
        versions = list(self.yolo_data.keys())
        years = [self.yolo_data[v]['year'] for v in versions]
        map_scores = [self.yolo_data[v]['mAP_coco'] for v in versions]
        fps_scores = [self.yolo_data[v]['fps'] for v in versions]
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('YOLO算法演进分析', fontsize=16, fontweight='bold')
        
        # 1. mAP演进
        ax1.plot(years, map_scores, 'o-', linewidth=2, markersize=8, color='#2E86AB')
        ax1.set_title('mAP@0.5:0.95 演进', fontweight='bold')
        ax1.set_xlabel('年份')
        ax1.set_ylabel('mAP (%)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(50, 80)
        
        # 添加版本标签
        for i, (year, map_val, version) in enumerate(zip(years, map_scores, versions)):
            ax1.annotate(version, (year, map_val), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        # 2. FPS演进
        ax2.plot(years, fps_scores, 's-', linewidth=2, markersize=8, color='#A23B72')
        ax2.set_title('推理速度(FPS)演进', fontweight='bold')
        ax2.set_xlabel('年份')
        ax2.set_ylabel('FPS')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # 使用对数刻度
        
        for i, (year, fps_val, version) in enumerate(zip(years, fps_scores, versions)):
            ax2.annotate(version, (year, fps_val), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        # 3. 参数量对比
        params = [self.yolo_data[v]['params_m'] for v in versions]
        colors = plt.cm.viridis(np.linspace(0, 1, len(versions)))
        bars = ax3.bar(range(len(versions)), params, color=colors)
        ax3.set_title('模型参数量对比', fontweight='bold')
        ax3.set_xlabel('YOLO版本')
        ax3.set_ylabel('参数量 (M)')
        ax3.set_xticks(range(len(versions)))
        ax3.set_xticklabels(versions, rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, param in zip(bars, params):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{param:.1f}M', ha='center', va='bottom')
        
        # 4. 效率对比 (mAP vs FPS)
        scatter = ax4.scatter(fps_scores, map_scores, 
                            s=[p*3 for p in params], 
                            c=years, cmap='plasma', alpha=0.7)
        ax4.set_title('精度 vs 速度权衡', fontweight='bold')
        ax4.set_xlabel('FPS (对数刻度)')
        ax4.set_ylabel('mAP (%)')
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
        
        # 添加版本标签
        for i, (fps, map_val, version) in enumerate(zip(fps_scores, map_scores, versions)):
            ax4.annotate(version, (fps, map_val), 
                        textcoords="offset points", xytext=(5,5), ha='left')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('年份')
        
        plt.tight_layout()
        
        # 保存图片
        output_path = self.output_dir / 'yolo_performance_evolution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ 性能演进图已保存: {output_path}")
        
        return fig
        
    def plot_architecture_comparison(self):
        """绘制架构对比图"""
        print("🏗️ 生成架构对比图...")
        
        # 创建架构特征对比
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('YOLO架构特征演进', fontsize=16, fontweight='bold')
        
        # 1. 计算复杂度对比
        versions = list(self.yolo_data.keys())
        flops = [self.yolo_data[v]['flops_g'] for v in versions]
        params = [self.yolo_data[v]['params_m'] for v in versions]
        
        # 创建气泡图
        colors = plt.cm.Set3(np.linspace(0, 1, len(versions)))
        for i, (version, f, p, color) in enumerate(zip(versions, flops, params, colors)):
            ax1.scatter(f, p, s=200, alpha=0.7, color=color, label=version)
            ax1.annotate(version, (f, p), ha='center', va='center', fontweight='bold')
        
        ax1.set_title('计算复杂度对比', fontweight='bold')
        ax1.set_xlabel('FLOPs (G)')
        ax1.set_ylabel('参数量 (M)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. 关键创新时间线
        innovations_timeline = {}
        for version, data in self.yolo_data.items():
            year = data['year']
            if year not in innovations_timeline:
                innovations_timeline[year] = []
            innovations_timeline[year].extend(data['innovations'])
        
        years = sorted(innovations_timeline.keys())
        y_pos = 0
        colors_timeline = plt.cm.tab10(np.linspace(0, 1, len(years)))
        
        for i, (year, color) in enumerate(zip(years, colors_timeline)):
            innovations = innovations_timeline[year]
            ax2.barh(y_pos, 1, color=color, alpha=0.7, height=0.8)
            
            # 添加年份标签
            ax2.text(0.5, y_pos, str(year), ha='center', va='center', 
                    fontweight='bold', fontsize=12)
            
            # 添加创新点
            innovation_text = '\n'.join(innovations[:2])  # 只显示前两个创新点
            ax2.text(1.1, y_pos, innovation_text, ha='left', va='center', 
                    fontsize=9, wrap=True)
            
            y_pos += 1
        
        ax2.set_title('关键创新时间线', fontweight='bold')
        ax2.set_xlabel('时间进展')
        ax2.set_yticks(range(len(years)))
        ax2.set_yticklabels([f'第{i+1}代' for i in range(len(years))])
        ax2.set_xlim(0, 3)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # 保存图片
        output_path = self.output_dir / 'yolo_architecture_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ 架构对比图已保存: {output_path}")
        
        return fig
        
    def generate_performance_table(self):
        """生成性能对比表格"""
        print("📋 生成性能对比表格...")
        
        # 创建DataFrame
        df_data = []
        for version, data in self.yolo_data.items():
            df_data.append({
                '版本': version,
                '年份': data['year'],
                'mAP(%)': data['mAP_coco'],
                'FPS': data['fps'],
                '参数量(M)': data['params_m'],
                'FLOPs(G)': data['flops_g'],
                '效率比': round(data['mAP_coco'] / data['params_m'], 2),
                '速度比': round(data['fps'] / data['params_m'], 2)
            })
        
        df = pd.DataFrame(df_data)
        
        # 保存为CSV
        csv_path = self.output_dir / 'yolo_performance_comparison.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 保存为HTML表格
        html_path = self.output_dir / 'yolo_performance_table.html'
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>YOLO性能对比表</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .best {{ background-color: #d4edda; font-weight: bold; }}
                .title {{ text-align: center; color: #333; }}
            </style>
        </head>
        <body>
            <h1 class="title">YOLO算法性能对比表</h1>
            <p class="title">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            {df.to_html(index=False, classes='table', escape=False)}
            
            <h2>说明</h2>
            <ul>
                <li><strong>效率比</strong>: mAP/参数量，数值越高表示模型越高效</li>
                <li><strong>速度比</strong>: FPS/参数量，数值越高表示推理速度相对参数量越快</li>
                <li>数据来源于各版本官方论文和测试报告</li>
            </ul>
        </body>
        </html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"   ✅ CSV表格已保存: {csv_path}")
        print(f"   ✅ HTML表格已保存: {html_path}")
        
        # 打印控制台表格
        print("\n📊 YOLO性能对比表:")
        print(df.to_string(index=False))
        
        return df
        
    def create_optimization_roadmap(self):
        """创建优化路线图"""
        print("🗺️ 生成优化路线图...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 定义优化维度
        optimization_areas = {
            '网络架构': ['Darknet', 'CSPNet', 'EfficientNet', 'RepVGG', 'C2f', 'C3k2'],
            '训练策略': ['基础训练', 'Mosaic增强', 'MixUp', 'CutMix', 'Auto-anchor', 'SimOTA'],
            '推理优化': ['基础推理', '模型剪枝', '量化', 'TensorRT', 'ONNX', 'OpenVINO'],
            '检测头设计': ['耦合头', '解耦头', 'Anchor-based', 'Anchor-free', 'NMS-free'],
            '特征融合': ['FPN', 'PANet', 'BiFPN', 'SPPF', 'GELAN', 'PGI']
        }
        
        # 创建时间线
        years = list(range(2016, 2025))
        y_positions = list(range(len(optimization_areas)))
        
        # 绘制优化路线
        colors = plt.cm.Set2(np.linspace(0, 1, len(optimization_areas)))
        
        for i, (area, techniques) in enumerate(optimization_areas.items()):
            # 绘制主线
            ax.plot(years, [i] * len(years), '-', color=colors[i], 
                   linewidth=3, alpha=0.7, label=area)
            
            # 添加技术节点
            for j, technique in enumerate(techniques):
                year = 2016 + j * 1.5  # 分布在时间线上
                if year <= 2024:
                    ax.scatter(year, i, s=100, color=colors[i], 
                             alpha=0.8, zorder=5)
                    ax.annotate(technique, (year, i), 
                              textcoords="offset points", 
                              xytext=(0, 15), ha='center', 
                              fontsize=8, rotation=45)
        
        ax.set_title('YOLO优化技术路线图', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('年份', fontsize=12)
        ax.set_ylabel('优化维度', fontsize=12)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(optimization_areas.keys())
        ax.set_xlim(2015.5, 2024.5)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # 保存图片
        output_path = self.output_dir / 'yolo_optimization_roadmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ 优化路线图已保存: {output_path}")
        
        return fig
        
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("📄 生成综合分析报告...")
        
        report_path = self.output_dir / 'yolo_evolution_comprehensive_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# YOLO算法演进综合分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 1. 执行摘要\n\n")
            f.write("本报告分析了YOLO系列算法从v1到v11的演进历程，")
            f.write("涵盖性能指标、架构创新、优化技术等多个维度。\n\n")
            
            f.write("### 主要发现\n\n")
            
            # 计算一些统计数据
            latest_version = max(self.yolo_data.keys(), key=lambda x: self.yolo_data[x]['year'])
            first_version = min(self.yolo_data.keys(), key=lambda x: self.yolo_data[x]['year'])
            
            map_improvement = (self.yolo_data[latest_version]['mAP_coco'] - 
                             self.yolo_data[first_version]['mAP_coco'])
            fps_improvement = (self.yolo_data[latest_version]['fps'] / 
                             self.yolo_data[first_version]['fps'])
            
            f.write(f"- **性能提升**: 从{first_version}到{latest_version}，")
            f.write(f"mAP提升了{map_improvement:.1f}个百分点\n")
            f.write(f"- **速度提升**: FPS提升了{fps_improvement:.1f}倍\n")
            f.write(f"- **效率优化**: 参数量从{self.yolo_data[first_version]['params_m']:.1f}M")
            f.write(f"降低到{self.yolo_data[latest_version]['params_m']:.1f}M\n\n")
            
            f.write("## 2. 版本详细分析\n\n")
            
            for version, data in self.yolo_data.items():
                f.write(f"### {version} ({data['year']})\n\n")
                f.write(f"**性能指标**:\n")
                f.write(f"- mAP@0.5:0.95: {data['mAP_coco']}%\n")
                f.write(f"- FPS: {data['fps']}\n")
                f.write(f"- 参数量: {data['params_m']}M\n")
                f.write(f"- FLOPs: {data['flops_g']}G\n\n")
                
                f.write(f"**关键特征**:\n")
                for feature in data['key_features']:
                    f.write(f"- {feature}\n")
                f.write("\n")
                
                f.write(f"**主要创新**:\n")
                for innovation in data['innovations']:
                    f.write(f"- {innovation}\n")
                f.write("\n")
            
            f.write("## 3. 优化建议\n\n")
            f.write("基于分析结果，我们建议:\n\n")
            f.write("1. **模型选择**: 根据应用场景选择合适的YOLO版本\n")
            f.write("   - 实时应用: YOLOv11n (高FPS)\n")
            f.write("   - 高精度需求: YOLOv4 (高mAP)\n")
            f.write("   - 平衡性能: YOLOv8/v11 (综合最优)\n\n")
            f.write("2. **部署优化**: 结合硬件特性进行模型优化\n")
            f.write("   - GPU部署: 使用TensorRT加速\n")
            f.write("   - 移动端: 模型量化和剪枝\n")
            f.write("   - 边缘设备: 选择轻量级版本\n\n")
            f.write("3. **训练策略**: 采用最新的训练技术\n")
            f.write("   - 数据增强: Mosaic, MixUp, CutMix\n")
            f.write("   - 标签分配: SimOTA, TaskAlignedAssigner\n")
            f.write("   - 损失函数: Focal Loss, IoU variants\n\n")
            
            f.write("## 4. 未来展望\n\n")
            f.write("YOLO算法的发展趋势:\n\n")
            f.write("- **架构创新**: Transformer集成、神经架构搜索\n")
            f.write("- **训练效率**: 自监督学习、知识蒸馏\n")
            f.write("- **部署优化**: 端到端优化、硬件协同设计\n")
            f.write("- **应用扩展**: 多模态融合、3D检测、视频理解\n\n")
            
        print(f"   ✅ 综合报告已保存: {report_path}")
        
    def run_all_visualizations(self):
        """运行所有可视化"""
        print("🎨 开始生成YOLO演进可视化")
        print("=" * 50)
        
        try:
            # 生成各种图表
            self.plot_performance_evolution()
            self.plot_architecture_comparison()
            self.generate_performance_table()
            self.create_optimization_roadmap()
            self.generate_comprehensive_report()
            
            print(f"\n✅ 所有可视化已完成！")
            print(f"📁 输出目录: {self.output_dir}")
            
            # 列出生成的文件
            print("\n📋 生成的文件:")
            for file_path in self.output_dir.glob('*'):
                print(f"   - {file_path.name}")
                
        except Exception as e:
            print(f"\n❌ 可视化生成失败: {e}")
            logger.error(f"Visualization error: {e}")

def main():
    """主函数"""
    print("YOLO算法演进可视化工具")
    print("生成YOLO系列算法的性能对比和发展趋势图表")
    print()
    
    # 创建可视化器
    visualizer = YOLOEvolutionVisualizer()
    
    # 运行所有可视化
    visualizer.run_all_visualizations()

if __name__ == "__main__":
    main()