#!/usr/bin/env python3
"""
NCCL性能数据可视化工具
从JSON数据文件生成各种性能分析图表
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import argparse
import sys
from typing import Dict, List, Tuple

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class NCCLPerformancePlotter:
    """NCCL性能数据可视化类"""
    
    def __init__(self, json_file: str):
        """初始化，加载JSON数据"""
        self.json_file = json_file
        self.data = self._load_data()
        self.df = self._create_dataframe()
        
        # 定义颜色方案
        self.colors = {
            'init_time': '#FF6B6B',      # 红色 - 初始化时间
            'first_comm_time': '#4ECDC4', # 青色 - 首次通信时间
            'add_time': '#45B7D1',       # 蓝色 - 动态添加时间
            'post_expansion_time': '#96CEB4'  # 绿色 - 扩展后通信时间
        }
        
        # English label mapping
        self.labels = {
            'init_time': 'NCCL Init',
            'first_comm_time': 'First Comm',
            'add_time': 'Dynamic Add',
            'post_expansion_time': 'Post Expansion'
        }
    
    def _load_data(self) -> List[Dict]:
        """加载JSON数据"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"错误: 找不到文件 {self.json_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"错误: JSON解析失败 - {e}")
            sys.exit(1)
    
    def _create_dataframe(self) -> pd.DataFrame:
        """将JSON数据转换为DataFrame"""
        rows = []
        for item in self.data:
            # 提取文件名中的GPU数量
            filename = Path(item['source_file']).stem
            gpu_count = int(filename.split('gpus')[0])
            
            row = {
                'filename': filename,
                'gpu_count': gpu_count,
                'initial_ranks': item['test_info']['initial_ranks'],
                'total_ranks': item['test_info']['total_ranks'],
                'nccl_version': item['test_info']['nccl_version'],
                'cuda_version': item['test_info']['cuda_version']
            }
            
            # 添加时间数据 (毫秒)
            for time_type, time_data in item['timing_data'].items():
                row[f'{time_type}_ms'] = time_data['milliseconds']
                row[f'{time_type}_us'] = time_data['microseconds']
            
            rows.append(row)
        
        return pd.DataFrame(rows).sort_values('gpu_count')
    
    def plot_timing_comparison(self, save_path: str = None, show_values: bool = True):
        """绘制时间对比柱状图"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 准备数据
        gpu_counts = self.df['gpu_count'].values
        x = np.arange(len(gpu_counts))
        width = 0.2
        
        timing_types = ['init_time', 'first_comm_time', 'add_time', 'post_expansion_time']
        
        # 绘制柱状图
        for i, time_type in enumerate(timing_types):
            values = self.df[f'{time_type}_ms'].values
            bars = ax.bar(x + i * width, values, width, 
                         label=self.labels[time_type], 
                         color=self.colors[time_type],
                         alpha=0.8)
            
            # 添加数值标签
            if show_values:
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.1f}ms', ha='center', va='bottom', 
                           fontsize=8, rotation=0)
        
        # 设置图表属性
        ax.set_xlabel('GPU数量', fontsize=12, fontweight='bold')
        ax.set_ylabel('时间 (毫秒)', fontsize=12, fontweight='bold')
        ax.set_title('NCCL动态扩展性能对比', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([f'{count} GPUs' for count in gpu_counts])
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 时间对比图已保存: {save_path}")
        
        return fig
    
    def plot_scaling_analysis(self, save_path: str = None):
        """绘制扩展性分析图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        gpu_counts = self.df['gpu_count'].values
        
        # 1. 总体时间趋势
        timing_types = ['init_time', 'first_comm_time', 'add_time', 'post_expansion_time']
        for time_type in timing_types:
            values = self.df[f'{time_type}_ms'].values
            ax1.plot(gpu_counts, values, marker='o', linewidth=2, 
                    label=self.labels[time_type], color=self.colors[time_type])
        
        ax1.set_xlabel('GPU Count')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('NCCL Performance Timing Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Dynamic scaling efficiency analysis
        add_times = self.df['add_time_ms'].values
        ax2.bar(gpu_counts, add_times, color=self.colors['add_time'], alpha=0.7)
        ax2.set_xlabel('GPU Count')
        ax2.set_ylabel('Dynamic Add Time (ms)')
        ax2.set_title('Dynamic Scaling Performance Analysis')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(add_times):
            ax2.text(gpu_counts[i], v + v*0.02, f'{v:.1f}ms', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 3. Communication time comparison
        first_comm = self.df['first_comm_time_ms'].values
        post_comm = self.df['post_expansion_time_ms'].values
        
        x = np.arange(len(gpu_counts))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, first_comm, width, label='First Comm', 
                       color=self.colors['first_comm_time'], alpha=0.8)
        bars2 = ax3.bar(x + width/2, post_comm, width, label='Post Expansion', 
                       color=self.colors['post_expansion_time'], alpha=0.8)
        
        ax3.set_xlabel('GPU Count')
        ax3.set_ylabel('Communication Time (ms)')
        ax3.set_title('Communication Performance Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'{count}' for count in gpu_counts])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 时间占比饼图 (以8GPU为例)
        if len(self.df) > 0:
            # 选择最大GPU数量的数据
            max_gpu_idx = self.df['gpu_count'].idxmax()
            max_gpu_data = self.df.loc[max_gpu_idx]
            
            times = [max_gpu_data[f'{t}_ms'] for t in timing_types]
            labels = [self.labels[t] for t in timing_types]
            colors = [self.colors[t] for t in timing_types]
            
            wedges, texts, autotexts = ax4.pie(times, labels=labels, colors=colors, 
                                              autopct='%1.1f%%', startangle=90)
            ax4.set_title(f'{max_gpu_data["gpu_count"]}GPU时间分布')
            
            # 美化饼图文本
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 扩展性分析图已保存: {save_path}")
        
        return fig
    
    def plot_stacked_timing(self, save_path: str = None):
        """绘制堆叠时间图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        gpu_counts = self.df['gpu_count'].values
        timing_types = ['init_time', 'first_comm_time', 'add_time', 'post_expansion_time']
        
        # 准备堆叠数据
        bottom = np.zeros(len(gpu_counts))
        
        for time_type in timing_types:
            values = self.df[f'{time_type}_ms'].values
            bars = ax.bar(gpu_counts, values, bottom=bottom, 
                         label=self.labels[time_type], 
                         color=self.colors[time_type], alpha=0.8)
            
            # 添加数值标签
            for i, (bar, value) in enumerate(zip(bars, values)):
                if value > 50:  # 只显示较大的值
                    ax.text(bar.get_x() + bar.get_width()/2., 
                           bottom[i] + value/2, f'{value:.0f}ms',
                           ha='center', va='center', fontweight='bold', 
                           color='white' if value > 200 else 'black')
            
            bottom += values
        
        ax.set_xlabel('GPU数量', fontsize=12, fontweight='bold')
        ax.set_ylabel('累积时间 (毫秒)', fontsize=12, fontweight='bold')
        ax.set_title('NCCL操作时间堆叠分析', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 设置x轴标签
        ax.set_xticks(gpu_counts)
        ax.set_xticklabels([f'{count} GPUs' for count in gpu_counts])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 堆叠时间图已保存: {save_path}")
        
        return fig
    
    def plot_efficiency_metrics(self, save_path: str = None):
        """绘制效率指标图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        gpu_counts = self.df['gpu_count'].values
        
        # 1. 动态添加效率 (时间/新增GPU)
        add_times = self.df['add_time_ms'].values
        efficiency = add_times  # 每个新增GPU的添加时间
        
        ax1.plot(gpu_counts, efficiency, marker='o', linewidth=3, 
                color=self.colors['add_time'], markersize=8)
        ax1.set_xlabel('总GPU数量')
        ax1.set_ylabel('动态添加时间 (毫秒)')
        ax1.set_title('动态添加效率趋势')
        ax1.grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(gpu_counts, efficiency, 1)
        p = np.poly1d(z)
        ax1.plot(gpu_counts, p(gpu_counts), "--", alpha=0.7, color='red')
        
        # 2. 通信性能提升比
        first_comm = self.df['first_comm_time_ms'].values
        post_comm = self.df['post_expansion_time_ms'].values
        improvement_ratio = first_comm / post_comm
        
        bars = ax2.bar(gpu_counts, improvement_ratio, color='orange', alpha=0.7)
        ax2.set_xlabel('GPU数量')
        ax2.set_ylabel('性能提升比')
        ax2.set_title('扩展后通信性能提升')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, ratio in zip(bars, improvement_ratio):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                    f'{ratio:.1f}x', ha='center', va='bottom', fontweight='bold')
        
        # 3. 总时间分析
        total_times = (self.df['init_time_ms'] + self.df['first_comm_time_ms'] + 
                      self.df['add_time_ms'] + self.df['post_expansion_time_ms'])
        
        ax3.plot(gpu_counts, total_times, marker='s', linewidth=3, 
                color='purple', markersize=8, label='总时间')
        ax3.set_xlabel('GPU数量')
        ax3.set_ylabel('总时间 (毫秒)')
        ax3.set_title('总体性能趋势')
        ax3.grid(True, alpha=0.3)
        
        # 4. 时间分布雷达图 (选择中间的GPU数量)
        if len(gpu_counts) >= 3:
            mid_idx = len(gpu_counts) // 2
            mid_data = self.df.iloc[mid_idx]
            
            categories = list(self.labels.values())
            values = [mid_data[f'{t}_ms'] for t in ['init_time', 'first_comm_time', 
                                                   'add_time', 'post_expansion_time']]
            
            # 标准化数值 (0-1)
            max_val = max(values)
            normalized_values = [v/max_val for v in values]
            
            # 雷达图
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            normalized_values += normalized_values[:1]  # 闭合图形
            angles += angles[:1]
            
            ax4.plot(angles, normalized_values, 'o-', linewidth=2, color='red')
            ax4.fill(angles, normalized_values, alpha=0.25, color='red')
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(categories)
            ax4.set_title(f'{mid_data["gpu_count"]}GPU时间分布雷达图')
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 效率指标图已保存: {save_path}")
        
        return fig
    
    def generate_summary_report(self, output_dir: str = "."):
        """生成完整的性能分析报告"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("🎨 正在生成NCCL性能分析报告...")
        
        # 生成各种图表
        self.plot_timing_comparison(output_path / "1_timing_comparison.png")
        self.plot_scaling_analysis(output_path / "2_scaling_analysis.png")
        self.plot_stacked_timing(output_path / "3_stacked_timing.png")
        self.plot_efficiency_metrics(output_path / "4_efficiency_metrics.png")
        
        # 生成数据摘要
        self._generate_text_summary(output_path / "performance_summary.txt")
        
        print(f"\n📊 完整报告已生成到目录: {output_path}")
        print("生成的文件:")
        print("  - 1_timing_comparison.png  (时间对比图)")
        print("  - 2_scaling_analysis.png   (扩展性分析)")
        print("  - 3_stacked_timing.png     (堆叠时间图)")
        print("  - 4_efficiency_metrics.png (效率指标)")
        print("  - performance_summary.txt  (性能摘要)")
    
    def _generate_text_summary(self, output_file: str):
        """生成文本摘要报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("NCCL动态扩展性能分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("📊 测试配置:\n")
            f.write(f"  - GPU数量范围: {self.df['gpu_count'].min()} - {self.df['gpu_count'].max()}\n")
            f.write(f"  - NCCL版本: {self.df['nccl_version'].iloc[0]}\n")
            f.write(f"  - CUDA版本: {self.df['cuda_version'].iloc[0]}\n\n")
            
            f.write("⏱️ 关键性能指标:\n")
            for _, row in self.df.iterrows():
                f.write(f"\n{row['gpu_count']} GPUs:\n")
                f.write(f"  - NCCL初始化: {row['init_time_ms']:.1f} ms\n")
                f.write(f"  - 首次通信: {row['first_comm_time_ms']:.1f} ms\n")
                f.write(f"  - 动态添加: {row['add_time_ms']:.1f} ms\n")
                f.write(f"  - 扩展后通信: {row['post_expansion_time_ms']:.1f} ms\n")
            
            f.write("\n🔍 性能分析:\n")
            add_times = self.df['add_time_ms'].values
            f.write(f"  - 平均动态添加时间: {np.mean(add_times):.1f} ms\n")
            f.write(f"  - 动态添加时间标准差: {np.std(add_times):.1f} ms\n")
            
            first_comm = self.df['first_comm_time_ms'].values
            post_comm = self.df['post_expansion_time_ms'].values
            avg_improvement = np.mean(first_comm / post_comm)
            f.write(f"  - 平均通信性能提升: {avg_improvement:.1f}x\n")
        
        print(f"✓ 性能摘要已保存: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='NCCL性能数据可视化工具')
    parser.add_argument('json_file', help='JSON数据文件路径')
    parser.add_argument('--output', '-o', default='./plots', 
                       help='输出目录 (默认: ./plots)')
    parser.add_argument('--chart', '-c', 
                       choices=['comparison', 'scaling', 'stacked', 'efficiency', 'all'],
                       default='all', help='生成的图表类型')
    parser.add_argument('--show', '-s', action='store_true', 
                       help='显示图表而不保存')
    
    args = parser.parse_args()
    
    # 创建绘图器
    plotter = NCCLPerformancePlotter(args.json_file)
    
    if args.show:
        # 显示图表
        if args.chart == 'comparison' or args.chart == 'all':
            plotter.plot_timing_comparison()
        if args.chart == 'scaling' or args.chart == 'all':
            plotter.plot_scaling_analysis()
        if args.chart == 'stacked' or args.chart == 'all':
            plotter.plot_stacked_timing()
        if args.chart == 'efficiency' or args.chart == 'all':
            plotter.plot_efficiency_metrics()
        
        plt.show()
    else:
        # 保存图表
        if args.chart == 'all':
            plotter.generate_summary_report(args.output)
        else:
            output_path = Path(args.output)
            output_path.mkdir(exist_ok=True)
            
            if args.chart == 'comparison':
                plotter.plot_timing_comparison(output_path / "timing_comparison.png")
            elif args.chart == 'scaling':
                plotter.plot_scaling_analysis(output_path / "scaling_analysis.png")
            elif args.chart == 'stacked':
                plotter.plot_stacked_timing(output_path / "stacked_timing.png")
            elif args.chart == 'efficiency':
                plotter.plot_efficiency_metrics(output_path / "efficiency_metrics.png")


if __name__ == "__main__":
    main()