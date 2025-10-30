#!/usr/bin/env python3
"""
Enhanced NCCL Performance Visualization Script with English Labels
Generates comprehensive performance analysis charts and reports
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NCCLPerformanceAnalyzer:
    def __init__(self, json_file: str):
        """Initialize analyzer with JSON data file"""
        self.json_file = json_file
        self.data = self.load_data()
        self.df = self.create_dataframe()
        
        # Color scheme
        self.colors = {
            'init_time': '#FF6B6B',
            'first_comm_time': '#4ECDC4', 
            'add_time': '#45B7D1',
            'post_expansion_time': '#96CEB4'
        }
        
        # English labels
        self.labels = {
            'init_time': 'NCCL Init',
            'first_comm_time': 'First Comm',
            'add_time': 'Dynamic Add',
            'post_expansion_time': 'Post Expansion'
        }
    
    def load_data(self):
        """Load JSON data"""
        with open(self.json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_dataframe(self):
        """Create pandas DataFrame from JSON data"""
        rows = []
        for item in self.data:
            # Extract GPU count from filename
            filename = item['source_file']
            gpu_count = int(filename.split('/')[-1].split('gpus')[0])
            
            row = {
                'filename': filename,
                'gpu_count': gpu_count,
                'initial_ranks': item['test_info']['initial_ranks'],
                'total_ranks': item['test_info']['total_ranks'],
                'nccl_version': item['test_info']['nccl_version'],
                'cuda_version': item['test_info']['cuda_version']
            }
            
            # Add timing data (milliseconds)
            for time_type, time_data in item['timing_data'].items():
                row[f'{time_type}_ms'] = time_data['milliseconds']
                row[f'{time_type}_us'] = time_data['microseconds']
            
            rows.append(row)
        
        return pd.DataFrame(rows).sort_values('gpu_count')
    
    def plot_comprehensive_analysis(self, save_path: str = None):
        """Generate comprehensive analysis chart"""
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        gpu_counts = self.df['gpu_count'].values
        timing_types = ['init_time', 'first_comm_time', 'add_time', 'post_expansion_time']
        
        # 1. Timing comparison bar chart (top 2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        x = np.arange(len(gpu_counts))
        width = 0.2
        
        for i, time_type in enumerate(timing_types):
            values = self.df[f'{time_type}_ms'].values
            bars = ax1.bar(x + i * width, values, width, 
                          label=self.labels[time_type], 
                          color=self.colors[time_type],
                          alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}',
                            ha='center', va='bottom', fontsize=8)
        
        ax1.set_xlabel('GPU Count', fontsize=12)
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title('NCCL Performance Timing Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(gpu_counts)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Dynamic add time scaling (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(gpu_counts, self.df['add_time_ms'], 'o-', 
                linewidth=3, markersize=8, color=self.colors['add_time'])
        ax2.set_xlabel('GPU Count')
        ax2.set_ylabel('Dynamic Add Time (ms)')
        ax2.set_title('Dynamic Add Scaling')
        ax2.grid(True, alpha=0.3)
        
        # 3. Communication improvement ratio (middle right)
        ax3 = fig.add_subplot(gs[1, 2])
        comm_improvement = self.df['first_comm_time_ms'] / self.df['post_expansion_time_ms']
        bars = ax3.bar(gpu_counts, comm_improvement, color='#FFA07A', alpha=0.8)
        ax3.set_xlabel('GPU Count')
        ax3.set_ylabel('Improvement Ratio')
        ax3.set_title('Communication Improvement')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}x',
                    ha='center', va='bottom', fontsize=9)
        
        # 4. Stacked timing chart (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        bottom = np.zeros(len(gpu_counts))
        
        for time_type in timing_types:
            values = self.df[f'{time_type}_ms'].values
            ax4.bar(gpu_counts, values, bottom=bottom, 
                   label=self.labels[time_type],
                   color=self.colors[time_type], alpha=0.8)
            bottom += values
        
        ax4.set_xlabel('GPU Count')
        ax4.set_ylabel('Cumulative Time (ms)')
        ax4.set_title('Time Composition Stack')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Add time percentage (bottom middle)
        ax5 = fig.add_subplot(gs[2, 1])
        total_time = sum(self.df[f'{t}_ms'] for t in timing_types)
        add_percentage = (self.df['add_time_ms'] / total_time) * 100
        
        bars = ax5.bar(gpu_counts, add_percentage, color=self.colors['add_time'], alpha=0.8)
        ax5.set_xlabel('GPU Count')
        ax5.set_ylabel('Add Time Percentage (%)')
        ax5.set_title('Dynamic Add Time Ratio')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)
        
        # 6. Performance efficiency (bottom right)
        ax6 = fig.add_subplot(gs[2, 2])
        efficiency = 1000 / self.df['add_time_ms']  # Operations per second
        ax6.plot(gpu_counts, efficiency, 'o-', 
                linewidth=3, markersize=8, color='#9370DB')
        ax6.set_xlabel('GPU Count')
        ax6.set_ylabel('Add Efficiency (ops/s)')
        ax6.set_title('Dynamic Add Efficiency')
        ax6.grid(True, alpha=0.3)
        
        # 7. System info table (bottom span)
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        # Create info table
        info_data = [
            ['GPU Range', f"{gpu_counts.min()} - {gpu_counts.max()}"],
            ['NCCL Version', self.df['nccl_version'].iloc[0]],
            ['CUDA Version', str(self.df['cuda_version'].iloc[0])],
            ['Avg Add Time', f"{self.df['add_time_ms'].mean():.1f} ms"],
            ['Add Time Std', f"{self.df['add_time_ms'].std():.1f} ms"],
            ['Best Improvement', f"{comm_improvement.max():.1f}x"]
        ]
        
        table = ax7.table(cellText=info_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.3, 0.2, 0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.suptitle('NCCL Dynamic Scaling Comprehensive Performance Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comprehensive analysis saved: {save_path}")
        
        return fig
    
    def plot_performance_dashboard(self, save_path: str = None):
        """Generate performance dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('NCCL Performance Dashboard', fontsize=16, fontweight='bold')
        
        gpu_counts = self.df['gpu_count'].values
        
        # 1. Main timing metrics
        ax1 = axes[0, 0]
        timing_types = ['init_time', 'first_comm_time', 'add_time', 'post_expansion_time']
        
        for time_type in timing_types:
            values = self.df[f'{time_type}_ms'].values
            ax1.plot(gpu_counts, values, 'o-', linewidth=2, markersize=6,
                    label=self.labels[time_type], color=self.colors[time_type])
        
        ax1.set_xlabel('GPU Count')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Timing Metrics Overview')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 2. Dynamic add performance
        ax2 = axes[0, 1]
        add_times = self.df['add_time_ms'].values
        bars = ax2.bar(gpu_counts, add_times, color=self.colors['add_time'], alpha=0.8)
        
        # Add trend line
        z = np.polyfit(gpu_counts, add_times, 1)
        p = np.poly1d(z)
        ax2.plot(gpu_counts, p(gpu_counts), "r--", alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('GPU Count')
        ax2.set_ylabel('Dynamic Add Time (ms)')
        ax2.set_title('Dynamic Add Performance')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
        
        # 3. Communication improvement
        ax3 = axes[1, 0]
        improvement = self.df['first_comm_time_ms'] / self.df['post_expansion_time_ms']
        bars = ax3.bar(gpu_counts, improvement, color='#FFA07A', alpha=0.8)
        
        ax3.set_xlabel('GPU Count')
        ax3.set_ylabel('Improvement Ratio')
        ax3.set_title('Communication Performance Improvement')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}x',
                    ha='center', va='bottom', fontsize=9)
        
        # 4. Performance summary heatmap
        ax4 = axes[1, 1]
        
        # Create normalized performance matrix
        metrics = ['Init Time', 'First Comm', 'Dynamic Add', 'Post Expansion']
        data_matrix = []
        
        for time_type in ['init_time', 'first_comm_time', 'add_time', 'post_expansion_time']:
            values = self.df[f'{time_type}_ms'].values
            normalized = (values - values.min()) / (values.max() - values.min())
            data_matrix.append(normalized)
        
        data_matrix = np.array(data_matrix)
        
        im = ax4.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto')
        ax4.set_xticks(range(len(gpu_counts)))
        ax4.set_xticklabels(gpu_counts)
        ax4.set_yticks(range(len(metrics)))
        ax4.set_yticklabels(metrics)
        ax4.set_xlabel('GPU Count')
        ax4.set_title('Performance Heatmap (Normalized)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Normalized Performance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Performance dashboard saved: {save_path}")
        
        return fig
    
    def generate_detailed_report(self, save_path: str = None):
        """Generate detailed text report"""
        report = []
        report.append("NCCL Dynamic Scaling Enhanced Performance Analysis")
        report.append("=" * 60)
        report.append("")
        
        # Test configuration
        report.append("📊 Test Configuration:")
        report.append(f"  - GPU Range: {self.df['gpu_count'].min()} - {self.df['gpu_count'].max()}")
        report.append(f"  - NCCL Version: {self.df['nccl_version'].iloc[0]}")
        report.append(f"  - CUDA Version: {self.df['cuda_version'].iloc[0]}")
        report.append("")
        
        # Performance metrics by GPU count
        report.append("⏱️ Performance Metrics by GPU Count:")
        report.append("")
        
        for _, row in self.df.iterrows():
            gpu_count = row['gpu_count']
            total_time = (row['init_time_ms'] + row['first_comm_time_ms'] + 
                         row['add_time_ms'] + row['post_expansion_time_ms'])
            add_ratio = (row['add_time_ms'] / total_time) * 100
            
            report.append(f"{gpu_count} GPUs:")
            report.append(f"  - NCCL Init: {row['init_time_ms']:.1f} ms")
            report.append(f"  - First Comm: {row['first_comm_time_ms']:.1f} ms")
            report.append(f"  - Dynamic Add: {row['add_time_ms']:.1f} ms")
            report.append(f"  - Post Expansion: {row['post_expansion_time_ms']:.1f} ms")
            report.append(f"  - Total Time: {total_time:.1f} ms")
            report.append(f"  - Add Time Ratio: {add_ratio:.1f}%")
            report.append("")
        
        # Statistical analysis
        add_times = self.df['add_time_ms']
        comm_improvement = self.df['first_comm_time_ms'] / self.df['post_expansion_time_ms']
        
        report.append("🔍 Statistical Analysis:")
        report.append(f"  - Average Dynamic Add Time: {add_times.mean():.1f} ms")
        report.append(f"  - Dynamic Add Time Std Dev: {add_times.std():.1f} ms")
        report.append(f"  - Min Dynamic Add Time: {add_times.min():.1f} ms")
        report.append(f"  - Max Dynamic Add Time: {add_times.max():.1f} ms")
        report.append(f"  - Average Communication Improvement: {comm_improvement.mean():.1f}x")
        report.append(f"  - Best Communication Improvement: {comm_improvement.max():.1f}x")
        report.append("")
        
        # Scaling analysis
        report.append("📈 Scaling Analysis:")
        base_add_time = add_times.iloc[0]
        for i, add_time in enumerate(add_times):
            gpu_count = self.df['gpu_count'].iloc[i]
            scaling_factor = add_time / base_add_time
            report.append(f"  - {gpu_count} GPUs: {scaling_factor:.2f}x base add time")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✓ Detailed analysis saved: {save_path}")
        
        return report_text
    
    def export_metrics_csv(self, save_path: str = None):
        """Export performance metrics to CSV"""
        # Calculate additional metrics
        export_df = self.df.copy()
        
        # Add derived metrics
        total_time = (export_df['init_time_ms'] + export_df['first_comm_time_ms'] + 
                     export_df['add_time_ms'] + export_df['post_expansion_time_ms'])
        export_df['Total_Time_ms'] = total_time
        export_df['Add_Time_Ratio_Percent'] = (export_df['add_time_ms'] / total_time) * 100
        export_df['Comm_Improvement_Ratio'] = (export_df['first_comm_time_ms'] / 
                                              export_df['post_expansion_time_ms'])
        
        # Select and rename columns for export
        columns_to_export = [
            'gpu_count', 'init_time_ms', 'first_comm_time_ms', 'add_time_ms', 
            'post_expansion_time_ms', 'Total_Time_ms', 'Add_Time_Ratio_Percent',
            'Comm_Improvement_Ratio', 'nccl_version', 'cuda_version'
        ]
        
        export_df = export_df[columns_to_export]
        export_df.columns = [
            'GPU_Count', 'Init_Time_ms', 'First_Comm_ms', 'Dynamic_Add_ms',
            'Post_Expansion_ms', 'Total_Time_ms', 'Add_Time_Ratio_Percent',
            'Comm_Improvement_Ratio', 'NCCL_Version', 'CUDA_Version'
        ]
        
        if save_path:
            export_df.to_csv(save_path, index=False)
            print(f"✓ CSV metrics saved: {save_path}")
        
        return export_df

def main():
    parser = argparse.ArgumentParser(description='Enhanced NCCL Performance Analysis')
    parser.add_argument('json_file', help='Path to JSON timing data file')
    parser.add_argument('--output', '-o', default='./enhanced_output', 
                       help='Output directory for generated files')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print("🎨 Generating Enhanced NCCL Performance Analysis Report...")
    
    # Initialize analyzer
    analyzer = NCCLPerformanceAnalyzer(args.json_file)
    
    # Generate comprehensive analysis
    analyzer.plot_comprehensive_analysis(
        save_path=output_dir / "comprehensive_analysis.png"
    )
    
    # Generate performance dashboard
    analyzer.plot_performance_dashboard(
        save_path=output_dir / "performance_dashboard.png"
    )
    
    # Generate detailed report
    analyzer.generate_detailed_report(
        save_path=output_dir / "detailed_analysis.txt"
    )
    
    # Export CSV metrics
    analyzer.export_metrics_csv(
        save_path=output_dir / "performance_metrics.csv"
    )
    
    print(f"\n📊 Enhanced report generated in: {args.output}")
    print("Generated files:")
    print("  - comprehensive_analysis.png  (Comprehensive Analysis)")
    print("  - performance_dashboard.png   (Performance Dashboard)")
    print("  - detailed_analysis.txt       (Detailed Analysis)")
    print("  - performance_metrics.csv     (Performance Metrics)")

if __name__ == "__main__":
    main()