#!/usr/bin/env python3
"""
NCCL测试日志时间数据提取工具
从addition.cc测试生成的日志文件中提取关键时间指标
"""

import re
import sys
import os
import json
from typing import Dict, List, Optional
from pathlib import Path


class NCCLTimingExtractor:
    """NCCL时间数据提取器"""
    
    def __init__(self):
        # 定义时间数据的正则表达式模式
        self.patterns = {
            'init_time': r'INIT TIME:\s+(\d+)\s+us\s+\(([0-9.]+)\s+ms\)',
            'first_comm_time': r'FIRST COMMUNICATION TIME:\s+(\d+)\s+us\s+\(([0-9.]+)\s+ms\)',
            'add_time': r'ADD TIME:\s+(\d+)\s+us\s+\(([0-9.]+)\s+ms\)',
            'post_expansion_time': r'POST-EXPANSION COMMUNICATION TIME:\s+(\d+)\s+us\s+\(([0-9.]+)\s+ms\)'
        }
        
        # 其他有用信息的模式
        self.info_patterns = {
            'ranks_info': r'Prepare to add 1 rank to (\d+) ranks \((\d+) in total\)',
            'allreduce_results': r'ncclAllReduce Success, sum result of rank (\d+): ([0-9.]+)',
            'nccl_version': r'NCCL version ([0-9.]+\+cuda[0-9.]+)',
            'cuda_version': r'cudaDriverVersion (\d+)'
        }
    
    def extract_from_file(self, log_file_path: str) -> Dict:
        """从日志文件中提取时间数据"""
        if not os.path.exists(log_file_path):
            raise FileNotFoundError(f"日志文件不存在: {log_file_path}")
        
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        return self.extract_from_content(content, log_file_path)
    
    def extract_from_content(self, content: str, source_file: str = "unknown") -> Dict:
        """从日志内容中提取时间数据"""
        result = {
            'source_file': source_file,
            'timing_data': {},
            'test_info': {},
            'allreduce_results': []
        }
        
        # 提取时间数据
        for key, pattern in self.patterns.items():
            match = re.search(pattern, content)
            if match:
                result['timing_data'][key] = {
                    'microseconds': int(match.group(1)),
                    'milliseconds': float(match.group(2))
                }
        
        # 提取测试信息
        for key, pattern in self.info_patterns.items():
            if key == 'ranks_info':
                match = re.search(pattern, content)
                if match:
                    result['test_info']['initial_ranks'] = int(match.group(1))
                    result['test_info']['total_ranks'] = int(match.group(2))
            
            elif key == 'allreduce_results':
                matches = re.findall(pattern, content)
                for match in matches:
                    result['allreduce_results'].append({
                        'rank': int(match[0]),
                        'result': float(match[1])
                    })
            
            elif key in ['nccl_version', 'cuda_version']:
                match = re.search(pattern, content)
                if match:
                    result['test_info'][key] = match.group(1)
        
        return result
    
    def extract_from_directory(self, log_dir: str) -> List[Dict]:
        """从目录中的所有.log文件提取数据"""
        log_dir_path = Path(log_dir)
        if not log_dir_path.exists():
            raise FileNotFoundError(f"日志目录不存在: {log_dir}")
        
        results = []
        log_files = list(log_dir_path.glob("*.log"))
        
        if not log_files:
            print(f"警告: 在目录 {log_dir} 中未找到.log文件")
            return results
        
        for log_file in log_files:
            try:
                result = self.extract_from_file(str(log_file))
                results.append(result)
                print(f"✓ 成功处理: {log_file.name}")
            except Exception as e:
                print(f"✗ 处理失败 {log_file.name}: {e}")
        
        return results
    
    def print_summary(self, data: Dict):
        """打印单个文件的时间数据摘要"""
        print(f"\n{'='*60}")
        print(f"文件: {os.path.basename(data['source_file'])}")
        print(f"{'='*60}")
        
        # 测试信息
        if data['test_info']:
            print("\n📊 测试配置:")
            for key, value in data['test_info'].items():
                print(f"  {key}: {value}")
        
        # 时间数据
        if data['timing_data']:
            print("\n⏱️  时间测量结果:")
            for key, timing in data['timing_data'].items():
                name = {
                    'init_time': 'NCCL初始化时间',
                    'first_comm_time': '首次通信时间',
                    'add_time': '动态添加时间',
                    'post_expansion_time': '扩展后通信时间'
                }.get(key, key)
                
                print(f"  {name:15}: {timing['microseconds']:8,} μs ({timing['milliseconds']:8.3f} ms)")
        
        # AllReduce结果
        if data['allreduce_results']:
            print("\n🔄 AllReduce结果:")
            for result in data['allreduce_results']:
                print(f"  Rank {result['rank']}: {result['result']}")
    
    def save_to_json(self, data: List[Dict], output_file: str):
        """保存数据到JSON文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 数据已保存到: {output_file}")
    
    def save_to_csv(self, data: List[Dict], output_file: str):
        """保存时间数据到CSV文件"""
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            headers = ['文件名', 'NCCL初始化(μs)', 'NCCL初始化(ms)', 
                      '首次通信(μs)', '首次通信(ms)', 
                      '动态添加(μs)', '动态添加(ms)', 
                      '扩展后通信(μs)', '扩展后通信(ms)',
                      '初始ranks', '总ranks', 'NCCL版本', 'CUDA版本']
            writer.writerow(headers)
            
            # 写入数据
            for item in data:
                row = [os.path.basename(item['source_file'])]
                
                # 时间数据
                timing_keys = ['init_time', 'first_comm_time', 'add_time', 'post_expansion_time']
                for key in timing_keys:
                    if key in item['timing_data']:
                        row.extend([
                            item['timing_data'][key]['microseconds'],
                            item['timing_data'][key]['milliseconds']
                        ])
                    else:
                        row.extend(['', ''])
                
                # 测试信息
                row.extend([
                    item['test_info'].get('initial_ranks', ''),
                    item['test_info'].get('total_ranks', ''),
                    item['test_info'].get('nccl_version', ''),
                    item['test_info'].get('cuda_version', '')
                ])
                
                writer.writerow(row)
        
        print(f"📊 CSV数据已保存到: {output_file}")


def main():
    """主函数"""
    extractor = NCCLTimingExtractor()
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python extract_timing_data.py <log_file>           # 处理单个文件")
        print("  python extract_timing_data.py <log_directory>      # 处理目录中所有.log文件")
        print("\n示例:")
        print("  python extract_timing_data.py /path/to/test.log")
        print("  python extract_timing_data.py /path/to/logs/")
        return
    
    input_path = sys.argv[1]
    
    try:
        if os.path.isfile(input_path):
            # 处理单个文件
            print(f"🔍 处理单个文件: {input_path}")
            result = extractor.extract_from_file(input_path)
            extractor.print_summary(result)
            
            # 保存结果
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            extractor.save_to_json([result], f"{base_name}_timing.json")
            extractor.save_to_csv([result], f"{base_name}_timing.csv")
            
        elif os.path.isdir(input_path):
            # 处理目录
            print(f"📁 处理目录: {input_path}")
            results = extractor.extract_from_directory(input_path)
            
            if results:
                # 打印所有结果的摘要
                for result in results:
                    extractor.print_summary(result)
                
                # 保存汇总结果
                dir_name = os.path.basename(input_path.rstrip('/'))
                extractor.save_to_json(results, f"{dir_name}_all_timing.json")
                extractor.save_to_csv(results, f"{dir_name}_all_timing.csv")
                
                print(f"\n📈 总计处理了 {len(results)} 个文件")
            else:
                print("❌ 未找到有效的日志文件")
        else:
            print(f"❌ 路径不存在: {input_path}")
    
    except Exception as e:
        print(f"❌ 错误: {e}")


if __name__ == "__main__":
    main()