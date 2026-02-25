#!/usr/bin/env python3
import json
import re
import subprocess
import sys
from html.parser import HTMLParser

class HTMLStepParser(HTMLParser):
    """解析HTML内容，提取步骤和命令"""
    def __init__(self):
        super().__init__()
        self.steps = []
        self.current_step = None
        self.in_code = False
        self.code_content = []
        self.in_pre = False
        self.in_h1 = False
        self.h1_content = []
    
    def handle_starttag(self, tag, attrs):
        if tag == 'h1':
            self.in_h1 = True
            self.h1_content = []
        elif tag == 'pre':
            self.in_pre = True
        elif tag == 'code' and self.in_pre:
            self.in_code = True
            self.code_content = []
    
    def handle_endtag(self, tag):
        if tag == 'h1':
            self.in_h1 = False
            if self.h1_content:
                step_title = ''.join(self.h1_content).strip()
                if step_title:
                    self.current_step = {
                        'title': step_title,
                        'commands': []
                    }
                    self.steps.append(self.current_step)
        elif tag == 'pre':
            self.in_pre = False
        elif tag == 'code' and self.in_pre:
            self.in_code = False
            if self.code_content and self.current_step:
                code = ''.join(self.code_content).strip()
                if code:
                    self.current_step['commands'].append(code)
    
    def handle_data(self, data):
        if self.in_h1:
            self.h1_content.append(data)
        elif self.in_code:
            self.code_content.append(data)

def parse_html_steps(html_content):
    """解析HTML内容，提取步骤和命令"""
    parser = HTMLStepParser()
    parser.feed(html_content)
    return parser.steps

def api_call(url):
    """调用hiascend API"""
    cmd = [
        'curl', '-s',
        '-H', 'referer: https://www.hiascend.com/cann/download',
        '-H', 'accept: */*',
        '-H', 'accept-language: zh-CN,zh;q=0.9',
        '-H', 'user-agent: Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Mobile Safari/537.36',
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return json.loads(result.stdout)
    return None

def get_product_series(version_id):
    """获取产品系列列表"""
    url = f"https://www.hiascend.com/ascendgateway/ascendservice/cannDownload/detail?lang=zh&versionId={version_id}&type=h04"
    response = api_call(url)
    if response and response.get('success'):
        return response.get('data', {}).get('subOutVOList', [])
    return []

def get_cpu_architectures(version_id, hardware_id):
    """获取CPU架构列表"""
    url = f"https://www.hiascend.com/ascendgateway/ascendservice/cannDownload/detail?lang=zh&versionId={version_id}&hardwareId={hardware_id}&type=h05"
    response = api_call(url)
    if response and response.get('success'):
        return response.get('data', {}).get('subOutVOList', [])
    return []

def get_operating_systems(version_id, hardware_id, cpu_arch_id):
    """获取操作系统列表"""
    url = f"https://www.hiascend.com/ascendgateway/ascendservice/cannDownload/detail?lang=zh&versionId={version_id}&hardwareId={hardware_id}&cpuArchitectureId={cpu_arch_id}&type=h06"
    response = api_call(url)
    if response and response.get('success'):
        return response.get('data', {}).get('subOutVOList', [])
    return []

def get_install_methods(version_id, hardware_id, cpu_arch_id, os_id):
    """获取安装方式列表"""
    url = f"https://www.hiascend.com/ascendgateway/ascendservice/cannDownload/detail?lang=zh&versionId={version_id}&hardwareId={hardware_id}&cpuArchitectureId={cpu_arch_id}&operateSystemId={os_id}&type=h07"
    response = api_call(url)
    if response and response.get('success'):
        return response.get('data', {}).get('subOutVOList', [])
    return []

def get_installation_steps(version_id, hardware_id, cpu_arch_id, os_id, install_method_id):
    """获取安装步骤"""
    url = f"https://www.hiascend.com/ascendgateway/ascendservice/cannDownload/detail?lang=zh&versionId={version_id}&hardwareId={hardware_id}&cpuArchitectureId={cpu_arch_id}&operateSystemId={os_id}&installMethodId={install_method_id}&type=h08"
    response = api_call(url)
    if response and response.get('success'):
        sub_out_list = response.get('data', {}).get('subOutVOList', [])
        if sub_out_list and len(sub_out_list) > 0:
            html_content = sub_out_list[0].get('name', '')
            return parse_html_steps(html_content)
    return []

def main():
    # 版本列表（手动记录）
    versions = [
        {'id': '657', 'name': '8.5.0'},
        {'id': '630', 'name': '8.5.0.alpha002'}
    ]
    
    config = {
        'version': '1.0.0',
        'updated_at': '2025-02-25',
        'versions': {}
    }
    
    for version_info in versions:
        version_id = version_info['id']
        version_name = version_info['name']
        print(f"Processing version {version_name} ({version_id})...")
        
        version_config = {
            'id': version_id,
            'name': version_name,
            'product_series': {}
        }
        
        product_series = get_product_series(version_id)
        print(f"  Found {len(product_series)} product series")
        
        for product in product_series:
            hardware_id = product['id']
            product_name = product['name']
            print(f"    Processing product: {product_name} ({hardware_id})")
            
            product_config = {
                'id': hardware_id,
                'name': product_name,
                'cpu_architectures': {}
            }
            
            cpu_archs = get_cpu_architectures(version_id, hardware_id)
            print(f"      Found {len(cpu_archs)} CPU architectures")
            
            for cpu_arch in cpu_archs:
                cpu_arch_id = cpu_arch['id']
                cpu_arch_name = cpu_arch['name']
                print(f"        Processing CPU: {cpu_arch_name} ({cpu_arch_id})")
                
                cpu_config = {
                    'id': cpu_arch_id,
                    'name': cpu_arch_name,
                    'operating_systems': {}
                }
                
                os_list = get_operating_systems(version_id, hardware_id, cpu_arch_id)
                print(f"          Found {len(os_list)} operating systems")
                
                for os_info in os_list:
                    os_id = os_info['id']
                    os_name = os_info['name']
                    print(f"            Processing OS: {os_name} ({os_id})")
                    
                    os_config = {
                        'id': os_id,
                        'name': os_name,
                        'install_methods': {}
                    }
                    
                    install_methods = get_install_methods(version_id, hardware_id, cpu_arch_id, os_id)
                    print(f"              Found {len(install_methods)} install methods")
                    
                    for install_method in install_methods:
                        install_method_id = install_method['id']
                        install_method_name = install_method['name']
                        print(f"                Processing install method: {install_method_name} ({install_method_id})")
                        
                        steps = get_installation_steps(version_id, hardware_id, cpu_arch_id, os_id, install_method_id)
                        
                        if steps:
                            install_config = {
                                'id': install_method_id,
                                'name': install_method_name,
                                'steps': steps
                            }
                            
                            os_config['install_methods'][install_method_id] = install_config
                    
                    cpu_config['operating_systems'][os_id] = os_config
                
                product_config['cpu_architectures'][cpu_arch_id] = cpu_config
            
            version_config['product_series'][hardware_id] = product_config
        
        config['versions'][version_id] = version_config
    
    # 保存配置文件
    output_file = '/Users/hua/code/docs/_static/ascend_config.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\nConfiguration saved to {output_file}")

if __name__ == '__main__':
    main()
