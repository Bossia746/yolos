#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置国内镜像 - 解决模型下载速度问题
"""

import os
import sys

def setup_insightface_mirror():
    """配置InsightFace使用国内镜像"""
    # 设置InsightFace模型下载镜像
    os.environ['INSIGHTFACE_MODEL_URL'] = 'https://mirror.ghproxy.com/https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'
    
    # 设置通用GitHub镜像
    os.environ['GITHUB_MIRROR'] = 'https://mirror.ghproxy.com/'
    
    print("已配置InsightFace国内镜像")
    print(f"INSIGHTFACE_MODEL_URL: {os.environ.get('INSIGHTFACE_MODEL_URL')}")
    print(f"GITHUB_MIRROR: {os.environ.get('GITHUB_MIRROR')}")

def setup_pip_mirror():
    """配置pip使用国内镜像"""
    pip_conf_content = """
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
"""
    
    # 创建pip配置目录
    if sys.platform.startswith('win'):
        pip_dir = os.path.expanduser('~/pip')
        pip_conf_path = os.path.join(pip_dir, 'pip.ini')
    else:
        pip_dir = os.path.expanduser('~/.pip')
        pip_conf_path = os.path.join(pip_dir, 'pip.conf')
    
    os.makedirs(pip_dir, exist_ok=True)
    
    with open(pip_conf_path, 'w', encoding='utf-8') as f:
        f.write(pip_conf_content)
    
    print(f"已配置pip国内镜像: {pip_conf_path}")

def setup_conda_mirror():
    """配置conda使用国内镜像"""
    conda_channels = [
        'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/',
        'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/',
        'https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/',
        'https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/'
    ]
    
    for channel in conda_channels:
        os.system(f'conda config --add channels {channel}')
    
    os.system('conda config --set show_channel_urls yes')
    print("已配置conda国内镜像")

if __name__ == '__main__':
    print("正在配置国内镜像...")
    setup_insightface_mirror()
    setup_pip_mirror()
    
    try:
        setup_conda_mirror()
    except:
        print("conda未安装，跳过conda镜像配置")
    
    print("\n镜像配置完成！")
    print("请重新运行程序以使用国内镜像下载模型。")