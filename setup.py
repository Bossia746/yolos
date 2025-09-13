from setuptools import setup, find_packages
import os
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "YOLOS - 多平台AIoT视觉大模型项目"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="yolos",
    version="1.0.0",
    author="YOLOS Team",
    author_email="team@yolos.ai",
    description="多平台AIoT视觉大模型项目 - Multi-platform AIoT Vision Large Model Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/yolos",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Graphics :: Capture :: Digital Camera",
        "Topic :: System :: Hardware",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "raspberry": [
            "RPi.GPIO>=0.7.1",
            "gpiozero>=1.6.2",
            "picamera>=1.13",
            "adafruit-circuitpython-motor>=3.4.0",
        ],
        "ros1": [
            "rospy",
            "sensor_msgs",
            "cv_bridge",
            "geometry_msgs",
        ],
        "ros2": [
            "rclpy",
            "sensor_msgs",
            "cv_bridge",
            "geometry_msgs",
            "std_msgs",
        ],
        "gpu": [
            "tensorrt>=8.0.0",
            "cupy>=11.0.0",
            "onnxruntime-gpu>=1.12.0",
        ],
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.10.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "web": [
            "fastapi>=0.95.0",
            "uvicorn>=0.20.0",
            "websockets>=10.4",
            "streamlit>=1.20.0",
        ],
        "medical": [
            "pydicom>=2.3.0",
            "nibabel>=4.0.0",
            "SimpleITK>=2.2.0",
        ],
        "all": [
            "RPi.GPIO>=0.7.1",
            "gpiozero>=1.6.2",
            "tensorrt>=8.0.0",
            "fastapi>=0.95.0",
            "streamlit>=1.20.0",
            "pydicom>=2.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "yolos-detect=recognition.cli:main",
            "yolos-train=training.cli:main",
            "yolos-deploy=deployment.cli:main",
            "yolos-web=web.app:main",
            "yolos-test=tests.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "yolos": [
            "config/*.yaml",
            "config/*.json",
            "models/*.pt",
            "models/*.onnx",
            "web/templates/*.html",
            "web/static/*",
        ],
    },
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/your-username/yolos/issues",
        "Source": "https://github.com/your-username/yolos",
        "Documentation": "https://yolos.readthedocs.io/",
    },
)