from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="yolos",
    version="1.0.0",
    author="YOLOS Team",
    author_email="team@yolos.ai",
    description="多平台AIoT视觉大模型项目",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/yolos",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "raspberry": ["RPi.GPIO>=0.7.1", "gpiozero>=1.6.2", "picamera>=1.13"],
        "ros1": ["rospy", "sensor_msgs", "cv_bridge"],
        "ros2": ["rclpy", "sensor_msgs", "cv_bridge"],
        "gpu": ["tensorrt>=8.0.0", "cupy>=11.0.0"],
        "dev": ["pytest>=7.2.0", "black>=22.10.0", "flake8>=5.0.0", "mypy>=0.991"],
    },
    entry_points={
        "console_scripts": [
            "yolos-detect=detection.cli:main",
            "yolos-train=training.cli:main",
            "yolos-deploy=deployment.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)