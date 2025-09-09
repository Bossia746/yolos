#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest configuration file for YOLOS project.
Configures test environment, fixtures, and module imports.
"""

import sys
import os
from pathlib import Path
import pytest
import logging
from unittest.mock import Mock, MagicMock

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@pytest.fixture(scope="session")
def project_root_path():
    """Provide project root path."""
    return project_root

@pytest.fixture(scope="session")
def test_data_dir():
    """Provide test data directory."""
    return project_root / "tests" / "data"

@pytest.fixture(scope="session")
def temp_dir(tmp_path_factory):
    """Provide temporary directory for tests."""
    return tmp_path_factory.mktemp("yolos_test")

@pytest.fixture
def mock_camera():
    """Mock camera for testing."""
    camera = Mock()
    camera.read.return_value = (True, Mock())
    camera.isOpened.return_value = True
    camera.release = Mock()
    return camera

@pytest.fixture
def mock_cv2(monkeypatch):
    """Mock OpenCV for testing."""
    mock_cv2 = MagicMock()
    mock_cv2.VideoCapture.return_value = Mock()
    mock_cv2.imread.return_value = Mock()
    mock_cv2.imwrite.return_value = True
    monkeypatch.setattr("cv2", mock_cv2)
    return mock_cv2

@pytest.fixture
def mock_torch(monkeypatch):
    """Mock PyTorch for testing."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.load.return_value = Mock()
    monkeypatch.setattr("torch", mock_torch)
    return mock_torch

@pytest.fixture
def sample_config():
    """Provide sample configuration for testing."""
    return {
        "camera": {
            "device_id": 0,
            "width": 640,
            "height": 480,
            "fps": 30
        },
        "detection": {
            "model_path": "models/yolov8n.pt",
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45
        },
        "logging": {
            "level": "INFO",
            "file": "logs/test.log"
        }
    }

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, tmp_path):
    """Setup test environment for each test."""
    # Set test environment variables
    monkeypatch.setenv("YOLOS_TEST_MODE", "true")
    monkeypatch.setenv("YOLOS_LOG_LEVEL", "DEBUG")
    
    # Create temporary directories
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)
    
    # Mock file paths
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "gui: mark test as GUI test"
    )
    config.addinivalue_line(
        "markers", "hardware: mark test as requiring hardware"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file names
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        elif "gui" in item.nodeid:
            item.add_marker(pytest.mark.gui)
        else:
            item.add_marker(pytest.mark.unit)

# Suppress specific warnings
pytest.register_assert_rewrite("tests.base_test")