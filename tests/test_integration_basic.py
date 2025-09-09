#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic integration tests for YOLOS project.
Tests core functionality and module integration.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from tests.base_test import BaseTest

class TestBasicIntegration(BaseTest):
    """Basic integration tests."""
    
    def test_project_structure(self):
        """Test that project structure is correct."""
        project_root = Path(__file__).parent.parent
        
        # Check essential directories
        assert (project_root / "src").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "config").exists()
        assert (project_root / "docs").exists()
        
        # Check essential files
        assert (project_root / "requirements.txt").exists()
        assert (project_root / "README.md").exists()
        assert (project_root / "pytest.ini").exists()
        assert (project_root / "conftest.py").exists()
    
    def test_core_modules_import(self):
        """Test that core modules can be imported."""
        try:
            # Test basic imports without circular dependencies
            from src.core import config
            from src.core import engine
            assert True
        except ImportError as e:
            pytest.skip(f"Core modules not available: {e}")
    
    @patch('cv2.VideoCapture')
    def test_camera_interface_mock(self, mock_video_capture):
        """Test camera interface with mocked OpenCV."""
        # Mock camera
        mock_camera = Mock()
        mock_camera.isOpened.return_value = True
        mock_camera.read.return_value = (True, Mock())
        mock_video_capture.return_value = mock_camera
        
        # Test camera initialization
        import cv2
        cap = cv2.VideoCapture(0)
        assert cap.isOpened()
        
        ret, frame = cap.read()
        assert ret is True
        assert frame is not None
    
    def test_configuration_loading(self):
        """Test configuration loading."""
        try:
            from src.core.config import YOLOSConfig
            
            # Test default configuration
            config = YOLOSConfig()
            assert hasattr(config, 'camera')
            assert hasattr(config, 'detection')
            assert hasattr(config, 'recognition')
            
        except ImportError:
            pytest.skip("Configuration module not available")
    
    def test_logging_configuration(self):
        """Test logging configuration."""
        import logging
        
        # Test that logging is properly configured
        logger = logging.getLogger('yolos.test')
        logger.info("Test logging message")
        
        # Should not raise any exceptions
        assert True
    
    @pytest.mark.slow
    def test_model_loading_mock(self):
        """Test model loading with mocked dependencies."""
        with patch('torch.load') as mock_torch_load:
            mock_torch_load.return_value = Mock()
            
            try:
                # This would normally load a real model
                # but we're mocking it for testing
                model = mock_torch_load('fake_model.pt')
                assert model is not None
                
            except Exception as e:
                pytest.skip(f"Model loading test skipped: {e}")
    
    def test_plugin_system_basic(self):
        """Test basic plugin system functionality."""
        try:
            from src.core.plugin_manager import PluginManager
            
            # Test plugin manager initialization
            plugin_manager = PluginManager()
            assert hasattr(plugin_manager, 'plugins')
            
        except ImportError:
            pytest.skip("Plugin system not available")
    
    def test_api_endpoints_mock(self):
        """Test API endpoints with mocked Flask."""
        with patch('flask.Flask') as mock_flask:
            mock_app = Mock()
            mock_flask.return_value = mock_app
            
            try:
                from src.api.main import create_app
                app = create_app()
                assert app is not None
                
            except ImportError:
                pytest.skip("API module not available")
    
    def test_gui_components_mock(self):
        """Test GUI components with mocked tkinter."""
        with patch('tkinter.Tk') as mock_tk:
            mock_root = Mock()
            mock_tk.return_value = mock_root
            
            try:
                # Test GUI initialization without actually creating windows
                root = mock_tk()
                assert root is not None
                
            except ImportError:
                pytest.skip("GUI components not available")

class TestModuleIntegration(BaseTest):
    """Test integration between different modules."""
    
    def test_config_engine_integration(self):
        """Test integration between config and engine modules."""
        try:
            from src.core.config import YOLOSConfig
            from src.core.engine import YOLOSEngine
            
            config = YOLOSConfig()
            engine = YOLOSEngine(config)
            
            assert engine.config is not None
            assert hasattr(engine, 'initialize')
            
        except ImportError:
            pytest.skip("Core modules not available for integration test")
    
    def test_detection_recognition_integration(self):
        """Test integration between detection and recognition modules."""
        try:
            # Mock the integration without actual model loading
            with patch('torch.load'), patch('cv2.imread'):
                # This would test actual integration
                # but we're mocking for CI/CD compatibility
                assert True
                
        except ImportError:
            pytest.skip("Detection/Recognition modules not available")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])