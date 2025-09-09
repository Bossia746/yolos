"""
模型转换器 - 支持多种格式转换和优化
"""

import torch
import onnx
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    import openvino as ov
    OV_AVAILABLE = True
except ImportError:
    OV_AVAILABLE = False


class ModelConverter:
    """模型转换器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 支持的格式
        self.supported_formats = ['onnx', 'torchscript', 'tensorrt', 'openvino']
        
        # 检查可用的转换器
        if not TRT_AVAILABLE:
            self.supported_formats.remove('tensorrt')
            self.logger.warning("TensorRT不可用，无法转换为TensorRT格式")
        
        if not OV_AVAILABLE:
            self.supported_formats.remove('openvino')
            self.logger.warning("OpenVINO不可用，无法转换为OpenVINO格式")
    
    def convert_to_onnx(self, 
                       model_path: str,
                       output_path: str,
                       input_size: Tuple[int, int] = (640, 640),
                       batch_size: int = 1,
                       dynamic_axes: bool = True,
                       opset_version: int = 11) -> str:
        """
        转换为ONNX格式
        
        Args:
            model_path: 输入模型路径
            output_path: 输出ONNX路径
            input_size: 输入尺寸
            batch_size: 批量大小
            dynamic_axes: 是否使用动态轴
            opset_version: ONNX操作集版本
            
        Returns:
            输出文件路径
        """
        try:
            # 加载PyTorch模型
            model = torch.load(model_path, map_location='cpu')
            if hasattr(model, 'model'):
                model = model.model
            
            model.eval()
            
            # 创建示例输入
            dummy_input = torch.randn(batch_size, 3, *input_size)
            
            # 设置动态轴
            dynamic_axes_dict = None
            if dynamic_axes:
                dynamic_axes_dict = {
                    'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                    'output': {0: 'batch_size'}
                }
            
            # 导出ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes_dict,
                verbose=False
            )
            
            # 验证ONNX模型
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            self.logger.info(f"ONNX转换成功: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"ONNX转换失败: {e}")
            raise
    
    def convert_to_torchscript(self,
                              model_path: str,
                              output_path: str,
                              input_size: Tuple[int, int] = (640, 640),
                              method: str = 'trace') -> str:
        """
        转换为TorchScript格式
        
        Args:
            model_path: 输入模型路径
            output_path: 输出TorchScript路径
            input_size: 输入尺寸
            method: 转换方法 ('trace' 或 'script')
            
        Returns:
            输出文件路径
        """
        try:
            # 加载模型
            model = torch.load(model_path, map_location='cpu')
            if hasattr(model, 'model'):
                model = model.model
            
            model.eval()
            
            if method == 'trace':
                # 使用trace方法
                dummy_input = torch.randn(1, 3, *input_size)
                traced_model = torch.jit.trace(model, dummy_input)
                traced_model.save(output_path)
            else:
                # 使用script方法
                scripted_model = torch.jit.script(model)
                scripted_model.save(output_path)
            
            self.logger.info(f"TorchScript转换成功: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"TorchScript转换失败: {e}")
            raise
    
    def convert_to_tensorrt(self,
                           onnx_path: str,
                           output_path: str,
                           precision: str = 'fp16',
                           max_batch_size: int = 1,
                           workspace_size: int = 1 << 30) -> str:
        """
        转换为TensorRT格式
        
        Args:
            onnx_path: ONNX模型路径
            output_path: 输出TensorRT引擎路径
            precision: 精度 ('fp32', 'fp16', 'int8')
            max_batch_size: 最大批量大小
            workspace_size: 工作空间大小
            
        Returns:
            输出文件路径
        """
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT不可用")
        
        try:
            # 创建TensorRT构建器
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            config = builder.create_builder_config()
            
            # 设置工作空间
            config.max_workspace_size = workspace_size
            
            # 设置精度
            if precision == 'fp16':
                config.set_flag(trt.BuilderFlag.FP16)
            elif precision == 'int8':
                config.set_flag(trt.BuilderFlag.INT8)
                # 这里需要设置校准器
                # config.int8_calibrator = calibrator
            
            # 解析ONNX模型
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    raise RuntimeError("ONNX解析失败")
            
            # 构建引擎
            engine = builder.build_engine(network, config)
            
            # 保存引擎
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            self.logger.info(f"TensorRT转换成功: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"TensorRT转换失败: {e}")
            raise
    
    def convert_to_openvino(self,
                           onnx_path: str,
                           output_dir: str,
                           precision: str = 'FP16') -> str:
        """
        转换为OpenVINO格式
        
        Args:
            onnx_path: ONNX模型路径
            output_dir: 输出目录
            precision: 精度 ('FP32', 'FP16', 'INT8')
            
        Returns:
            输出文件路径
        """
        if not OV_AVAILABLE:
            raise ImportError("OpenVINO不可用")
        
        try:
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 转换模型
            core = ov.Core()
            model = core.read_model(onnx_path)
            
            # 编译模型
            if precision == 'FP16':
                # 转换为FP16
                from openvino.tools import mo
                compressed_model = mo.convert_model(onnx_path, compress_to_fp16=True)
                model = compressed_model
            
            # 保存模型
            xml_path = output_path / "model.xml"
            bin_path = output_path / "model.bin"
            
            ov.serialize(model, str(xml_path))
            
            self.logger.info(f"OpenVINO转换成功: {xml_path}")
            return str(xml_path)
            
        except Exception as e:
            self.logger.error(f"OpenVINO转换失败: {e}")
            raise
    
    def quantize_model(self,
                      model_path: str,
                      output_path: str,
                      calibration_data: Optional[np.ndarray] = None,
                      method: str = 'dynamic') -> str:
        """
        量化模型
        
        Args:
            model_path: 输入模型路径
            output_path: 输出模型路径
            calibration_data: 校准数据
            method: 量化方法 ('dynamic', 'static')
            
        Returns:
            输出文件路径
        """
        try:
            if model_path.endswith('.onnx'):
                return self._quantize_onnx(model_path, output_path, method)
            else:
                return self._quantize_pytorch(model_path, output_path, method)
                
        except Exception as e:
            self.logger.error(f"模型量化失败: {e}")
            raise
    
    def _quantize_onnx(self, model_path: str, output_path: str, method: str) -> str:
        """量化ONNX模型"""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            if method == 'dynamic':
                quantize_dynamic(
                    model_path,
                    output_path,
                    weight_type=QuantType.QUInt8
                )
            else:
                # 静态量化需要校准数据
                raise NotImplementedError("ONNX静态量化暂未实现")
            
            return output_path
            
        except ImportError:
            raise ImportError("需要安装onnxruntime: pip install onnxruntime")
    
    def _quantize_pytorch(self, model_path: str, output_path: str, method: str) -> str:
        """量化PyTorch模型"""
        try:
            model = torch.load(model_path, map_location='cpu')
            if hasattr(model, 'model'):
                model = model.model
            
            model.eval()
            
            if method == 'dynamic':
                # 动态量化
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear, torch.nn.Conv2d},
                    dtype=torch.qint8
                )
            else:
                # 静态量化
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model, inplace=True)
                
                # 这里需要运行校准数据
                # calibrate(model, calibration_data)
                
                quantized_model = torch.quantization.convert(model, inplace=False)
            
            # 保存量化模型
            torch.save(quantized_model, output_path)
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"PyTorch量化失败: {e}")
    
    def prune_model(self,
                   model_path: str,
                   output_path: str,
                   sparsity: float = 0.3,
                   method: str = 'magnitude') -> str:
        """
        剪枝模型
        
        Args:
            model_path: 输入模型路径
            output_path: 输出模型路径
            sparsity: 稀疏度
            method: 剪枝方法
            
        Returns:
            输出文件路径
        """
        try:
            import torch.nn.utils.prune as prune
            
            model = torch.load(model_path, map_location='cpu')
            if hasattr(model, 'model'):
                model = model.model
            
            # 对所有卷积层和线性层进行剪枝
            parameters_to_prune = []
            for module in model.modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    parameters_to_prune.append((module, 'weight'))
            
            # 全局剪枝
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=sparsity,
            )
            
            # 移除剪枝重参数化
            for module, param in parameters_to_prune:
                prune.remove(module, param)
            
            # 保存剪枝后的模型
            torch.save(model, output_path)
            
            self.logger.info(f"模型剪枝成功: {output_path}, 稀疏度: {sparsity}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"模型剪枝失败: {e}")
            raise
    
    def optimize_model(self,
                      model_path: str,
                      output_path: str,
                      optimizations: list = None) -> str:
        """
        优化模型
        
        Args:
            model_path: 输入模型路径
            output_path: 输出模型路径
            optimizations: 优化选项列表
            
        Returns:
            输出文件路径
        """
        if optimizations is None:
            optimizations = ['quantize', 'prune']
        
        current_path = model_path
        
        try:
            # 应用优化
            for opt in optimizations:
                temp_path = f"temp_{opt}_{Path(current_path).name}"
                
                if opt == 'quantize':
                    current_path = self.quantize_model(current_path, temp_path)
                elif opt == 'prune':
                    current_path = self.prune_model(current_path, temp_path, sparsity=0.2)
                
            # 移动到最终输出路径
            import shutil
            shutil.move(current_path, output_path)
            
            # 清理临时文件
            for opt in optimizations:
                temp_path = f"temp_{opt}_{Path(model_path).name}"
                if Path(temp_path).exists():
                    Path(temp_path).unlink()
            
            self.logger.info(f"模型优化完成: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"模型优化失败: {e}")
            raise
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """获取模型信息"""
        try:
            info = {
                'path': model_path,
                'format': Path(model_path).suffix[1:],
                'size_mb': Path(model_path).stat().st_size / (1024 * 1024)
            }
            
            if model_path.endswith('.pt'):
                # PyTorch模型信息
                model = torch.load(model_path, map_location='cpu')
                if hasattr(model, 'model'):
                    model = model.model
                
                # 计算参数数量
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                info.update({
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params,
                    'model_type': type(model).__name__
                })
                
            elif model_path.endswith('.onnx'):
                # ONNX模型信息
                onnx_model = onnx.load(model_path)
                info.update({
                    'ir_version': onnx_model.ir_version,
                    'opset_version': onnx_model.opset_import[0].version,
                    'producer_name': onnx_model.producer_name
                })
            
            return info
            
        except Exception as e:
            self.logger.error(f"获取模型信息失败: {e}")
            return {'error': str(e)}