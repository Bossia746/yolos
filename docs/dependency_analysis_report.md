# YOLOS项目依赖分析报告

## 📊 项目概览

- **总模块数**: 140
- **总依赖数**: 41
- **循环依赖**: 0 个
- **冗余问题**: 4 个
- **复杂度问题**: 122 个

## 🔄 循环依赖分析

✅ **未发现循环依赖**

## 🔍 冗余分析

### 过多GUI模块

发现 5 个GUI模块:
- basic_pet_recognition_gui
- tests.simple_multimodal_gui_test
- tests.test_gui_multimodal
- src.gui.main_gui
- src.gui.yolos_training_gui

**建议**: 考虑合并相似的GUI模块

### 重复类名

- **BasicPetRecognitionGUI**: basic_pet_recognition_gui, src.gui.main_gui
- **PerformanceMetrics**: tests.performance_test, src.utils.metrics
- **RecognitionTask**: src.api.external_api_system, src.recognition.priority_recognition_system
- **CameraConfig**: src.core.config, src.recognition.usb_medical_camera_system
- **TrainingConfig**: src.core.config, src.training.enhanced_human_trainer
- **YOLOSConfig**: src.core.config, src.sdk.yolos_client_sdk
- **ConfigManager**: src.core.config_manager, src.utils.config_manager, src.utils.file_utils
- **DetectionResult**: src.core.engine, src.recognition.optimized_multimodal_detector
- **RecognitionResult**: src.core.engine, src.recognition.improved_multimodal_detector, src.recognition.integrated_self_learning_recognition, src.recognition.intelligent_recognition_system, src.sdk.yolos_client_sdk
- **YOLOSLogger**: src.core.logger, src.utils.logging_manager
- **YOLOSTrainingGUI**: src.gui.main_gui, src.gui.yolos_training_gui
- **DatasetManager**: src.gui.yolos_training_gui, src.training.dataset_manager
- **OfflineTrainingManager**: src.gui.yolos_training_gui, src.training.offline_training_manager
- **ShapeDetector**: src.recognition.enhanced_object_recognizer, src.recognition.object_recognizer
- **TextDetector**: src.recognition.enhanced_object_recognizer, src.recognition.object_recognizer
- **PetSpecies**: src.recognition.enhanced_pet_recognizer, src.plugins.domain.pet_recognition
- **PetBehavior**: src.recognition.enhanced_pet_recognizer, src.plugins.domain.pet_recognition
- **PetHealthStatus**: src.recognition.enhanced_pet_recognizer, src.plugins.domain.pet_recognition
- **PetDetection**: src.recognition.enhanced_pet_recognizer, src.plugins.domain.pet_recognition
- **PetBehaviorResult**: src.recognition.enhanced_pet_recognizer, src.plugins.domain.pet_recognition
- **PetHealthResult**: src.recognition.enhanced_pet_recognizer, src.plugins.domain.pet_recognition
- **PetBehaviorAnalyzer**: src.recognition.enhanced_pet_recognizer, src.plugins.domain.pet_recognition
- **PetHealthMonitor**: src.recognition.enhanced_pet_recognizer, src.plugins.domain.pet_recognition
- **ImageQualityMetrics**: src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer
- **ImageQualityEnhancer**: src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer
- **AdaptiveImageProcessor**: src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer
- **ObjectCategory**: src.recognition.priority_recognition_system, src.plugins.domain.static_object_recognition
### 重复函数名

- **main**: basic_pet_recognition_gui, examples.api_usage_examples, examples.basic_detection, examples.mqtt_detection, examples.realtime_detection, examples.windows_multimodal_debug, scripts.analyze_dependencies, scripts.download_training_datasets, scripts.install, scripts.quick_start_enhanced_training, scripts.setup_hybrid_system, scripts.train_offline_models, tests.comprehensive_scenario_testing, tests.real_world_validation, tests.simple_multimodal_gui_test, tests.standalone_scenario_validator, tests.test_aiot_simple, tests.test_aiot_standalone, tests.test_gui_multimodal, tests.test_headless_multimodal, tests.test_multi_target_priority_system, tests.test_opencv_display, tests.test_optimized_multimodal, tests.test_quality_and_spoofing, tests.test_self_learning_system, esp32.yolos_esp32_cam.esp32_recognition_adapter, ros_workspace.src.yolos_ros.scripts.detection_node, src.api.external_api_system, src.detection.cli, src.gui.main_gui, src.gui.yolos_training_gui, src.recognition.usb_medical_camera_system, src.ui.accessibility_manager, src.plugins.platform.raspberry_pi_adapter, src.plugins.platform.ros_integration
- **__init__**: basic_pet_recognition_gui, basic_pet_recognition_gui, examples.mqtt_detection, scripts.analyze_dependencies, scripts.download_training_datasets, scripts.install, scripts.quick_start_enhanced_training, scripts.setup_hybrid_system, tests.comprehensive_scenario_testing, tests.integration_test, tests.integration_test, tests.integration_test, tests.integration_test, tests.integration_test, tests.mock_data, tests.performance_test, tests.performance_test, tests.performance_test, tests.performance_test, tests.performance_test, tests.performance_test, tests.real_world_validation, tests.simple_multimodal_gui_test, tests.standalone_scenario_validator, tests.test_config, tests.test_gui_multimodal, tests.test_headless_multimodal, tests.test_multi_target_priority_system, tests.test_optimized_multimodal, tests.test_self_learning_system, esp32.yolos_esp32_cam.esp32_recognition_adapter, ros_workspace.src.yolos_ros.scripts.detection_node, src.api.external_api_system, src.api.external_api_system, src.communication.external_communication_system, src.communication.mqtt_client, src.core.base_plugin, src.core.base_plugin, src.core.config, src.core.config_manager, src.core.cross_platform_manager, src.core.data_manager, src.core.data_manager, src.core.data_manager, src.core.data_manager, src.core.engine, src.core.event_bus, src.core.event_bus, src.core.integrated_aiot_platform, src.core.logger, src.core.modular_extension_manager, src.core.platform_adapter, src.core.plugin_manager, src.core.storage_factory, src.core.storage_factory, src.core.unified_config_manager, src.detection.camera_detector, src.detection.image_detector, src.detection.realtime_detector, src.detection.video_detector, src.gui.main_gui, src.gui.yolos_training_gui, src.gui.yolos_training_gui, src.gui.yolos_training_gui, src.gui.yolos_training_gui, src.gui.yolos_training_gui, src.gui.yolos_training_gui, src.gui.yolos_training_gui, src.models.base_model, src.models.model_converter, src.models.pretrained_model_loader, src.models.pretrained_model_loader, src.models.pretrained_resources_manager, src.models.pretrained_resources_manager, src.models.pretrained_resources_manager, src.models.yolov5_model, src.models.yolov8_model, src.models.yolo_world_model, src.recognition.anti_spoofing_detector, src.recognition.emergency_response_system, src.recognition.enhanced_face_recognizer, src.recognition.enhanced_fall_detection_system, src.recognition.enhanced_gesture_recognizer, src.recognition.enhanced_object_recognizer, src.recognition.enhanced_object_recognizer, src.recognition.enhanced_object_recognizer, src.recognition.enhanced_object_recognizer, src.recognition.enhanced_object_recognizer, src.recognition.enhanced_object_recognizer, src.recognition.enhanced_object_recognizer, src.recognition.enhanced_object_recognizer, src.recognition.enhanced_pet_recognizer, src.recognition.enhanced_pet_recognizer, src.recognition.enhanced_pet_recognizer, src.recognition.enhanced_pet_recognizer, src.recognition.enhanced_pet_recognizer, src.recognition.enhanced_scene_analyzer, src.recognition.enhanced_scene_analyzer, src.recognition.face_recognizer, src.recognition.fall_detector, src.recognition.gesture_recognizer, src.recognition.hybrid_recognition_system, src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer, src.recognition.improved_multimodal_detector, src.recognition.improved_multimodal_detector, src.recognition.integrated_self_learning_recognition, src.recognition.intelligent_multi_target_system, src.recognition.intelligent_recognition_system, src.recognition.llm_self_learning_system, src.recognition.medical_facial_analyzer, src.recognition.medication_recognition_system, src.recognition.multimodal_detector, src.recognition.object_recognizer, src.recognition.object_recognizer, src.recognition.optimized_face_recognizer, src.recognition.optimized_fall_detector, src.recognition.optimized_fall_detector, src.recognition.optimized_gesture_recognizer, src.recognition.optimized_multimodal_detector, src.recognition.optimized_multimodal_detector, src.recognition.optimized_pose_recognizer, src.recognition.pose_recognizer, src.recognition.priority_recognition_system, src.recognition.usb_medical_camera_system, src.recognition.yolov7_pose_recognizer, src.safety.medical_safety_manager, src.safety.medical_safety_manager, src.sdk.yolos_client_sdk, src.training.augmentation, src.training.augmentation, src.training.dataset_manager, src.training.enhanced_human_trainer, src.training.enhanced_human_trainer, src.training.enhanced_human_trainer, src.training.offline_training_manager, src.training.offline_training_manager, src.training.trainer, src.ui.accessibility_manager, src.ui.accessibility_manager, src.utils.config_manager, src.utils.file_utils, src.utils.file_utils, src.utils.logging_manager, src.utils.metrics, src.utils.metrics, src.utils.visualization, src.utils.visualization, src.plugins.domain.dynamic_object_recognition, src.plugins.domain.dynamic_object_recognition, src.plugins.domain.dynamic_object_recognition, src.plugins.domain.human_recognition, src.plugins.domain.pet_recognition, src.plugins.domain.pet_recognition, src.plugins.domain.pet_recognition, src.plugins.domain.plant_recognition, src.plugins.domain.plant_recognition, src.plugins.domain.static_object_recognition, src.plugins.domain.static_object_recognition, src.plugins.domain.static_object_recognition, src.plugins.hardware.robotic_arm_controller, src.plugins.platform.aiot_boards_adapter, src.plugins.platform.arduino_adapter, src.plugins.platform.arduino_adapter, src.plugins.platform.esp32_plugin, src.plugins.platform.esp32_plugin, src.plugins.platform.esp32_plugin, src.plugins.platform.esp32_plugin, src.plugins.platform.esp32_plugin, src.plugins.platform.raspberry_pi_adapter, src.plugins.platform.raspberry_pi_plugin, src.plugins.platform.raspberry_pi_plugin, src.plugins.platform.raspberry_pi_plugin, src.plugins.platform.raspberry_pi_plugin, src.plugins.platform.raspberry_pi_plugin, src.plugins.platform.ros_integration, src.plugins.platform.ros_integration, src.plugins.platform.ros_integration
- **initialize_camera**: basic_pet_recognition_gui, tests.simple_multimodal_gui_test, tests.test_gui_multimodal, src.gui.yolos_training_gui, src.recognition.usb_medical_camera_system
- **draw_detections**: basic_pet_recognition_gui, src.gui.yolos_training_gui
- **run**: basic_pet_recognition_gui, examples.mqtt_detection, ros_workspace.src.yolos_ros.scripts.detection_node, src.api.external_api_system, src.gui.main_gui, src.gui.main_gui, src.gui.main_gui, src.gui.yolos_training_gui, src.ui.accessibility_manager, src.plugins.platform.ros_integration
- **cleanup**: basic_pet_recognition_gui, tests.simple_multimodal_gui_test, tests.test_gui_multimodal, esp32.yolos_esp32_cam.esp32_recognition_adapter, src.core.base_plugin, src.recognition.enhanced_pet_recognizer, src.recognition.integrated_self_learning_recognition, src.recognition.optimized_fall_detector, src.recognition.optimized_multimodal_detector, src.recognition.optimized_pose_recognizer, src.plugins.domain.dynamic_object_recognition, src.plugins.domain.human_recognition, src.plugins.domain.pet_recognition, src.plugins.domain.plant_recognition, src.plugins.domain.static_object_recognition, src.plugins.platform.esp32_plugin, src.plugins.platform.raspberry_pi_adapter, src.plugins.platform.raspberry_pi_plugin
- **detection_callback**: examples.realtime_detection, examples.windows_multimodal_debug
- **_generate_markdown_report**: scripts.analyze_dependencies, tests.standalone_scenario_validator
- **setUp**: tests.base_test, tests.base_test, tests.test_aiot_compatibility, tests.test_aiot_compatibility
- **tearDown**: tests.base_test, tests.base_test
- **test_plugin_lifecycle**: tests.base_test, tests.integration_test
- **_setup_logger**: tests.comprehensive_scenario_testing, tests.real_world_validation, tests.standalone_scenario_validator, src.safety.medical_safety_manager, src.sdk.yolos_client_sdk, src.ui.accessibility_manager
- **run_all_tests**: tests.comprehensive_scenario_testing, tests.integration_test, tests.test_headless_multimodal, tests.test_optimized_multimodal, tests.test_self_learning_system
- **generate_test_report**: tests.comprehensive_scenario_testing, tests.integration_test, tests.test_gui_multimodal, tests.test_self_learning_system
- **_generate_recommendations**: tests.comprehensive_scenario_testing, src.recognition.enhanced_pet_recognizer, src.recognition.intelligent_recognition_system, src.plugins.domain.plant_recognition
- **__post_init__**: tests.integration_test, src.api.external_api_system, src.recognition.priority_recognition_system, src.plugins.domain.human_recognition, src.plugins.platform.esp32_plugin, src.plugins.platform.raspberry_pi_plugin
- **validate_all_scenarios**: tests.real_world_validation, tests.standalone_scenario_validator
- **validate_technical_feasibility**: tests.real_world_validation, tests.standalone_scenario_validator
- **generate_validation_report**: tests.real_world_validation, tests.standalone_scenario_validator
- **detect_multimodal**: tests.simple_multimodal_gui_test, src.recognition.improved_multimodal_detector
- **draw_info_overlay**: tests.simple_multimodal_gui_test, tests.test_gui_multimodal
- **save_screenshot**: tests.simple_multimodal_gui_test, tests.test_gui_multimodal
- **reset_stats**: tests.simple_multimodal_gui_test, tests.test_gui_multimodal, src.core.engine, src.recognition.multimodal_detector
- **generate_report**: tests.simple_multimodal_gui_test, src.utils.metrics
- **test_supported_boards_coverage**: tests.test_aiot_compatibility, tests.test_aiot_simple
- **setup_test_environment**: tests.test_headless_multimodal, tests.test_optimized_multimodal
- **_generate_test_images**: tests.test_headless_multimodal, tests.test_optimized_multimodal
- **test_encoding**: tests.test_headless_multimodal, tests.test_optimized_multimodal
- **test_camera_access**: tests.test_headless_multimodal, tests.test_optimized_multimodal
- **test_detector_initialization**: tests.test_headless_multimodal, tests.test_optimized_multimodal
- **test_performance**: tests.test_headless_multimodal, tests.test_optimized_multimodal
- **_generate_test_report**: tests.test_headless_multimodal, tests.test_optimized_multimodal
- **test_configuration_loading**: tests.test_modular_aiot_platform, tests.test_self_learning_system
- **run_comprehensive_test**: tests.test_modular_aiot_platform, tests.test_usb_medical_camera
- **create_test_images**: tests.test_quality_and_spoofing, tests.test_self_learning_system
- **create_fall_detection_image**: tests.test_self_learning_system, tests.test_usb_medical_camera
- **detect_image**: web.app, src.detection.image_detector
- **get_stats**: web.app, src.communication.mqtt_client, src.core.data_manager, src.core.event_bus, src.detection.camera_detector, src.recognition.multimodal_detector, src.plugins.domain.human_recognition
- **capture_image**: esp32.yolos_esp32_cam.esp32_recognition_adapter, src.gui.yolos_training_gui, src.gui.yolos_training_gui, src.plugins.platform.esp32_plugin, src.plugins.platform.raspberry_pi_adapter, src.plugins.platform.raspberry_pi_plugin
- **recognize**: esp32.yolos_esp32_cam.esp32_recognition_adapter, src.recognition.hybrid_recognition_system, src.recognition.integrated_self_learning_recognition, src.recognition.intelligent_recognition_system, src.plugins.platform.raspberry_pi_adapter
- **continuous_recognition**: esp32.yolos_esp32_cam.esp32_recognition_adapter, src.plugins.platform.raspberry_pi_adapter
- **get_status**: esp32.yolos_esp32_cam.esp32_recognition_adapter, src.core.base_plugin, src.plugins.hardware.robotic_arm_controller, src.plugins.platform.esp32_plugin, src.plugins.platform.raspberry_pi_plugin
- **image_callback**: ros_workspace.src.yolos_ros.scripts.detection_node, src.plugins.platform.ros_integration, src.plugins.platform.ros_integration
- **publish_annotated_image**: ros_workspace.src.yolos_ros.scripts.detection_node, src.plugins.platform.ros_integration, src.plugins.platform.ros_integration
- **_load_config**: src.api.external_api_system, src.recognition.intelligent_multi_target_system
- **health_check**: src.api.external_api_system, src.sdk.yolos_client_sdk
- **listen_voice_command**: src.api.external_api_system, src.sdk.yolos_client_sdk
- **move_device**: src.api.external_api_system, src.sdk.yolos_client_sdk
- **start_recognition_task**: src.api.external_api_system, src.sdk.yolos_client_sdk
- **get_task_status**: src.api.external_api_system, src.sdk.yolos_client_sdk
- **get_device_status**: src.api.external_api_system, src.sdk.yolos_client_sdk
- **recognize_image**: src.api.external_api_system, src.sdk.yolos_client_sdk
- **connect**: src.communication.mqtt_client, src.core.base_plugin, src.sdk.yolos_client_sdk, src.plugins.platform.arduino_adapter, src.plugins.platform.esp32_plugin
- **disconnect**: src.communication.mqtt_client, src.core.base_plugin, src.sdk.yolos_client_sdk, src.plugins.platform.arduino_adapter, src.plugins.platform.esp32_plugin
- **set_callbacks**: src.communication.mqtt_client, src.detection.camera_detector, src.recognition.multimodal_detector, src.recognition.optimized_multimodal_detector
- **__enter__**: src.communication.mqtt_client, src.core.engine, src.core.event_bus
- **__exit__**: src.communication.mqtt_client, src.core.engine, src.core.event_bus
- **to_dict**: src.core.base_plugin, src.core.config
- **initialize**: src.core.base_plugin, src.core.engine, src.core.platform_adapter, src.plugins.domain.dynamic_object_recognition, src.plugins.domain.human_recognition, src.plugins.domain.pet_recognition, src.plugins.domain.plant_recognition, src.plugins.domain.static_object_recognition, src.plugins.platform.esp32_plugin, src.plugins.platform.esp32_plugin, src.plugins.platform.raspberry_pi_plugin, src.plugins.platform.raspberry_pi_plugin
- **process**: src.core.base_plugin, src.core.base_plugin, src.core.base_plugin, src.core.base_plugin, src.training.augmentation, src.plugins.platform.arduino_adapter, src.plugins.platform.arduino_adapter, src.plugins.platform.arduino_adapter, src.plugins.platform.arduino_adapter, src.plugins.platform.arduino_adapter
- **get_config**: src.core.base_plugin, src.core.config, src.utils.file_utils
- **__str__**: src.core.base_plugin, src.core.config
- **__repr__**: src.core.base_plugin, src.core.config
- **detect**: src.core.base_plugin, src.gui.yolos_training_gui, src.recognition.multimodal_detector, src.recognition.optimized_multimodal_detector, src.plugins.domain.dynamic_object_recognition, src.plugins.domain.dynamic_object_recognition, src.plugins.domain.pet_recognition, src.plugins.domain.pet_recognition, src.plugins.domain.plant_recognition, src.plugins.domain.plant_recognition, src.plugins.domain.static_object_recognition, src.plugins.domain.static_object_recognition
- **load_config**: src.core.config, src.core.config_manager, src.core.storage_factory, src.utils.file_utils
- **save**: src.core.config, src.utils.config_manager
- **validate**: src.core.config, src.models.yolov8_model, src.training.trainer, src.utils.config_manager
- **update**: src.core.config, src.utils.config_manager
- **_calculate_checksum**: src.core.config_manager, src.core.data_manager
- **_deep_merge**: src.core.config_manager, src.utils.config_manager
- **get**: src.core.config_manager, src.core.data_manager, src.utils.config_manager
- **set**: src.core.config_manager, src.utils.config_manager
- **get_all**: src.core.config_manager, src.utils.config_manager
- **_detect_hardware**: src.core.cross_platform_manager, src.plugins.platform.raspberry_pi_plugin
- **_detect_gpu**: src.core.cross_platform_manager, src.core.plugin_manager
- **_get_optimization_config**: src.core.cross_platform_manager, src.plugins.platform.aiot_boards_adapter
- **_check_torch_cuda**: src.core.cross_platform_manager, src.plugins.platform.aiot_boards_adapter
- **store**: src.core.data_manager, src.core.data_manager
- **retrieve**: src.core.data_manager, src.core.data_manager
- **delete**: src.core.data_manager, src.core.data_manager, src.utils.config_manager
- **list_data**: src.core.data_manager, src.core.data_manager
- **get_storage_info**: src.core.data_manager, src.core.data_manager
- **close**: src.core.data_manager, src.recognition.enhanced_face_recognizer, src.recognition.face_recognizer, src.recognition.gesture_recognizer, src.recognition.multimodal_detector, src.recognition.optimized_face_recognizer, src.recognition.optimized_gesture_recognizer, src.recognition.pose_recognizer, src.recognition.yolov7_pose_recognizer, src.sdk.yolos_client_sdk
- **process_frame**: src.core.engine, src.gui.yolos_training_gui, src.recognition.emergency_response_system, src.plugins.domain.dynamic_object_recognition, src.plugins.domain.human_recognition, src.plugins.domain.pet_recognition, src.plugins.domain.plant_recognition, src.plugins.domain.static_object_recognition
- **process_image**: src.core.engine, src.plugins.platform.ros_integration, src.plugins.platform.ros_integration
- **stop**: src.core.engine, src.detection.realtime_detector, src.recognition.optimized_multimodal_detector
- **get_performance_stats**: src.core.engine, src.recognition.optimized_face_recognizer, src.recognition.optimized_gesture_recognizer
- **_update_performance_stats**: src.core.engine, src.recognition.optimized_face_recognizer, src.recognition.optimized_fall_detector, src.recognition.optimized_gesture_recognizer, src.recognition.optimized_pose_recognizer, src.plugins.platform.raspberry_pi_adapter
- **__call__**: src.core.event_bus, src.models.pretrained_resources_manager
- **get_recent_events**: src.core.event_bus, src.recognition.enhanced_fall_detection_system
- **shutdown**: src.core.event_bus, src.core.plugin_manager, src.recognition.intelligent_multi_target_system, src.recognition.priority_recognition_system
- **log_function_call**: src.core.logger, src.utils.logging_manager
- **log_class_methods**: src.core.logger, src.utils.logging_manager
- **get_logger**: src.core.logger, src.core.logger, src.utils.logger, src.utils.logging_manager
- **__new__**: src.core.logger, src.utils.logging_manager
- **_setup_handlers**: src.core.logger, src.utils.logging_manager
- **debug**: src.core.logger, src.utils.logging_manager
- **info**: src.core.logger, src.utils.logging_manager
- **warning**: src.core.logger, src.utils.logging_manager
- **error**: src.core.logger, src.utils.logging_manager
- **critical**: src.core.logger, src.utils.logging_manager
- **_log_with_context**: src.core.logger, src.utils.logging_manager
- **log_performance**: src.core.logger, src.utils.logging_manager
- **start_timer**: src.core.logger, src.utils.logging_manager
- **end_timer**: src.core.logger, src.utils.logging_manager
- **create_debug_snapshot**: src.core.logger, src.utils.logging_manager
- **decorator**: src.core.logger, src.core.logger, src.utils.logging_manager, src.utils.logging_manager
- **wrapper**: src.core.logger, src.utils.logging_manager
- **_check_dependencies**: src.core.modular_extension_manager, src.core.plugin_manager
- **get_platform_type**: src.core.platform_adapter, src.core.platform_adapter, src.core.platform_adapter, src.core.platform_adapter
- **detect_hardware**: src.core.platform_adapter, src.core.platform_adapter, src.core.platform_adapter, src.core.platform_adapter
- **get_resource_usage**: src.core.platform_adapter, src.core.platform_adapter, src.core.platform_adapter, src.core.platform_adapter
- **optimize_for_platform**: src.core.platform_adapter, src.core.platform_adapter, src.core.platform_adapter, src.core.platform_adapter
- **get_camera_devices**: src.core.platform_adapter, src.core.platform_adapter, src.core.platform_adapter, src.core.platform_adapter
- **get_compute_devices**: src.core.platform_adapter, src.core.platform_adapter, src.core.platform_adapter, src.core.platform_adapter
- **start_monitoring**: src.core.platform_adapter, src.recognition.emergency_response_system, src.recognition.usb_medical_camera_system
- **stop_monitoring**: src.core.platform_adapter, src.recognition.emergency_response_system, src.recognition.usb_medical_camera_system
- **_monitoring_loop**: src.core.platform_adapter, src.recognition.emergency_response_system
- **get_hardware_info**: src.core.platform_adapter, src.plugins.platform.esp32_plugin, src.plugins.platform.raspberry_pi_plugin
- **_get_cpu_temperature**: src.core.platform_adapter, src.plugins.platform.raspberry_pi_adapter
- **_get_default_config**: src.core.storage_factory, src.recognition.anti_spoofing_detector, src.recognition.emergency_response_system, src.recognition.enhanced_fall_detection_system, src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer, src.recognition.integrated_self_learning_recognition, src.recognition.intelligent_recognition_system, src.recognition.llm_self_learning_system, src.recognition.medical_facial_analyzer, src.recognition.priority_recognition_system, src.recognition.usb_medical_camera_system, src.plugins.platform.aiot_boards_adapter, src.plugins.platform.arduino_adapter, src.plugins.platform.raspberry_pi_adapter
- **set_detection_params**: src.detection.camera_detector, src.recognition.multimodal_detector
- **get_model_info**: src.detection.image_detector, src.detection.realtime_detector, src.detection.video_detector, src.models.base_model, src.models.model_converter, src.models.yolo_factory
- **update_status**: src.gui.main_gui, src.gui.yolos_training_gui
- **show_about**: src.gui.main_gui, src.gui.yolos_training_gui
- **on_closing**: src.gui.main_gui, src.gui.yolos_training_gui
- **start_recording**: src.gui.yolos_training_gui, src.gui.yolos_training_gui, src.sdk.yolos_client_sdk
- **stop_recording**: src.gui.yolos_training_gui, src.gui.yolos_training_gui, src.sdk.yolos_client_sdk
- **create_gui**: src.gui.yolos_training_gui, src.recognition.usb_medical_camera_system
- **start_training**: src.gui.yolos_training_gui, src.gui.yolos_training_gui
- **load_dataset**: src.gui.yolos_training_gui, src.gui.yolos_training_gui, src.training.dataset_manager
- **save_dataset**: src.gui.yolos_training_gui, src.gui.yolos_training_gui
- **export_model**: src.gui.yolos_training_gui, src.training.trainer
- **setup_logger**: src.gui.yolos_training_gui, src.utils.logger
- **train**: src.gui.yolos_training_gui, src.models.yolov8_model, src.training.enhanced_human_trainer, src.training.trainer
- **load_model**: src.models.base_model, src.models.pretrained_model_loader, src.models.yolov5_model, src.models.yolov8_model, src.models.yolo_world_model, src.training.trainer
- **predict**: src.models.base_model, src.models.pretrained_resources_manager, src.models.pretrained_resources_manager, src.models.yolov5_model, src.models.yolov8_model, src.models.yolo_world_model
- **preprocess**: src.models.base_model, src.models.yolov5_model, src.models.yolov8_model, src.models.yolo_world_model
- **postprocess**: src.models.base_model, src.models.yolov5_model, src.models.yolov8_model, src.models.yolo_world_model
- **draw_results**: src.models.base_model, src.recognition.enhanced_pet_recognizer
- **list_available_models**: src.models.pretrained_model_loader, src.models.yolo_factory
- **create_model**: src.models.yolo_factory, src.training.enhanced_human_trainer
- **_init_detectors**: src.recognition.anti_spoofing_detector, src.recognition.optimized_pose_recognizer
- **_update_history**: src.recognition.anti_spoofing_detector, src.recognition.enhanced_gesture_recognizer, src.recognition.fall_detector, src.recognition.medical_facial_analyzer
- **_detect_fall**: src.recognition.emergency_response_system, src.recognition.priority_recognition_system, src.plugins.domain.human_recognition
- **get_system_status**: src.recognition.emergency_response_system, src.recognition.hybrid_recognition_system, src.recognition.intelligent_multi_target_system, src.recognition.optimized_multimodal_detector, src.plugins.platform.raspberry_pi_adapter
- **detect_faces**: src.recognition.enhanced_face_recognizer, src.recognition.face_recognizer, src.recognition.optimized_face_recognizer
- **_get_bbox_from_detection**: src.recognition.enhanced_face_recognizer, src.recognition.face_recognizer, src.recognition.optimized_face_recognizer
- **_extract_face_landmarks**: src.recognition.enhanced_face_recognizer, src.recognition.optimized_face_recognizer
- **_recognize_face_insightface**: src.recognition.enhanced_face_recognizer, src.recognition.optimized_face_recognizer
- **_recognize_face_traditional**: src.recognition.enhanced_face_recognizer, src.recognition.optimized_face_recognizer
- **_draw_enhanced_face_detection**: src.recognition.enhanced_face_recognizer, src.recognition.optimized_face_recognizer
- **_draw_face_detection**: src.recognition.enhanced_face_recognizer, src.recognition.face_recognizer, src.recognition.optimized_face_recognizer
- **_draw_face_landmarks**: src.recognition.enhanced_face_recognizer, src.recognition.optimized_face_recognizer
- **add_face_to_database**: src.recognition.enhanced_face_recognizer, src.recognition.face_recognizer, src.recognition.multimodal_detector
- **save_face_database**: src.recognition.enhanced_face_recognizer, src.recognition.face_recognizer, src.recognition.improved_multimodal_detector, src.recognition.multimodal_detector
- **load_face_database**: src.recognition.enhanced_face_recognizer, src.recognition.face_recognizer, src.recognition.improved_multimodal_detector, src.recognition.multimodal_detector
- **get_database_info**: src.recognition.enhanced_face_recognizer, src.recognition.face_recognizer
- **get_recognizer_info**: src.recognition.enhanced_face_recognizer, src.recognition.enhanced_gesture_recognizer
- **_initialize_models**: src.recognition.enhanced_fall_detection_system, src.recognition.medication_recognition_system
- **detect_fall**: src.recognition.enhanced_fall_detection_system, src.recognition.fall_detector, src.sdk.yolos_client_sdk
- **_detect_poses**: src.recognition.enhanced_fall_detection_system, src.recognition.optimized_multimodal_detector
- **_analyze_body_angle**: src.recognition.enhanced_fall_detection_system, src.recognition.optimized_fall_detector
- **_analyze_velocity**: src.recognition.enhanced_fall_detection_system, src.recognition.optimized_fall_detector
- **_determine_fall_status**: src.recognition.enhanced_fall_detection_system, src.recognition.optimized_fall_detector
- **_extract_landmarks**: src.recognition.enhanced_gesture_recognizer, src.recognition.gesture_recognizer, src.recognition.optimized_gesture_recognizer, src.recognition.pose_recognizer
- **_recognize_static_gesture**: src.recognition.enhanced_gesture_recognizer, src.recognition.optimized_gesture_recognizer
- **_get_finger_states**: src.recognition.enhanced_gesture_recognizer, src.recognition.gesture_recognizer
- **_recognize_dynamic_gesture**: src.recognition.enhanced_gesture_recognizer, src.recognition.optimized_gesture_recognizer
- **_analyze_trajectory**: src.recognition.enhanced_gesture_recognizer, src.recognition.optimized_gesture_recognizer
- **_is_circular_motion**: src.recognition.enhanced_gesture_recognizer, src.plugins.domain.dynamic_object_recognition
- **_calculate_hand_bbox**: src.recognition.enhanced_gesture_recognizer, src.recognition.optimized_gesture_recognizer
- **_calculate_hand_center_orientation**: src.recognition.enhanced_gesture_recognizer, src.recognition.optimized_gesture_recognizer
- **_detect_hand_interaction**: src.recognition.enhanced_gesture_recognizer, src.recognition.optimized_gesture_recognizer
- **draw_annotations**: src.recognition.enhanced_gesture_recognizer, src.recognition.optimized_gesture_recognizer
- **_get_default_result**: src.recognition.enhanced_gesture_recognizer, src.recognition.fall_detector, src.recognition.optimized_fall_detector, src.recognition.optimized_gesture_recognizer, src.recognition.optimized_multimodal_detector, src.recognition.optimized_pose_recognizer
- **_init_color_ranges**: src.recognition.enhanced_object_recognizer, src.recognition.object_recognizer
- **detect_objects**: src.recognition.enhanced_object_recognizer, src.recognition.object_recognizer
- **_detect_dominant_color**: src.recognition.enhanced_object_recognizer, src.recognition.object_recognizer
- **_get_color_name**: src.recognition.enhanced_object_recognizer, src.recognition.object_recognizer
- **_hsv_to_rgb**: src.recognition.enhanced_object_recognizer, src.recognition.object_recognizer
- **detect_shape**: src.recognition.enhanced_object_recognizer, src.recognition.object_recognizer
- **_classify_shape**: src.recognition.enhanced_object_recognizer, src.recognition.object_recognizer, src.plugins.platform.arduino_adapter
- **detect_text**: src.recognition.enhanced_object_recognizer, src.recognition.object_recognizer
- **_detect_text_ocr**: src.recognition.enhanced_object_recognizer, src.recognition.object_recognizer
- **_detect_text_basic**: src.recognition.enhanced_object_recognizer, src.recognition.object_recognizer
- **_load_model**: src.recognition.enhanced_pet_recognizer, src.plugins.domain.pet_recognition
- **_analyze_pose_type**: src.recognition.enhanced_pet_recognizer, src.plugins.domain.human_recognition
- **_calculate_body_orientation**: src.recognition.enhanced_pet_recognizer, src.recognition.optimized_fall_detector
- **analyze_behavior**: src.recognition.enhanced_pet_recognizer, src.plugins.domain.pet_recognition, src.plugins.domain.pet_recognition
- **_get_bbox_center**: src.recognition.enhanced_pet_recognizer, src.recognition.enhanced_pet_recognizer
- **assess_health**: src.recognition.enhanced_pet_recognizer, src.plugins.domain.pet_recognition, src.plugins.domain.pet_recognition
- **_analyze_health_indicators**: src.recognition.enhanced_pet_recognizer, src.recognition.priority_recognition_system
- **_detect_pets**: src.recognition.enhanced_pet_recognizer, src.recognition.enhanced_scene_analyzer
- **_assign_tracking_id**: src.recognition.enhanced_pet_recognizer, src.plugins.domain.dynamic_object_recognition, src.plugins.domain.pet_recognition, src.plugins.domain.plant_recognition, src.plugins.domain.static_object_recognition
- **get_statistics**: src.recognition.enhanced_pet_recognizer, src.recognition.optimized_fall_detector, src.recognition.optimized_multimodal_detector, src.recognition.optimized_pose_recognizer, src.recognition.priority_recognition_system, src.plugins.domain.dynamic_object_recognition, src.plugins.domain.pet_recognition, src.plugins.domain.plant_recognition, src.plugins.domain.static_object_recognition
- **reset_statistics**: src.recognition.enhanced_pet_recognizer, src.recognition.optimized_fall_detector, src.recognition.optimized_multimodal_detector, src.recognition.optimized_pose_recognizer
- **analyze**: src.recognition.enhanced_scene_analyzer, src.recognition.enhanced_scene_analyzer, src.recognition.enhanced_scene_analyzer, src.recognition.enhanced_scene_analyzer, src.recognition.enhanced_scene_analyzer, src.recognition.enhanced_scene_analyzer, src.recognition.enhanced_scene_analyzer
- **_detect_falls**: src.recognition.enhanced_scene_analyzer, src.recognition.optimized_multimodal_detector
- **_recognize_gesture**: src.recognition.enhanced_scene_analyzer, src.recognition.priority_recognition_system
- **_analyze_behavior**: src.recognition.enhanced_scene_analyzer, src.recognition.priority_recognition_system
- **_classify_plant_type**: src.recognition.enhanced_scene_analyzer, src.plugins.domain.plant_recognition
- **_analyze_colors**: src.recognition.enhanced_scene_analyzer, src.plugins.domain.plant_recognition
- **remove_face_from_database**: src.recognition.face_recognizer, src.recognition.multimodal_detector
- **reset**: src.recognition.fall_detector, src.utils.config_manager
- **_classify_gesture**: src.recognition.gesture_recognizer, src.plugins.domain.human_recognition
- **_preprocess_image**: src.recognition.hybrid_recognition_system, src.recognition.llm_self_learning_system, src.recognition.medication_recognition_system
- **_generate_cache_key**: src.recognition.hybrid_recognition_system, src.recognition.optimized_multimodal_detector
- **batch_recognize**: src.recognition.hybrid_recognition_system, src.recognition.integrated_self_learning_recognition
- **analyze_image_quality**: src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer
- **_estimate_noise_level**: src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer
- **_detect_reflection**: src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer
- **_calculate_quality_score**: src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer
- **enhance_image**: src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer
- **_adjust_brightness**: src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer
- **_enhance_contrast**: src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer
- **_reduce_reflection**: src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer
- **_correct_exposure**: src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer
- **_denoise_image**: src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer
- **_sharpen_image**: src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer
- **is_image_acceptable**: src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer
- **get_enhancement_recommendations**: src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer
- **process_image_stream**: src.recognition.image_quality_enhancer, src.recognition.image_quality_enhancer
- **_init_recognizers**: src.recognition.improved_multimodal_detector, src.recognition.optimized_multimodal_detector
- **_update_statistics**: src.recognition.improved_multimodal_detector, src.recognition.integrated_self_learning_recognition, src.recognition.intelligent_multi_target_system, src.recognition.intelligent_recognition_system, src.recognition.optimized_multimodal_detector, src.recognition.priority_recognition_system
- **get_performance_report**: src.recognition.improved_multimodal_detector, src.recognition.intelligent_recognition_system
- **_check_cache**: src.recognition.integrated_self_learning_recognition, src.recognition.intelligent_multi_target_system
- **_cache_result**: src.recognition.integrated_self_learning_recognition, src.recognition.optimized_multimodal_detector
- **_create_error_result**: src.recognition.integrated_self_learning_recognition, src.recognition.medical_facial_analyzer
- **update_configuration**: src.recognition.integrated_self_learning_recognition, src.recognition.llm_self_learning_system
- **deep_merge**: src.recognition.integrated_self_learning_recognition, src.recognition.llm_self_learning_system
- **_process_single_object**: src.recognition.intelligent_multi_target_system, src.recognition.priority_recognition_system
- **_calculate_confidence**: src.recognition.intelligent_recognition_system, src.recognition.medical_facial_analyzer, src.plugins.domain.dynamic_object_recognition
- **_detect_faces**: src.recognition.medical_facial_analyzer, src.recognition.optimized_multimodal_detector
- **_generate_medical_recommendations**: src.recognition.medical_facial_analyzer, src.safety.medical_safety_manager
- **detect_from_camera**: src.recognition.multimodal_detector, src.recognition.optimized_multimodal_detector
- **get_supported_gestures**: src.recognition.multimodal_detector, src.recognition.optimized_gesture_recognizer
- **_init_mediapipe**: src.recognition.optimized_face_recognizer, src.recognition.optimized_gesture_recognizer
- **_ensure_proper_encoding**: src.recognition.optimized_face_recognizer, src.recognition.optimized_gesture_recognizer, src.recognition.optimized_pose_recognizer
- **_detect_with_mediapipe**: src.recognition.optimized_face_recognizer, src.recognition.optimized_pose_recognizer
- **optimize_for_realtime**: src.recognition.optimized_face_recognizer, src.recognition.optimized_gesture_recognizer
- **_calculate_stability_score**: src.recognition.optimized_fall_detector, src.recognition.optimized_pose_recognizer
- **_get_performance_metrics**: src.recognition.optimized_fall_detector, src.recognition.optimized_gesture_recognizer, src.recognition.optimized_pose_recognizer
- **forward**: src.recognition.optimized_fall_detector, src.training.enhanced_human_trainer
- **_detect_fist**: src.recognition.optimized_gesture_recognizer, src.plugins.domain.human_recognition
- **_detect_open_palm**: src.recognition.optimized_gesture_recognizer, src.plugins.domain.human_recognition
- **_detect_thumbs_up**: src.recognition.optimized_gesture_recognizer, src.plugins.domain.human_recognition
- **_detect_peace_sign**: src.recognition.optimized_gesture_recognizer, src.plugins.domain.human_recognition
- **_detect_pointing**: src.recognition.optimized_gesture_recognizer, src.plugins.domain.human_recognition
- **_detect_gestures**: src.recognition.optimized_multimodal_detector, src.plugins.domain.human_recognition
- **detect_poses**: src.recognition.optimized_pose_recognizer, src.recognition.yolov7_pose_recognizer
- **_calculate_angle**: src.recognition.optimized_pose_recognizer, src.recognition.pose_recognizer
- **_classify_pose**: src.recognition.pose_recognizer, src.recognition.yolov7_pose_recognizer
- **_is_sitting**: src.recognition.pose_recognizer, src.recognition.yolov7_pose_recognizer
- **_is_lying**: src.recognition.pose_recognizer, src.recognition.yolov7_pose_recognizer
- **_is_arms_up**: src.recognition.pose_recognizer, src.recognition.yolov7_pose_recognizer
- **_is_waving**: src.recognition.pose_recognizer, src.recognition.yolov7_pose_recognizer
- **_is_clapping**: src.recognition.pose_recognizer, src.recognition.yolov7_pose_recognizer
- **_is_leaning_left**: src.recognition.pose_recognizer, src.recognition.yolov7_pose_recognizer
- **get_pose_info**: src.recognition.pose_recognizer, src.recognition.yolov7_pose_recognizer
- **__len__**: src.training.enhanced_human_trainer, src.training.offline_training_manager
- **__getitem__**: src.training.enhanced_human_trainer, src.training.offline_training_manager
- **get_supported_types**: src.plugins.domain.dynamic_object_recognition, src.plugins.domain.dynamic_object_recognition, src.plugins.domain.plant_recognition, src.plugins.domain.plant_recognition, src.plugins.domain.static_object_recognition, src.plugins.domain.static_object_recognition
- **analyze_motion**: src.plugins.domain.dynamic_object_recognition, src.plugins.domain.dynamic_object_recognition
- **_get_center**: src.plugins.domain.dynamic_object_recognition, src.plugins.domain.pet_recognition, src.plugins.domain.pet_recognition, src.plugins.domain.plant_recognition, src.plugins.domain.static_object_recognition, src.plugins.domain.static_object_recognition
- **get_supported_domains**: src.plugins.domain.dynamic_object_recognition, src.plugins.domain.human_recognition, src.plugins.domain.pet_recognition, src.plugins.domain.plant_recognition, src.plugins.domain.static_object_recognition
- **get_supported_species**: src.plugins.domain.pet_recognition, src.plugins.domain.pet_recognition
- **analyze_health**: src.plugins.domain.plant_recognition, src.plugins.domain.plant_recognition
- **monitor_growth**: src.plugins.domain.plant_recognition, src.plugins.domain.plant_recognition
- **monitor_state**: src.plugins.domain.static_object_recognition, src.plugins.domain.static_object_recognition
- **setup_pin**: src.plugins.platform.esp32_plugin, src.plugins.platform.raspberry_pi_plugin
- **digital_write**: src.plugins.platform.esp32_plugin, src.plugins.platform.raspberry_pi_plugin
- **digital_read**: src.plugins.platform.esp32_plugin, src.plugins.platform.raspberry_pi_plugin
- **capture_frame**: src.plugins.platform.esp32_plugin, src.plugins.platform.raspberry_pi_plugin
- **set_power_mode**: src.plugins.platform.esp32_plugin, src.plugins.platform.esp32_plugin
- **control_gpio**: src.plugins.platform.esp32_plugin, src.plugins.platform.raspberry_pi_plugin
- **compressed_image_callback**: src.plugins.platform.ros_integration, src.plugins.platform.ros_integration
- **scene_callback**: src.plugins.platform.ros_integration, src.plugins.platform.ros_integration
- **publish_results**: src.plugins.platform.ros_integration, src.plugins.platform.ros_integration

## 📊 复杂度分析

🟡 **large_file**: basic_pet_recognition_gui
  - 行数: 657

🟡 **too_many_definitions**: basic_pet_recognition_gui
  - 定义数: 22

🟡 **large_file**: examples.api_usage_examples
  - 行数: 537

🟡 **large_file**: scripts.install
  - 行数: 567

🟡 **too_many_definitions**: scripts.install
  - 定义数: 23

🟡 **too_many_definitions**: tests.base_test
  - 定义数: 36

🟡 **large_file**: tests.comprehensive_scenario_testing
  - 行数: 752

🟡 **too_many_definitions**: tests.comprehensive_scenario_testing
  - 定义数: 22

🟡 **large_file**: tests.integration_test
  - 行数: 820

🟡 **too_many_definitions**: tests.integration_test
  - 定义数: 26

🟡 **too_many_definitions**: tests.mock_data
  - 定义数: 24

🟡 **large_file**: tests.performance_test
  - 行数: 910

🟡 **too_many_definitions**: tests.performance_test
  - 定义数: 27

🔴 **large_file**: tests.real_world_validation
  - 行数: 1220

🟡 **too_many_definitions**: tests.real_world_validation
  - 定义数: 69

🔴 **large_file**: tests.standalone_scenario_validator
  - 行数: 1443

🟡 **too_many_definitions**: tests.standalone_scenario_validator
  - 定义数: 69

🟡 **too_many_definitions**: tests.test_aiot_compatibility
  - 定义数: 23

🟡 **large_file**: tests.test_gui_multimodal
  - 行数: 530

🟡 **large_file**: tests.test_headless_multimodal
  - 行数: 742

🟡 **large_file**: tests.test_multi_target_priority_system
  - 行数: 576

🟡 **large_file**: tests.test_optimized_multimodal
  - 行数: 649

🟡 **large_file**: tests.test_self_learning_system
  - 行数: 684

🔴 **large_file**: src.api.external_api_system
  - 行数: 1060

🔴 **too_many_imports**: src.api.external_api_system
  - 导入数: 31

🟡 **too_many_definitions**: src.api.external_api_system
  - 定义数: 41

🟡 **too_many_definitions**: src.core.base_plugin
  - 定义数: 30

🟡 **large_file**: src.core.config
  - 行数: 534

🟡 **too_many_definitions**: src.core.config
  - 定义数: 26

🟡 **large_file**: src.core.config_manager
  - 行数: 595

🟡 **too_many_definitions**: src.core.config_manager
  - 定义数: 29

🟡 **large_file**: src.core.cross_platform_manager
  - 行数: 695

🟡 **too_many_definitions**: src.core.cross_platform_manager
  - 定义数: 26

🟡 **large_file**: src.core.data_manager
  - 行数: 716

🟡 **too_many_definitions**: src.core.data_manager
  - 定义数: 44

🟡 **too_many_definitions**: src.core.engine
  - 定义数: 26

🟡 **too_many_definitions**: src.core.event_bus
  - 定义数: 26

🟡 **too_many_definitions**: src.core.logger
  - 定义数: 26

🟡 **large_file**: src.core.platform_adapter
  - 行数: 947

🟡 **too_many_imports**: src.core.platform_adapter
  - 导入数: 21

🟡 **too_many_definitions**: src.core.platform_adapter
  - 定义数: 54

🟡 **too_many_definitions**: src.core.storage_factory
  - 定义数: 26

🟡 **large_file**: src.gui.main_gui
  - 行数: 542

🟡 **too_many_definitions**: src.gui.main_gui
  - 定义数: 25

🟡 **large_file**: src.gui.yolos_training_gui
  - 行数: 840

🟡 **too_many_definitions**: src.gui.yolos_training_gui
  - 定义数: 73

🟡 **large_file**: src.models.pretrained_resources_manager
  - 行数: 663

🟡 **too_many_definitions**: src.models.pretrained_resources_manager
  - 定义数: 23

🟡 **large_file**: src.recognition.anti_spoofing_detector
  - 行数: 552

🟡 **too_many_definitions**: src.recognition.anti_spoofing_detector
  - 定义数: 21

🟡 **large_file**: src.recognition.emergency_response_system
  - 行数: 679

🟡 **too_many_definitions**: src.recognition.emergency_response_system
  - 定义数: 33

🟡 **large_file**: src.recognition.enhanced_gesture_recognizer
  - 行数: 635

🟡 **too_many_definitions**: src.recognition.enhanced_gesture_recognizer
  - 定义数: 24

🔴 **large_file**: src.recognition.enhanced_object_recognizer
  - 行数: 1084

🟡 **too_many_definitions**: src.recognition.enhanced_object_recognizer
  - 定义数: 56

🔴 **large_file**: src.recognition.enhanced_pet_recognizer
  - 行数: 1150

🟡 **too_many_definitions**: src.recognition.enhanced_pet_recognizer
  - 定义数: 51

🟡 **large_file**: src.recognition.enhanced_scene_analyzer
  - 行数: 916

🟡 **too_many_definitions**: src.recognition.enhanced_scene_analyzer
  - 定义数: 59

🟡 **large_file**: src.recognition.hybrid_recognition_system
  - 行数: 558

🟡 **too_many_definitions**: src.recognition.hybrid_recognition_system
  - 定义数: 25

🟡 **large_file**: src.recognition.image_quality_enhancer
  - 行数: 944

🟡 **too_many_definitions**: src.recognition.image_quality_enhancer
  - 定义数: 40

🟡 **large_file**: src.recognition.improved_multimodal_detector
  - 行数: 516

🟡 **large_file**: src.recognition.integrated_self_learning_recognition
  - 行数: 788

🟡 **too_many_definitions**: src.recognition.integrated_self_learning_recognition
  - 定义数: 28

🟡 **large_file**: src.recognition.intelligent_multi_target_system
  - 行数: 903

🟡 **too_many_definitions**: src.recognition.intelligent_multi_target_system
  - 定义数: 33

🟡 **large_file**: src.recognition.llm_self_learning_system
  - 行数: 998

🟡 **too_many_definitions**: src.recognition.llm_self_learning_system
  - 定义数: 42

🟡 **large_file**: src.recognition.medical_facial_analyzer
  - 行数: 874

🟡 **too_many_definitions**: src.recognition.medical_facial_analyzer
  - 定义数: 29

🟡 **large_file**: src.recognition.multimodal_detector
  - 行数: 507

🟡 **large_file**: src.recognition.optimized_face_recognizer
  - 行数: 681

🟡 **too_many_definitions**: src.recognition.optimized_face_recognizer
  - 定义数: 24

🔴 **large_file**: src.recognition.optimized_fall_detector
  - 行数: 1221

🟡 **too_many_definitions**: src.recognition.optimized_fall_detector
  - 定义数: 44

🔴 **large_file**: src.recognition.optimized_gesture_recognizer
  - 行数: 1086

🟡 **too_many_definitions**: src.recognition.optimized_gesture_recognizer
  - 定义数: 43

🔴 **large_file**: src.recognition.optimized_multimodal_detector
  - 行数: 1133

🟡 **too_many_imports**: src.recognition.optimized_multimodal_detector
  - 导入数: 22

🟡 **too_many_definitions**: src.recognition.optimized_multimodal_detector
  - 定义数: 34

🔴 **large_file**: src.recognition.optimized_pose_recognizer
  - 行数: 1667

🟡 **too_many_definitions**: src.recognition.optimized_pose_recognizer
  - 定义数: 48

🟡 **large_file**: src.recognition.pose_recognizer
  - 行数: 503

🟡 **too_many_definitions**: src.recognition.pose_recognizer
  - 定义数: 21

🟡 **large_file**: src.recognition.priority_recognition_system
  - 行数: 930

🟡 **too_many_definitions**: src.recognition.priority_recognition_system
  - 定义数: 53

🟡 **large_file**: src.recognition.usb_medical_camera_system
  - 行数: 827

🟡 **too_many_definitions**: src.recognition.usb_medical_camera_system
  - 定义数: 33

🟡 **large_file**: src.safety.medical_safety_manager
  - 行数: 539

🟡 **too_many_definitions**: src.safety.medical_safety_manager
  - 定义数: 33

🟡 **large_file**: src.sdk.yolos_client_sdk
  - 行数: 888

🟡 **too_many_definitions**: src.sdk.yolos_client_sdk
  - 定义数: 49

🟡 **large_file**: src.training.enhanced_human_trainer
  - 行数: 618

🟡 **too_many_definitions**: src.training.enhanced_human_trainer
  - 定义数: 23

🟡 **large_file**: src.training.offline_training_manager
  - 行数: 644

🟡 **large_file**: src.ui.accessibility_manager
  - 行数: 878

🟡 **too_many_definitions**: src.ui.accessibility_manager
  - 定义数: 39

🟡 **too_many_definitions**: src.utils.file_utils
  - 定义数: 25

🟡 **too_many_definitions**: src.utils.logging_manager
  - 定义数: 25

🟡 **large_file**: src.plugins.domain.dynamic_object_recognition
  - 行数: 676

🟡 **too_many_definitions**: src.plugins.domain.dynamic_object_recognition
  - 定义数: 38

🔴 **large_file**: src.plugins.domain.human_recognition
  - 行数: 1054

🟡 **too_many_definitions**: src.plugins.domain.human_recognition
  - 定义数: 38

🟡 **too_many_definitions**: src.plugins.domain.pet_recognition
  - 定义数: 34

🟡 **large_file**: src.plugins.domain.plant_recognition
  - 行数: 661

🟡 **too_many_definitions**: src.plugins.domain.plant_recognition
  - 定义数: 42

🟡 **large_file**: src.plugins.domain.static_object_recognition
  - 行数: 621

🟡 **too_many_definitions**: src.plugins.domain.static_object_recognition
  - 定义数: 36

🔴 **large_file**: src.plugins.platform.aiot_boards_adapter
  - 行数: 1228

🟡 **too_many_imports**: src.plugins.platform.aiot_boards_adapter
  - 导入数: 22

🟡 **too_many_definitions**: src.plugins.platform.aiot_boards_adapter
  - 定义数: 29

🟡 **large_file**: src.plugins.platform.arduino_adapter
  - 行数: 858

🟡 **too_many_definitions**: src.plugins.platform.arduino_adapter
  - 定义数: 31

🟡 **too_many_definitions**: src.plugins.platform.esp32_plugin
  - 定义数: 35

🟡 **large_file**: src.plugins.platform.raspberry_pi_adapter
  - 行数: 569

🟡 **too_many_definitions**: src.plugins.platform.raspberry_pi_adapter
  - 定义数: 24

🟡 **too_many_definitions**: src.plugins.platform.raspberry_pi_plugin
  - 定义数: 38

🟡 **too_many_imports**: src.plugins.platform.ros_integration
  - 导入数: 22

🟡 **too_many_definitions**: src.plugins.platform.ros_integration
  - 定义数: 21


## 📋 模块详情

| 模块 | 行数 | 导入 | 类 | 函数 |
|------|------|------|----|----- |
| basic_pet_recognition_gui | 657 | 9 | 2 | 20 |
| setup | 51 | 1 | 0 | 0 |
| examples.api_usage_examples | 537 | 6 | 0 | 8 |
| examples.basic_detection | 99 | 7 | 0 | 1 |
| examples.mqtt_detection | 162 | 6 | 1 | 8 |
| examples.realtime_detection | 70 | 4 | 0 | 2 |
| examples.windows_multimodal_debug | 344 | 8 | 0 | 4 |
| scripts.analyze_dependencies | 414 | 8 | 1 | 17 |
| scripts.download_training_datasets | 438 | 12 | 1 | 12 |
| scripts.install | 567 | 8 | 1 | 22 |
| scripts.quick_start_enhanced_training | 324 | 10 | 1 | 9 |
| scripts.setup_hybrid_system | 435 | 10 | 1 | 8 |
| scripts.setup_mirrors | 71 | 2 | 0 | 3 |
| scripts.train_offline_models | 116 | 7 | 0 | 1 |
| src.__init__ | 12 | 4 | 0 | 0 |
| tests.base_test | 357 | 17 | 2 | 34 |
| tests.comprehensive_scenario_testing | 752 | 15 | 1 | 21 |
| tests.integration_test | 820 | 12 | 6 | 20 |
| tests.mock_data | 485 | 8 | 4 | 20 |
| tests.performance_test | 910 | 13 | 8 | 19 |
| tests.real_world_validation | 1220 | 9 | 1 | 68 |
| tests.simple_multimodal_gui_test | 384 | 12 | 1 | 11 |
| tests.standalone_scenario_validator | 1443 | 7 | 1 | 68 |
| tests.test_aiot_compatibility | 441 | 7 | 2 | 21 |
| tests.test_aiot_simple | 270 | 9 | 0 | 5 |
| tests.test_aiot_standalone | 435 | 10 | 0 | 6 |
| tests.test_config | 420 | 7 | 4 | 14 |
| tests.test_gui_multimodal | 530 | 16 | 1 | 12 |
| tests.test_headless_multimodal | 742 | 13 | 1 | 12 |
| tests.test_modular_aiot_platform | 442 | 13 | 0 | 8 |
| tests.test_multi_target_priority_system | 576 | 11 | 1 | 10 |
| tests.test_opencv_display | 157 | 4 | 0 | 3 |
| tests.test_optimized_multimodal | 649 | 13 | 1 | 13 |
| tests.test_quality_and_spoofing | 445 | 9 | 0 | 7 |
| tests.test_self_learning_system | 684 | 16 | 1 | 16 |
| tests.test_usb_medical_camera | 403 | 10 | 0 | 12 |
| tests.__init__ | 24 | 5 | 0 | 0 |
| web.app | 436 | 15 | 0 | 9 |
| esp32.yolos_esp32_cam.esp32_recognition_adapter | 416 | 9 | 1 | 13 |
| ros_workspace.src.yolos_ros.scripts.detection_node | 207 | 17 | 1 | 10 |
| src.api.external_api_system | 1060 | 31 | 7 | 34 |
| src.communication.external_communication_system | 126 | 6 | 3 | 2 |
| src.communication.mqtt_client | 274 | 6 | 1 | 16 |
| src.communication.__init__ | 15 | 4 | 0 | 0 |
| src.core.base_plugin | 284 | 4 | 7 | 23 |
| src.core.config | 534 | 7 | 9 | 17 |
| src.core.config_manager | 595 | 14 | 3 | 26 |
| src.core.cross_platform_manager | 695 | 19 | 1 | 25 |
| src.core.data_manager | 716 | 15 | 6 | 38 |
| src.core.engine | 430 | 15 | 5 | 21 |
| src.core.event_bus | 415 | 10 | 3 | 23 |
| src.core.integrated_aiot_platform | 147 | 5 | 3 | 1 |
| src.core.logger | 376 | 12 | 1 | 25 |
| src.core.modular_extension_manager | 129 | 5 | 3 | 7 |
| src.core.platform_adapter | 947 | 21 | 9 | 45 |
| src.core.plugin_manager | 471 | 15 | 3 | 16 |
| src.core.storage_factory | 446 | 11 | 4 | 22 |
| src.core.unified_config_manager | 380 | 9 | 3 | 17 |
| src.core.__init__ | 78 | 8 | 0 | 0 |
| src.detection.camera_detector | 283 | 11 | 1 | 11 |
| src.detection.cli | 176 | 7 | 0 | 4 |
| src.detection.image_detector | 196 | 6 | 1 | 7 |
| src.detection.realtime_detector | 210 | 7 | 1 | 9 |
| src.detection.video_detector | 274 | 7 | 1 | 6 |
| src.detection.__init__ | 15 | 4 | 0 | 0 |
| src.gui.main_gui | 542 | 13 | 3 | 22 |
| src.gui.yolos_training_gui | 840 | 19 | 7 | 66 |
| src.models.base_model | 133 | 6 | 1 | 10 |
| src.models.model_converter | 476 | 12 | 1 | 11 |
| src.models.pretrained_model_loader | 420 | 12 | 2 | 12 |
| src.models.pretrained_resources_manager | 663 | 14 | 4 | 19 |
| src.models.yolov5_model | 130 | 6 | 1 | 8 |
| src.models.yolov8_model | 150 | 7 | 1 | 9 |
| src.models.yolo_factory | 70 | 6 | 1 | 3 |
| src.models.yolo_world_model | 121 | 5 | 1 | 8 |
| src.models.__init__ | 17 | 5 | 0 | 0 |
| src.plugins.__init__ | 24 | 3 | 0 | 0 |
| src.recognition.anti_spoofing_detector | 552 | 6 | 3 | 18 |
| src.recognition.emergency_response_system | 679 | 13 | 5 | 28 |
| src.recognition.enhanced_face_recognizer | 455 | 10 | 1 | 15 |
| src.recognition.enhanced_fall_detection_system | 196 | 7 | 4 | 13 |
| src.recognition.enhanced_gesture_recognizer | 635 | 10 | 1 | 23 |
| src.recognition.enhanced_object_recognizer | 1084 | 9 | 9 | 47 |
| src.recognition.enhanced_pet_recognizer | 1150 | 15 | 13 | 38 |
| src.recognition.enhanced_scene_analyzer | 916 | 6 | 8 | 51 |
| src.recognition.face_recognizer | 307 | 8 | 1 | 12 |
| src.recognition.fall_detector | 435 | 9 | 1 | 14 |
| src.recognition.gesture_recognizer | 230 | 5 | 1 | 11 |
| src.recognition.hybrid_recognition_system | 558 | 17 | 4 | 21 |
| src.recognition.image_quality_enhancer | 944 | 10 | 6 | 34 |
| src.recognition.improved_multimodal_detector | 516 | 16 | 4 | 13 |
| src.recognition.integrated_self_learning_recognition | 788 | 20 | 4 | 24 |
| src.recognition.intelligent_multi_target_system | 903 | 18 | 5 | 28 |
| src.recognition.intelligent_recognition_system | 312 | 7 | 3 | 11 |
| src.recognition.llm_self_learning_system | 998 | 12 | 5 | 37 |
| src.recognition.medical_facial_analyzer | 874 | 8 | 5 | 24 |
| src.recognition.medication_recognition_system | 97 | 5 | 2 | 7 |
| src.recognition.multimodal_detector | 507 | 13 | 1 | 16 |
| src.recognition.object_recognizer | 398 | 7 | 3 | 15 |
| src.recognition.optimized_face_recognizer | 681 | 18 | 2 | 22 |
| src.recognition.optimized_fall_detector | 1221 | 15 | 4 | 40 |
| src.recognition.optimized_gesture_recognizer | 1086 | 15 | 3 | 40 |
| src.recognition.optimized_multimodal_detector | 1133 | 22 | 4 | 30 |
| src.recognition.optimized_pose_recognizer | 1667 | 16 | 3 | 45 |
| src.recognition.pose_recognizer | 503 | 5 | 1 | 20 |
| src.recognition.priority_recognition_system | 930 | 10 | 7 | 46 |
| src.recognition.usb_medical_camera_system | 827 | 20 | 3 | 30 |
| src.recognition.yolov7_pose_recognizer | 448 | 7 | 1 | 16 |
| src.recognition.__init__ | 15 | 4 | 0 | 0 |
| src.safety.medical_safety_manager | 539 | 6 | 5 | 28 |
| src.sdk.yolos_client_sdk | 888 | 20 | 6 | 43 |
| src.training.augmentation | 40 | 2 | 2 | 6 |
| src.training.dataset_manager | 33 | 4 | 1 | 3 |
| src.training.enhanced_human_trainer | 618 | 14 | 5 | 18 |
| src.training.offline_training_manager | 644 | 15 | 3 | 17 |
| src.training.trainer | 294 | 8 | 1 | 13 |
| src.training.__init__ | 13 | 3 | 0 | 0 |
| src.ui.accessibility_manager | 878 | 9 | 6 | 33 |
| src.utils.config_manager | 331 | 4 | 1 | 16 |
| src.utils.file_utils | 236 | 8 | 3 | 22 |
| src.utils.logger | 29 | 3 | 0 | 2 |
| src.utils.logging_manager | 328 | 12 | 1 | 24 |
| src.utils.metrics | 119 | 4 | 3 | 11 |
| src.utils.visualization | 122 | 3 | 2 | 9 |
| src.utils.__init__ | 17 | 5 | 0 | 0 |
| src.plugins.domain.dynamic_object_recognition | 676 | 10 | 14 | 24 |
| src.plugins.domain.human_recognition | 1054 | 13 | 7 | 31 |
| src.plugins.domain.pet_recognition | 472 | 8 | 14 | 20 |
| src.plugins.domain.plant_recognition | 661 | 9 | 14 | 28 |
| src.plugins.domain.static_object_recognition | 621 | 8 | 11 | 25 |
| src.plugins.domain.__init__ | 15 | 5 | 0 | 0 |
| src.plugins.hardware.robotic_arm_controller | 140 | 5 | 3 | 4 |
| src.plugins.platform.aiot_boards_adapter | 1228 | 22 | 1 | 28 |
| src.plugins.platform.arduino_adapter | 858 | 13 | 7 | 24 |
| src.plugins.platform.esp32_plugin | 381 | 6 | 9 | 26 |
| src.plugins.platform.raspberry_pi_adapter | 569 | 15 | 1 | 23 |
| src.plugins.platform.raspberry_pi_plugin | 442 | 6 | 9 | 29 |
| src.plugins.platform.ros_integration | 435 | 22 | 3 | 18 |
| src.plugins.platform.__init__ | 26 | 6 | 0 | 0 |
| src.plugins.utility.__init__ | 26 | 6 | 0 | 0 |

## 🎯 优化建议

### 高优先级
1. **解决循环依赖**: 重构模块间的依赖关系
2. **合并重复模块**: 消除功能重复的模块
3. **拆分大文件**: 将超过500行的文件进行模块化拆分

### 中优先级
1. **减少导入数量**: 优化模块间的耦合度
2. **统一接口设计**: 建立标准化的模块接口
3. **代码重构**: 提高代码的内聚性

### 低优先级
1. **性能优化**: 基于实际使用情况优化性能
2. **文档完善**: 补充模块间的依赖关系文档

---

*报告生成时间: 2025-09-09 09:56:04*
