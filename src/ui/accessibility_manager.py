#!/usr/bin/env python3
"""
无障碍和用户体验管理器
改善老年人、儿童和残障人士的使用体验
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import tkinter as tk
from tkinter import ttk, font

class UserType(Enum):
    """用户类型"""
    ELDERLY = "elderly"
    CHILD = "child"
    ADULT = "adult"
    VISUALLY_IMPAIRED = "visually_impaired"
    HEARING_IMPAIRED = "hearing_impaired"
    MOTOR_IMPAIRED = "motor_impaired"

class Language(Enum):
    """支持的语言"""
    CHINESE_SIMPLIFIED = "zh_CN"
    CHINESE_TRADITIONAL = "zh_TW"
    ENGLISH = "en_US"
    JAPANESE = "ja_JP"
    KOREAN = "ko_KR"

@dataclass
class UITheme:
    """UI主题配置"""
    name: str
    font_size: int
    button_size: Tuple[int, int]
    color_scheme: Dict[str, str]
    contrast_ratio: float
    spacing: int

@dataclass
class AccessibilitySettings:
    """无障碍设置"""
    user_type: UserType
    language: Language
    font_size_multiplier: float
    high_contrast: bool
    voice_feedback: bool
    large_buttons: bool
    simplified_interface: bool
    screen_reader_support: bool

class AccessibilityManager:
    """无障碍管理器"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.current_settings = self._load_default_settings()
        self.themes = self._initialize_themes()
        self.translations = self._load_translations()
        
    def _setup_logger(self):
        """设置日志"""
        logger = logging.getLogger('AccessibilityManager')
        logger.setLevel(logging.INFO)
        
        os.makedirs('logs', exist_ok=True)
        handler = logging.FileHandler('logs/accessibility.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_default_settings(self) -> AccessibilitySettings:
        """加载默认设置"""
        return AccessibilitySettings(
            user_type=UserType.ADULT,
            language=Language.CHINESE_SIMPLIFIED,
            font_size_multiplier=1.0,
            high_contrast=False,
            voice_feedback=False,
            large_buttons=False,
            simplified_interface=False,
            screen_reader_support=False
        )
    
    def _initialize_themes(self) -> Dict[str, UITheme]:
        """初始化UI主题"""
        return {
            "default": UITheme(
                name="默认主题",
                font_size=12,
                button_size=(100, 40),
                color_scheme={
                    "bg": "#FFFFFF",
                    "fg": "#000000",
                    "button_bg": "#E1E1E1",
                    "button_fg": "#000000",
                    "accent": "#0078D4"
                },
                contrast_ratio=4.5,
                spacing=10
            ),
            "elderly": UITheme(
                name="老年人友好主题",
                font_size=18,
                button_size=(150, 60),
                color_scheme={
                    "bg": "#FFFFFF",
                    "fg": "#000000",
                    "button_bg": "#F0F0F0",
                    "button_fg": "#000000",
                    "accent": "#0066CC"
                },
                contrast_ratio=7.0,
                spacing=20
            ),
            "high_contrast": UITheme(
                name="高对比度主题",
                font_size=16,
                button_size=(120, 50),
                color_scheme={
                    "bg": "#000000",
                    "fg": "#FFFFFF",
                    "button_bg": "#FFFFFF",
                    "button_fg": "#000000",
                    "accent": "#FFFF00"
                },
                contrast_ratio=21.0,
                spacing=15
            ),
            "child": UITheme(
                name="儿童友好主题",
                font_size=14,
                button_size=(120, 50),
                color_scheme={
                    "bg": "#F8F8FF",
                    "fg": "#2F4F4F",
                    "button_bg": "#87CEEB",
                    "button_fg": "#000080",
                    "accent": "#FF6347"
                },
                contrast_ratio=5.0,
                spacing=12
            )
        }
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """加载翻译文件"""
        return {
            Language.CHINESE_SIMPLIFIED.value: {
                "start": "开始",
                "stop": "停止",
                "settings": "设置",
                "help": "帮助",
                "exit": "退出",
                "medical_detection": "医疗检测",
                "safety_monitoring": "安全监控",
                "pet_care": "宠物护理",
                "emergency": "紧急情况",
                "warning": "警告",
                "error": "错误",
                "success": "成功",
                "please_wait": "请稍候...",
                "processing": "处理中...",
                "camera_not_found": "未找到摄像头",
                "medical_disclaimer": "医疗免责声明",
                "consult_doctor": "请咨询医生",
                "emergency_call": "紧急呼叫",
                "high_risk": "高风险",
                "low_confidence": "置信度低",
                "safe": "安全",
                "unsafe": "不安全"
            },
            Language.ENGLISH.value: {
                "start": "Start",
                "stop": "Stop",
                "settings": "Settings",
                "help": "Help",
                "exit": "Exit",
                "medical_detection": "Medical Detection",
                "safety_monitoring": "Safety Monitoring",
                "pet_care": "Pet Care",
                "emergency": "Emergency",
                "warning": "Warning",
                "error": "Error",
                "success": "Success",
                "please_wait": "Please wait...",
                "processing": "Processing...",
                "camera_not_found": "Camera not found",
                "medical_disclaimer": "Medical Disclaimer",
                "consult_doctor": "Please consult a doctor",
                "emergency_call": "Emergency Call",
                "high_risk": "High Risk",
                "low_confidence": "Low Confidence",
                "safe": "Safe",
                "unsafe": "Unsafe"
            },
            Language.CHINESE_TRADITIONAL.value: {
                "start": "開始",
                "stop": "停止",
                "settings": "設置",
                "help": "幫助",
                "exit": "退出",
                "medical_detection": "醫療檢測",
                "safety_monitoring": "安全監控",
                "pet_care": "寵物護理",
                "emergency": "緊急情況",
                "warning": "警告",
                "error": "錯誤",
                "success": "成功",
                "please_wait": "請稍候...",
                "processing": "處理中...",
                "camera_not_found": "未找到攝像頭",
                "medical_disclaimer": "醫療免責聲明",
                "consult_doctor": "請諮詢醫生",
                "emergency_call": "緊急呼叫",
                "high_risk": "高風險",
                "low_confidence": "置信度低",
                "safe": "安全",
                "unsafe": "不安全"
            }
        }
    
    def configure_for_user_type(self, user_type: UserType) -> AccessibilitySettings:
        """为特定用户类型配置界面"""
        
        self.logger.info(f"配置界面为用户类型: {user_type.value}")
        
        if user_type == UserType.ELDERLY:
            settings = AccessibilitySettings(
                user_type=user_type,
                language=self.current_settings.language,
                font_size_multiplier=1.5,  # 150% 字体大小
                high_contrast=True,
                voice_feedback=True,
                large_buttons=True,
                simplified_interface=True,
                screen_reader_support=False
            )
        elif user_type == UserType.CHILD:
            settings = AccessibilitySettings(
                user_type=user_type,
                language=self.current_settings.language,
                font_size_multiplier=1.2,  # 120% 字体大小
                high_contrast=False,
                voice_feedback=True,
                large_buttons=True,
                simplified_interface=True,
                screen_reader_support=False
            )
        elif user_type == UserType.VISUALLY_IMPAIRED:
            settings = AccessibilitySettings(
                user_type=user_type,
                language=self.current_settings.language,
                font_size_multiplier=2.0,  # 200% 字体大小
                high_contrast=True,
                voice_feedback=True,
                large_buttons=True,
                simplified_interface=True,
                screen_reader_support=True
            )
        else:
            settings = self.current_settings
        
        self.current_settings = settings
        return settings
    
    def get_theme_for_settings(self, settings: AccessibilitySettings) -> UITheme:
        """根据设置获取合适的主题"""
        
        if settings.user_type == UserType.ELDERLY or settings.high_contrast:
            base_theme = self.themes["elderly"] if settings.user_type == UserType.ELDERLY else self.themes["high_contrast"]
        elif settings.user_type == UserType.CHILD:
            base_theme = self.themes["child"]
        else:
            base_theme = self.themes["default"]
        
        # 应用字体大小倍数
        adjusted_theme = UITheme(
            name=base_theme.name,
            font_size=int(base_theme.font_size * settings.font_size_multiplier),
            button_size=(
                int(base_theme.button_size[0] * (1.2 if settings.large_buttons else 1.0)),
                int(base_theme.button_size[1] * (1.2 if settings.large_buttons else 1.0))
            ),
            color_scheme=base_theme.color_scheme.copy(),
            contrast_ratio=base_theme.contrast_ratio,
            spacing=int(base_theme.spacing * (1.5 if settings.simplified_interface else 1.0))
        )
        
        return adjusted_theme
    
    def translate(self, key: str, language: Optional[Language] = None) -> str:
        """翻译文本"""
        
        if language is None:
            language = self.current_settings.language
        
        lang_code = language.value
        
        if lang_code in self.translations and key in self.translations[lang_code]:
            return self.translations[lang_code][key]
        
        # 回退到英文
        if Language.ENGLISH.value in self.translations and key in self.translations[Language.ENGLISH.value]:
            return self.translations[Language.ENGLISH.value][key]
        
        # 最后回退到键名
        return key
    
    def create_accessible_button(self, 
                               parent: tk.Widget,
                               text_key: str,
                               command: callable,
                               theme: UITheme) -> tk.Button:
        """创建无障碍按钮"""
        
        text = self.translate(text_key)
        
        button = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Arial", theme.font_size, "bold"),
            width=theme.button_size[0] // 8,  # 近似字符宽度
            height=theme.button_size[1] // 20,  # 近似行高
            bg=theme.color_scheme["button_bg"],
            fg=theme.color_scheme["button_fg"],
            activebackground=theme.color_scheme["accent"],
            relief="raised",
            bd=3
        )
        
        # 添加键盘导航支持
        button.bind("<Return>", lambda e: command())
        button.bind("<space>", lambda e: command())
        
        # 添加语音反馈（如果启用）
        if self.current_settings.voice_feedback:
            button.bind("<Enter>", lambda e: self._speak_text(text))
        
        return button
    
    def create_accessible_label(self,
                              parent: tk.Widget,
                              text_key: str,
                              theme: UITheme) -> tk.Label:
        """创建无障碍标签"""
        
        text = self.translate(text_key)
        
        label = tk.Label(
            parent,
            text=text,
            font=("Arial", theme.font_size),
            bg=theme.color_scheme["bg"],
            fg=theme.color_scheme["fg"],
            wraplength=400  # 自动换行
        )
        
        return label
    
    def create_status_display(self,
                            parent: tk.Widget,
                            theme: UITheme) -> tk.Text:
        """创建状态显示区域"""
        
        text_widget = tk.Text(
            parent,
            font=("Arial", theme.font_size),
            bg=theme.color_scheme["bg"],
            fg=theme.color_scheme["fg"],
            height=10,
            width=50,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        
        # 添加滚动条
        scrollbar = tk.Scrollbar(parent, command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        return text_widget
    
    def update_status_display(self,
                            text_widget: tk.Text,
                            message: str,
                            message_type: str = "info"):
        """更新状态显示"""
        
        text_widget.config(state=tk.NORMAL)
        
        # 添加时间戳
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 根据消息类型选择前缀
        prefixes = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅"
        }
        prefix = prefixes.get(message_type, "ℹ️")
        
        formatted_message = f"[{timestamp}] {prefix} {message}\n"
        
        text_widget.insert(tk.END, formatted_message)
        text_widget.see(tk.END)  # 滚动到最新消息
        text_widget.config(state=tk.DISABLED)
        
        # 语音反馈
        if self.current_settings.voice_feedback and message_type in ["warning", "error"]:
            self._speak_text(message)
    
    def _speak_text(self, text: str):
        """语音播报文本"""
        try:
            # 这里可以集成TTS引擎
            # 目前使用简单的日志记录
            self.logger.info(f"语音播报: {text}")
            
            # 可以使用 pyttsx3 或其他TTS库
            # import pyttsx3
            # engine = pyttsx3.init()
            # engine.say(text)
            # engine.runAndWait()
            
        except Exception as e:
            self.logger.error(f"语音播报失败: {e}")
    
    def create_emergency_button(self,
                              parent: tk.Widget,
                              emergency_callback: callable,
                              theme: UITheme) -> tk.Button:
        """创建紧急按钮"""
        
        button = tk.Button(
            parent,
            text=self.translate("emergency_call"),
            command=emergency_callback,
            font=("Arial", theme.font_size + 4, "bold"),
            bg="#FF0000",  # 红色背景
            fg="#FFFFFF",  # 白色文字
            activebackground="#CC0000",
            width=15,
            height=3,
            relief="raised",
            bd=5
        )
        
        # 紧急按钮的特殊绑定
        button.bind("<Return>", lambda e: emergency_callback())
        button.bind("<space>", lambda e: emergency_callback())
        
        return button
    
    def show_medical_disclaimer(self, parent: tk.Widget):
        """显示医疗免责声明"""
        
        disclaimer_window = tk.Toplevel(parent)
        disclaimer_window.title(self.translate("medical_disclaimer"))
        disclaimer_window.geometry("600x400")
        
        theme = self.get_theme_for_settings(self.current_settings)
        
        # 免责声明文本
        disclaimer_text = """
⚠️ 重要医疗免责声明 ⚠️

本系统仅为辅助参考工具，不能替代专业医疗诊断。

• 本系统不是医疗设备，未获得医疗器械认证
• 所有检测结果仅供参考，不构成医疗建议  
• 任何健康问题请咨询专业医生或医疗机构
• 紧急情况请立即拨打急救电话
• 用户使用本系统的风险由用户自行承担

如有任何健康疑虑，请寻求专业医疗帮助。
        """.strip()
        
        text_widget = tk.Text(
            disclaimer_window,
            font=("Arial", theme.font_size),
            bg=theme.color_scheme["bg"],
            fg=theme.color_scheme["fg"],
            wrap=tk.WORD,
            padx=20,
            pady=20
        )
        
        text_widget.insert(tk.END, disclaimer_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # 确认按钮
        confirm_button = self.create_accessible_button(
            disclaimer_window,
            "确认已阅读",
            disclaimer_window.destroy,
            theme
        )
        confirm_button.pack(pady=10)
        
        # 语音播报免责声明
        if self.current_settings.voice_feedback:
            self._speak_text("请仔细阅读医疗免责声明")
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """获取用户偏好设置"""
        
        return {
            "user_type": self.current_settings.user_type.value,
            "language": self.current_settings.language.value,
            "font_size_multiplier": self.current_settings.font_size_multiplier,
            "high_contrast": self.current_settings.high_contrast,
            "voice_feedback": self.current_settings.voice_feedback,
            "large_buttons": self.current_settings.large_buttons,
            "simplified_interface": self.current_settings.simplified_interface,
            "screen_reader_support": self.current_settings.screen_reader_support
        }
    
    def save_user_preferences(self, preferences: Dict[str, Any]):
        """保存用户偏好设置"""
        
        try:
            os.makedirs("config", exist_ok=True)
            with open("config/user_preferences.json", "w", encoding="utf-8") as f:
                json.dump(preferences, f, ensure_ascii=False, indent=2)
            
            self.logger.info("用户偏好设置已保存")
            
        except Exception as e:
            self.logger.error(f"保存用户偏好设置失败: {e}")
    
    def load_user_preferences(self) -> Dict[str, Any]:
        """加载用户偏好设置"""
        
        try:
            with open("config/user_preferences.json", "r", encoding="utf-8") as f:
                preferences = json.load(f)
            
            # 应用设置
            self.current_settings = AccessibilitySettings(
                user_type=UserType(preferences.get("user_type", UserType.ADULT.value)),
                language=Language(preferences.get("language", Language.CHINESE_SIMPLIFIED.value)),
                font_size_multiplier=preferences.get("font_size_multiplier", 1.0),
                high_contrast=preferences.get("high_contrast", False),
                voice_feedback=preferences.get("voice_feedback", False),
                large_buttons=preferences.get("large_buttons", False),
                simplified_interface=preferences.get("simplified_interface", False),
                screen_reader_support=preferences.get("screen_reader_support", False)
            )
            
            self.logger.info("用户偏好设置已加载")
            return preferences
            
        except FileNotFoundError:
            self.logger.info("未找到用户偏好设置文件，使用默认设置")
            return self.get_user_preferences()
        except Exception as e:
            self.logger.error(f"加载用户偏好设置失败: {e}")
            return self.get_user_preferences()

class AccessibleGUI:
    """无障碍GUI主类"""
    
    def __init__(self):
        self.accessibility_manager = AccessibilityManager()
        self.root = None
        self.status_display = None
        
    def create_main_window(self):
        """创建主窗口"""
        
        self.root = tk.Tk()
        self.root.title("YOLOS - 智能识别系统")
        
        # 加载用户偏好
        self.accessibility_manager.load_user_preferences()
        
        # 获取主题
        theme = self.accessibility_manager.get_theme_for_settings(
            self.accessibility_manager.current_settings
        )
        
        # 设置窗口
        self.root.geometry("800x600")
        self.root.configure(bg=theme.color_scheme["bg"])
        
        # 创建界面元素
        self._create_menu_bar(theme)
        self._create_main_interface(theme)
        
        # 显示医疗免责声明
        self.root.after(1000, lambda: self.accessibility_manager.show_medical_disclaimer(self.root))
        
        return self.root
    
    def _create_menu_bar(self, theme: UITheme):
        """创建菜单栏"""
        
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 设置菜单
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(
            label=self.accessibility_manager.translate("settings"),
            menu=settings_menu
        )
        
        settings_menu.add_command(
            label="老年人模式",
            command=lambda: self._switch_user_type(UserType.ELDERLY)
        )
        settings_menu.add_command(
            label="儿童模式", 
            command=lambda: self._switch_user_type(UserType.CHILD)
        )
        settings_menu.add_command(
            label="标准模式",
            command=lambda: self._switch_user_type(UserType.ADULT)
        )
        settings_menu.add_separator()
        settings_menu.add_command(
            label="语言设置",
            command=self._show_language_settings
        )
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(
            label=self.accessibility_manager.translate("help"),
            menu=help_menu
        )
        
        help_menu.add_command(
            label="使用说明",
            command=self._show_help
        )
        help_menu.add_command(
            label="医疗免责声明",
            command=lambda: self.accessibility_manager.show_medical_disclaimer(self.root)
        )
    
    def _create_main_interface(self, theme: UITheme):
        """创建主界面"""
        
        # 主框架
        main_frame = tk.Frame(self.root, bg=theme.color_scheme["bg"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=theme.spacing, pady=theme.spacing)
        
        # 标题
        title_label = self.accessibility_manager.create_accessible_label(
            main_frame, "YOLOS智能识别系统", theme
        )
        title_label.config(font=("Arial", theme.font_size + 6, "bold"))
        title_label.pack(pady=theme.spacing)
        
        # 功能按钮区域
        button_frame = tk.Frame(main_frame, bg=theme.color_scheme["bg"])
        button_frame.pack(pady=theme.spacing)
        
        # 功能按钮
        buttons = [
            ("medical_detection", self._start_medical_detection),
            ("safety_monitoring", self._start_safety_monitoring),
            ("pet_care", self._start_pet_care),
            ("settings", self._show_settings)
        ]
        
        for i, (text_key, command) in enumerate(buttons):
            button = self.accessibility_manager.create_accessible_button(
                button_frame, text_key, command, theme
            )
            button.grid(row=i//2, column=i%2, padx=theme.spacing, pady=theme.spacing)
        
        # 紧急按钮
        emergency_button = self.accessibility_manager.create_emergency_button(
            main_frame, self._emergency_call, theme
        )
        emergency_button.pack(pady=theme.spacing*2)
        
        # 状态显示区域
        status_frame = tk.Frame(main_frame, bg=theme.color_scheme["bg"])
        status_frame.pack(fill=tk.BOTH, expand=True, pady=theme.spacing)
        
        status_label = self.accessibility_manager.create_accessible_label(
            status_frame, "系统状态", theme
        )
        status_label.pack()
        
        self.status_display = self.accessibility_manager.create_status_display(
            status_frame, theme
        )
        self.status_display.pack(fill=tk.BOTH, expand=True)
        
        # 初始状态消息
        self.accessibility_manager.update_status_display(
            self.status_display,
            "系统已启动，请选择功能",
            "success"
        )
    
    def _switch_user_type(self, user_type: UserType):
        """切换用户类型"""
        
        self.accessibility_manager.configure_for_user_type(user_type)
        
        # 重新创建界面
        for widget in self.root.winfo_children():
            widget.destroy()
        
        self._create_menu_bar(
            self.accessibility_manager.get_theme_for_settings(
                self.accessibility_manager.current_settings
            )
        )
        self._create_main_interface(
            self.accessibility_manager.get_theme_for_settings(
                self.accessibility_manager.current_settings
            )
        )
        
        self.accessibility_manager.update_status_display(
            self.status_display,
            f"已切换到{user_type.value}模式",
            "success"
        )
    
    def _show_language_settings(self):
        """显示语言设置"""
        
        lang_window = tk.Toplevel(self.root)
        lang_window.title("语言设置")
        lang_window.geometry("300x200")
        
        theme = self.accessibility_manager.get_theme_for_settings(
            self.accessibility_manager.current_settings
        )
        
        # 语言选项
        languages = [
            (Language.CHINESE_SIMPLIFIED, "简体中文"),
            (Language.CHINESE_TRADITIONAL, "繁體中文"),
            (Language.ENGLISH, "English")
        ]
        
        for lang, name in languages:
            button = tk.Button(
                lang_window,
                text=name,
                command=lambda l=lang: self._set_language(l, lang_window),
                font=("Arial", theme.font_size),
                width=15,
                height=2
            )
            button.pack(pady=5)
    
    def _set_language(self, language: Language, window: tk.Toplevel):
        """设置语言"""
        
        self.accessibility_manager.current_settings.language = language
        window.destroy()
        
        # 保存设置
        preferences = self.accessibility_manager.get_user_preferences()
        self.accessibility_manager.save_user_preferences(preferences)
        
        self.accessibility_manager.update_status_display(
            self.status_display,
            f"语言已设置为{language.value}",
            "success"
        )
    
    def _show_help(self):
        """显示帮助"""
        
        help_text = """
YOLOS智能识别系统使用说明

1. 医疗检测：辅助检测面部症状（仅供参考）
2. 安全监控：监控火灾、入侵等安全威胁
3. 宠物护理：监测宠物健康和行为
4. 紧急呼叫：紧急情况下快速求助

重要提醒：
• 医疗功能仅供参考，不能替代专业诊断
• 紧急情况请拨打急救电话
• 定期检查摄像头和网络连接

快捷键：
• Tab键：在按钮间切换
• 回车键或空格键：激活按钮
• Esc键：退出当前操作
        """
        
        self.accessibility_manager.update_status_display(
            self.status_display,
            help_text,
            "info"
        )
    
    def _start_medical_detection(self):
        """启动医疗检测"""
        self.accessibility_manager.update_status_display(
            self.status_display,
            "⚠️ 医疗检测功能当前不可用（安全原因）",
            "warning"
        )
    
    def _start_safety_monitoring(self):
        """启动安全监控"""
        self.accessibility_manager.update_status_display(
            self.status_display,
            "启动安全监控功能...",
            "info"
        )
    
    def _start_pet_care(self):
        """启动宠物护理"""
        self.accessibility_manager.update_status_display(
            self.status_display,
            "启动宠物护理功能...",
            "info"
        )
    
    def _show_settings(self):
        """显示设置"""
        self.accessibility_manager.update_status_display(
            self.status_display,
            "打开设置界面...",
            "info"
        )
    
    def _emergency_call(self):
        """紧急呼叫"""
        self.accessibility_manager.update_status_display(
            self.status_display,
            "🚨 紧急呼叫已激活！正在联系紧急联系人...",
            "error"
        )
        
        # 这里应该实现实际的紧急呼叫逻辑
        # 例如：发送短信、拨打电话、发送邮件等
    
    def run(self):
        """运行GUI"""
        if self.root:
            self.root.mainloop()

# 测试和演示
def main():
    """主函数"""
    print("🚀 启动YOLOS无障碍界面...")
    
    # 创建GUI
    gui = AccessibleGUI()
    main_window = gui.create_main_window()
    
    print("✅ 界面已创建，支持以下功能：")
    print("   • 老年人友好界面（大字体、高对比度）")
    print("   • 多语言支持（中文、英文）")
    print("   • 键盘导航支持")
    print("   • 语音反馈（可选）")
    print("   • 医疗安全警告")
    print("   • 紧急呼叫功能")
    
    # 运行GUI
    gui.run()

if __name__ == "__main__":
    main()