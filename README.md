# 调色灵感专家 (AI-LUT)

**AI-LUT** 是一个基于人工智能的色彩分级（Color Grading）辅助工具。它能够深度分析参考图像的色彩、光影和构图特征，并利用先进的生成式 AI 模型（如 Google Gemini、豆包）生成高精度的 3D LUT（查找表），帮助摄影师和调色师快速获得电影级的调色灵感。

![应用截图](docs/screenshot_demo.png)
*(请在此处添加应用运行截图，如 `docs/screenshot_demo.png`)*

## ✨ 主要功能

- **AI 深度视觉分析**：自动识别画面的物理结构、光影逻辑、色温倾向及动态范围。
- **智能 LUT 生成**：基于分析结果，生成匹配目标风格的 `.cube` 格式 3D LUT 文件。
- **多模型支持**：支持 Google Gemini Pro/Flash 及字节跳动豆包（Doubao）模型。
- **风格预设系统**：内置多种电影级调色风格（如胶片感、赛博朋克、日系小清新等）。
- **用户积分系统**：完整的用户注册、登录及积分消耗机制。
- **云存储集成**：支持七牛云存储，方便管理生成的结果文件。
- **Web 可视化界面**：直观的 Web UI，支持实时预览和历史记录查看。
- **CLI 命令行模式**：支持通过命令行进行批量处理。

## 🛠️ 技术栈

- **后端**：Python 3.11+, Flask
- **数据处理**：NumPy, Pillow (PIL)
- **数据库**：MySQL (aiomysql)
- **AI 服务**：OpenAI SDK (用于兼容调用), Google Generative AI
- **前端**：HTML5, CSS3, JavaScript (原生)

## 🚀 快速开始

### 1. 环境准备

确保已安装 Python 3.11+ 和 MySQL 数据库。推荐使用 Conda 管理环境：

```bash
conda create -n ai-lut python=3.11
conda activate ai-lut
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置文件

复制示例配置文件并修改：

```bash
cp config.example.yaml config.yaml
```

编辑 `config.yaml`，填入以下关键信息：

```yaml
# 数据库连接 (必须)
database_url: "mysql+aiomysql://username:password@localhost:3306/ai_lut_db"

# 管理员账户 (初始化时自动创建)
admin_username: "admin"
admin_password: "your_secure_password"

# AI 模型 API Key (二选一或全部配置)
api_key: "YOUR_GEMINI_API_KEY"          # Google Gemini
doubao_api_key: "YOUR_DOUBAO_API_KEY"   # 字节跳动豆包

# 模型选择
analysis_model: "gemini-1.5-flash"      # 用于图像分析
image_model: "gemini-1.5-flash"         # 用于参考图生成

# 七牛云存储 (可选，用于云端存储生成结果)
qiniu_access_key: ""
qiniu_secret_key: ""
qiniu_bucket: ""
qiniu_domain: ""
```

### 4. 数据库初始化

首次运行时，系统会自动检查并初始化必要的数据库表结构。
> **注意**：如果从旧版本升级（且表结构不兼容），请先备份并清理旧表。

### 5. 启动服务

#### Web 模式 (默认)

```bash
python main.py
```
服务启动后，访问 `http://127.0.0.1:7860` 即可使用。

#### CLI 模式 (命令行)

```bash
CLI_MODE=1 python main.py
```
分析结果和 LUT 文件将默认输出到 `outputs/` 目录。

## 📂 项目结构

```
.
├── app/
│   ├── routes/         # 路由定义 (Web, API, Auth)
│   ├── services/       # 核心业务逻辑 (AI, LUT, User)
│   ├── models/         # 数据库模型
│   └── utils/          # 工具函数
├── static/             # 静态资源 (JS, CSS)
├── templates/          # HTML 模板
├── main.py             # 程序入口
├── config.yaml         # 配置文件 (需手动创建)
└── requirements.txt    # 项目依赖
```

## 📝 数据库表说明

系统会自动创建以下表：
- `users`: 用户账户信息
- `user_points`: 用户积分余额
- `points_transactions`: 积分流水记录
- `analysis_records`: 图片分析历史记录
- `app_settings`: 系统动态配置

## 🤝 贡献

欢迎提交 Issue 或 Pull Request 来改进项目！

## 📄 许可证

[MIT License](LICENSE)
