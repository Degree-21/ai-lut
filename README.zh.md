# 调色灵感专家 (AI-LUT)

**AI-LUT** 是一个面向摄影、视频后期与影视调色的 **AI 调色/色彩分级（Color Grading）与 3D LUT 生成系统**。它会深度分析参考图像的色彩、光影与构图特征，并结合生成式 AI（Google Gemini、豆包等）输出高精度 `.cube` LUT，帮助快速获得电影级调色灵感与可复用的调色风格。

**关键词**：AI 调色、色彩分级、Color Grading、3D LUT、LUT Generator、电影级调色、Flask、MySQL、生成式 AI、Gemini、豆包、摄影后期、视频调色。

**适用场景**：
- 影视/短视频后期的风格化调色与批量 LUT 生成
- 摄影师快速套用风格与校色参考
- 调色师的风格预设与版本比对
- 课堂/研究中的色彩分析与风格迁移探索

## 🖼️ 应用截图

![首页](docs/index.png)
![功能示意](docs/function.png)
![使用示例](docs/generate.png)

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

## 🏗️ 基础架构

核心流程：前端上传图片 -> Flask 路由 -> 业务服务 -> AI 分析/生成 -> 结果入库与输出文件。

```
Browser
  |
  v
Flask (routes)
  |
  v
Services (analysis / lut / user)
  |           |             |
  |           |             v
  |           |          Qiniu (optional)
  |           v
  |        AI providers (Gemini / Doubao)
  v
MySQL
```

## 🚀 快速开始

### 1. 环境准备

确保已安装 Python 3.11+ 和 MySQL 数据库。推荐使用 Conda 管理环境：

```bash
conda create -n ai-lut python=3.11
conda activate ai-lut
```

### 1.5 使用 Docker 启动 MySQL（可选）

仓库内已提供 `docker-compose.yml`，可直接启动本地 MySQL：

```bash
docker compose up -d db
```

默认数据库与账号（见 `docker-compose.yml`）：
- 数据库：`ai_lut`
- 用户：`ai_lut`
- 密码：`ai_lut_password`
- Root 密码：`ai_lut_root`

对应 `config.yaml` 示例（本地直连）：

```yaml
database_url: "mysql+aiomysql://ai_lut:ai_lut_password@127.0.0.1:3306/ai_lut"
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

> **提示**：未配置七牛时，系统会使用本地 `outputs/` 并通过 `/api/download/...` 提供文件访问。  
> **注意**：若使用豆包模型（`analysis_model` 或 `image_model` 以 `doubao-` 开头），必须配置七牛以提供图片 URL 输入，否则请求会报错。

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

## ❓ 常见问题

**Q: 没有配置七牛是否可以正常使用？**  
A: 可以。未配置七牛时，结果文件会落地到本地 `outputs/`，并通过 `/api/download/...` 访问。  
**注意**：如果选择豆包模型（`analysis_model` 或 `image_model` 以 `doubao-` 开头），必须配置七牛用于图片 URL 输入，否则请求会报错。

**Q: 是否需要初始化 SQL 或导入 `app_settings`？**  
A: 不需要。首次启动会自动建表，`app_settings` 仅在管理员页面保存配置时写入。

## 🤝 贡献

欢迎提交 Issue 或 Pull Request 来改进项目！

## 📄 许可证

[MIT License](LICENSE)
