# AI-LUT — AI Color Grading & 3D LUT Generator

**Language**: English | [中文](README.zh.md)

**AI-LUT** is an AI-powered color grading and 3D LUT (.cube) generator for photography and video post-production. It analyzes reference images (color, light, composition) and uses generative AI (Gemini, Doubao) to produce high-precision LUTs, helping creators quickly achieve cinematic looks and reusable grading styles.

**Keywords**: AI color grading, Color Grading, 3D LUT, LUT generator, cinematic look, Flask, MySQL, generative AI, Gemini, Doubao, photography, video post-production.

**Use cases**:
- cinematic look development and batch LUT generation for film/short video
- fast style matching and color reference for photographers
- reusable grading presets and version comparison for colorists
- teaching/research on color analysis and style transfer

## 🖼️ Screenshots

![Home](docs/index.png)
![Feature Preview](docs/function.png)

## ✨ Features

- **AI visual analysis**: detects structure, lighting, color temperature, and dynamic range.
- **3D LUT generation**: outputs high-precision `.cube` LUT files.
- **Multi-model support**: Google Gemini Pro/Flash and ByteDance Doubao.
- **Style presets**: built-in cinematic looks (film, cyberpunk, Japanese fresh, etc.).
- **User points system**: registration/login and points consumption.
- **Cloud storage**: Qiniu integration for generated assets.
- **Web UI**: preview, history, and configuration.
- **CLI mode**: batch processing for advanced workflows.

## 🛠️ Tech Stack

- **Backend**: Python 3.11+, Flask
- **Image processing**: NumPy, Pillow (PIL)
- **Database**: MySQL (aiomysql)
- **AI services**: OpenAI SDK (compat), Google Generative AI
- **Frontend**: HTML5, CSS3, Vanilla JS

## 🏗️ Architecture

Main flow: upload -> Flask routes -> services -> AI analysis/generation -> database + output files.

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

## 🚀 Quick Start

### 1. Requirements

Install Python 3.11+ and MySQL. Recommended with Conda:

```bash
conda create -n ai-lut python=3.11
conda activate ai-lut
```

### 1.5 Start MySQL with Docker (optional)

A `docker-compose.yml` is provided. Start MySQL with:

```bash
docker compose up -d db
```

Default database/user (see `docker-compose.yml`):
- DB: `ai_lut`
- User: `ai_lut`
- Password: `ai_lut_password`
- Root password: `ai_lut_root`

Example `config.yaml` (local connection):

```yaml
database_url: "mysql+aiomysql://ai_lut:ai_lut_password@127.0.0.1:3306/ai_lut"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configuration

Copy the sample config and edit:

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` and fill key fields:

```yaml
# Database (required)
database_url: "mysql+aiomysql://username:password@localhost:3306/ai_lut_db"

# Admin account (auto-created on startup)
admin_username: "admin"
admin_password: "your_secure_password"

# AI API keys (one or both)
api_key: "YOUR_GEMINI_API_KEY"
doubao_api_key: "YOUR_DOUBAO_API_KEY"

# Models
analysis_model: "gemini-1.5-flash"
image_model: "gemini-1.5-flash"

# Qiniu (optional)
qiniu_access_key: ""
qiniu_secret_key: ""
qiniu_bucket: ""
qiniu_domain: ""
```

> **Note**: Without Qiniu, files are stored locally in `outputs/` and served via `/api/download/...`.
> **Important**: If you use Doubao models (names start with `doubao-`), Qiniu is required to provide image URL input.

### 4. Database initialization

Tables are created automatically on first startup. No manual SQL is required.
> **Note**: If upgrading from an old schema, back up and drop old tables before restarting.

### 5. Run

#### Web mode (default)

```bash
python main.py
```
Visit `http://127.0.0.1:7860`.

#### CLI mode

```bash
CLI_MODE=1 python main.py
```
Outputs (analysis/LUT) are written to `outputs/`.

## 📂 Project Structure

```
.
├── app/
│   ├── routes/         # Web/API/Auth routes
│   ├── services/       # Core logic (AI, LUT, User)
│   ├── models/         # DB models
│   └── utils/          # Utilities
├── static/             # Static assets (JS, CSS)
├── templates/          # HTML templates
├── main.py             # Entry point
├── config.yaml         # Local config (create manually)
└── requirements.txt    # Dependencies
```

## 📝 Database Tables

Tables are created automatically:
- `users`: user accounts
- `user_points`: point balances
- `points_transactions`: point history
- `analysis_records`: analysis history
- `app_settings`: runtime config storage

## ❓ FAQ

**Q: Can I run without Qiniu?**  
A: Yes. Files are stored in local `outputs/` and served via `/api/download/...`.  
**Important**: Doubao models require Qiniu for image URL input, otherwise requests will fail.

**Q: Do I need to import SQL or prefill `app_settings`?**  
A: No. Tables are created on startup. `app_settings` is only populated when you save settings in the admin panel.

## 🤝 Contributing

Issues and Pull Requests are welcome.

## 📄 License

[MIT License](LICENSE)
