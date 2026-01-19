# 调色灵感专家

## 环境准备（Conda）
```bash
conda create -n ai-lut python=3.11
conda activate ai-lut
pip install -r requirements.txt
```

## 配置
1. 复制配置模板：
```bash
cp config.example.yaml config.yaml
```
2. 修改 `config.yaml`：
   - 设置 `database_url`（需要可用的 MySQL 实例）
   - 设置 `admin_username` / `admin_password`
   - 按需填写 `api_key` / `doubao_api_key`、`analysis_model`、`image_model`、`register_bonus_points`

## 运行
```bash
python main.py
```
默认地址：`http://127.0.0.1:7860`

## 数据库规范
- 所有表必须包含 `id` 主键列，以及 `created_at` / `updated_at` 时间戳列。
- `updated_at` 需要开启自动更新时间（`ON UPDATE CURRENT_TIMESTAMP`）。

## 系统配置存储
- 系统配置（模型、Key、注册赠送积分）默认写入数据库 `app_settings` 表。

## CLI 模式
```bash
CLI_MODE=1 python main.py
```
输出会写入 `outputs/`。
