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
- 若从旧版本（用户表 `id` 为字符串）升级，请直接清空表后重启：
```sql
DROP TABLE IF EXISTS app_settings;
DROP TABLE IF EXISTS analysis_records;
DROP TABLE IF EXISTS points_transactions;
DROP TABLE IF EXISTS user_points;
DROP TABLE IF EXISTS users;
```

## 系统配置存储
- 系统配置（模型、Key、注册赠送积分）默认写入数据库 `app_settings` 表。

## 七牛云存储
- 所有生成文件（分析文本、参考图、LUT）可上传到七牛公有空间。
- 配置项（`config.yaml` 或环境变量）：
  - `qiniu_access_key` / `QINIU_ACCESS_KEY`
  - `qiniu_secret_key` / `QINIU_SECRET_KEY`
  - `qiniu_bucket` / `QINIU_BUCKET`
  - `qiniu_domain` / `QINIU_DOMAIN`（示例：`https://qn.3xx3x.cn`）

## CLI 模式
```bash
CLI_MODE=1 python main.py
```
输出会写入 `outputs/`。
