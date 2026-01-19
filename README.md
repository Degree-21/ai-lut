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
   - 按需填写 `api_key` / `doubao_api_key`、`analysis_model`、`image_model`

## 运行
```bash
python main.py
```
默认地址：`http://127.0.0.1:7860`

## CLI 模式
```bash
CLI_MODE=1 python main.py
```
输出会写入 `outputs/`。

