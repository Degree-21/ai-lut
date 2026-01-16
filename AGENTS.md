# Repository Guidelines

## 项目结构与模块组织
核心后端入口为 `main.py`（Flask），前端模板在 `templates/`，静态资源在 `static/`。配置读取在 `config.py`，示例配置为 `config.example.yaml`，实际密钥请放在本地 `config.yaml`（已忽略）。生成结果默认写入 `outputs/`。

## 构建、测试与开发命令
- `python main.py`：启动本地 Web 服务（默认 `127.0.0.1:7860`）。
- `CLI_MODE=1 python main.py`：仅运行命令行流程（输出分析与文件到 `outputs/`）。
- 当前未配置构建流程或打包命令；如需发布，可补充 `pyproject.toml` 或 `setup.cfg`。
- 目前无测试命令；如引入测试，建议使用 `pytest`。

## 编码风格与命名规范
遵循 PEP 8，使用 4 空格缩进；变量与函数采用 `snake_case`，常量使用 `UPPER_SNAKE_CASE`。优先使用 f-string，保持函数短小清晰，必要时补充简洁注释和文档字符串。

## 测试指南
当前未包含测试用例。建议新增 `tests/` 目录，测试文件命名为 `test_*.py`，用例聚焦关键逻辑与异常路径。运行方式：`pytest`。

## 提交与 PR 指南
仓库暂无历史提交，未发现既定提交规范。建议采用简洁、祈使语气的提交信息，或使用 `feat:`、`fix:` 等前缀。PR 需包含变更说明、验证步骤；若涉及可视化输出，请附截图。

## 配置与环境提示
建议使用 Python 3，并在本地创建虚拟环境（如 `.venv/`）。新增第三方依赖时，请维护 `requirements.txt` 并在 PR 中说明用途。
