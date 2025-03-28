# VChart仓库智能问答机器人

## 项目简介

这是一个基于 LangChain、FAISS 和 Ollama 的智能问答机器人，专门用于回答仓库文档相关的问题。该项目采用 Streamlit 构建用户界面，支持中英文双语切换，并提供了丰富的交互功能。

## 主要特点

- 🌍 双语支持
  - 中文界面
  - 英文界面
  - 实时语言切换

- 💬 智能对话
  - 基于文档的精准回答
  - 上下文理解
  - 自然语言交互

- 📚 文档处理
  - 自动加载 Markdown 文档
  - 文本智能分块
  - 向量化存储
  - 文档去重优化

- 🔄 会话管理
  - 新建对话
  - 清空对话
  - 导出对话记录
  - 对话历史保存

- ⚡ 性能优化
  - 向量存储持久化
  - 增量更新支持
  - 批量处理优化
  - 内存使用优化

## 技术栈

- Streamlit：用户界面框架
- LangChain：大语言模型应用框架
- FAISS：向量检索引擎
- Ollama：本地大语言模型服务
- Python：开发语言（3.8 或更高版本）

## 环境要求

1. 系统要求：
   - Windows 10/11
   - 8GB 以上内存
   - 10GB 以上硬盘空间（用于存储模型）

2. 软件要求：
   - Python 3.8 或更高版本
   - Ollama 最新版本
   - Git（可选，用于克隆项目）

## 安装说明

1. 克隆项目：
```bash
git clone [项目地址]
cd [项目目录]
```

2. 创建并激活虚拟环境（推荐）：
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 安装 Ollama：
- 访问 [Ollama官网](https://ollama.ai/) 下载安装包
- 安装完成后运行：
```bash
ollama serve
ollama pull llama2
```

5. 创建必要的目录结构：
```bash
mkdir -p assets/guide assets/api assets/faq
```

## 使用方法

1. 准备文档：
   - 将 Markdown 格式的文档放入对应目录：
     - `assets/guide/`: 指南文档
     - `assets/api/`: API文档
     - `assets/faq/`: 常见问题
   - 文档要求：
     - 必须是 .md 格式
     - 使用 UTF-8 编码
     - 建议每个文档大小不超过 1MB

2. 启动应用：
```bash
streamlit run chatbot.py
```

3. 首次使用：
   - 程序会自动加载文档并创建向量存储
   - 这个过程可能需要几分钟，请耐心等待
   - 向量存储会保存在 `vector_store` 目录中

4. 使用功能：
   - 选择界面语言（中文/English）
   - 在输入框中输入问题
   - 点击"提交问题"获取回答
   - 使用工具栏管理对话

5. 工具栏功能：
   - 🆕 新对话：开始新的对话
   - 🗑️ 清空对话：清除当前对话记录
   - 📋 导出对话：保存对话记录到文件
   - 🔄 重新初始化：重新加载文档和模型
   - 💾 更新知识库：更新文档内容

## 常见问题解决

1. Ollama 服务问题：
   - 确保 Ollama 服务正在运行
   - 如遇端口占用，可以结束 Ollama 进程后重启
   - 检查防火墙设置

2. 向量存储问题：
   - 如果加载失败，可以删除 `vector_store` 目录后重试
   - 确保磁盘有足够空间
   - 不要手动修改向量存储文件

3. 文档加载问题：
   - 检查文档编码（推荐 UTF-8）
   - 确保文档格式正确
   - 检查文件权限

4. 内存问题：
   - 如果内存占用过高，可以减小批处理大小
   - 关闭其他占用内存的应用
   - 考虑增加系统内存

## 性能优化建议

1. 文档处理：
   - 文档大小建议控制在 1MB 以内
   - 适当调整文本分块大小
   - 定期清理无用文档

2. 向量存储：
   - 定期重建向量索引
   - 避免频繁更新
   - 备份重要的向量存储

3. 系统资源：
   - 保持足够的磁盘空间
   - 监控内存使用
   - 适时清理缓存

## 安全注意事项

1. 文档安全：
   - 不要加载不信任来源的文档
   - 定期备份重要文档
   - 注意文档的访问权限

2. 向量存储安全：
   - 不要随意删除或修改向量存储文件
   - 在加载向量存储时注意来源安全
   - 定期备份向量存储

## 更新日志

### v1.0.0
- 实现基础问答功能
- 添加中英文界面
- 支持文档向量化存储
- 添加会话管理功能
- 优化性能和用户体验

## 贡献指南

欢迎提交 Issue 和 Pull Request 来帮助改进项目。在提交之前，请确保：
1. 代码符合 PEP 8 规范
2. 添加必要的注释和文档
3. 测试所有功能正常

## 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 目录结构

```
项目根目录/
├── chatbot.py      # 主程序
├── assets/         # 文档目录
│   ├── guide/      # 指南文档
│   ├── api/        # API文档
│   └── faq/        # 常见问题
├── vector_store/   # 向量存储
└── README.md       # 项目说明
```

## 配置说明

1. 文档存放：
   - 将 Markdown 文档放入 `assets` 目录的对应子文件夹中
   - 支持的文件格式：`.md`
   - 自动跳过项目自身的 README.md

2. 向量存储：
   - 首次运行时自动创建
   - 存储在 `vector_store` 目录
   - 支持增量更新

## 注意事项

1. 运行要求：
   - Python 3.8 或更高版本
   - 确保 Ollama 服务正常运行
   - 需要安装 llama2 模型

2. 性能优化：
   - 文档块大小：500字符
   - 批处理大小：50
   - 支持文档去重

3. 使用建议：
   - 问题尽量具体
   - 如果回答不准确，可以换个方式提问
   - 支持中英文提问

## 附加内容
- 您可以对代码获得README文件的形式进行更改，以改变该机器人所能回答的范围，但在更改后请重新加载向量存储
- 本仓库已上传现有的向量存储，且您可以自行定义交互界面
- 您可以修改文件后缀读取部分的代码，使其可以接收其他类型的文件

总而言之，该项目的功能不仅仅限于对于VChart仓库的智能问答，您也可以将他作为自建知识库的问答机器人。

Nothing is impossible.
