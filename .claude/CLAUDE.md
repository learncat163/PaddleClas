要求：

1. 如果有待完成的任务，则 记录到 .claude/todos/todo.md 文件里；如果这个任务完成，则从此文件里清除。
2. 参考 .github/copilot-instructions.md 的配置说明
3. 当前项目的主要任务是：从 transfomer中迁移代码到paddleclas框架里，已知信息如下：raw-transformer-dinov3 目录是原始的transfomer的代码目录；paddle-transfer-dinov3 是通过paddle的 paconvert工具直接翻译的代码，不能直接运行，只能参考。
4. 我们的目标是：把dinov3的逻辑全部移植到 arch/backbone/model_zoo/dinov3.py 通过单文件实现。
5. 如果有临时脚本、临时日志、临时文件、临时文档要生成，统一放到 tmp目录下
6. 任意一个回应开始的时候，都要在前面带上【小曹】这个称谓。
7. 当前项目paddleclas就是要替代transformer，不能尝试pip 安装这个库
