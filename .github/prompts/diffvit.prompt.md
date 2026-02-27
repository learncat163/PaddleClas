现在我在改进 ppcls/arch/backbone/model_zoo/efficientvit.py 文件。已知的信息如下：

1. cyy_test/vit-diff/raw-arch.txt 文件是原版的 EfficientViT 架构定义文件。
2. cyy_test/vit-diff/paddle-arch.txt 文件是 ppcls/arch/backbone/model_zoo/efficientvit.py 输出的 架构定义文件。

我们的目标是：
1. 尝试不断修改 ppcls/arch/backbone/model_zoo/efficientvit.py 文件，直到与 cyy_test/vit-diff/paddle-arch.txt 文件  和  cyy_test/vit-diff/raw-arch.txt 文件的结构意义一致。

限制：
目前 cyy_test/vit-diff/paddle-arch.txt 是我自行填写的，你可以编写代码自动填充。