# 算法示例

## 使用指南

1. 按 `CTRL + P` 打开命令行面板，输入 "terminal: Create New Terminal" 打开一个命令行终端.
2. 在命令行里输入 `cd 1_算法示例` 并按 `ENTER` 进入"算法示例"目录。
3. 在命令行里输入 `python predict.py` 按 `ENTER` 运行示例程序，已经上传了训练好的模型可以直接使用。
4. 如果需要手动训练模型，请按以下步骤进行：
- 在命令行里输入 `cd 1_算法示例` 并按 `ENTER` 进入"算法示例"目录。
- 在命令行里输入 `python data.py` 按 `ENTER` 运行，会在1_算法示例/data/NYT_CoType下生成entity_labels.txt、relation_labels.txt等文件。
- 在命令行里输入 `python train.py` 按 `ENTER` 运行，即可进行模型训练。


## 文件指南

- data:data文件夹存储了数据集文件和word2vec文件。
- conv_net.py:CNN模型文件。
- data.py:处理data文件并存储为二进制文件。
- model.pt:训练好的模型文件，直接使用。
- model.py:主要的模型文件。
- predict.py:运行示例代码文件。
- utils.py:存储了工具类的文件。
- train.py:训练模型文件。