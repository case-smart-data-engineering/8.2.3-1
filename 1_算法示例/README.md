# 算法示例

## 使用指南

1. 按 `CTRL + P` 打开命令行面板，输入 "terminal: Create New Terminal" 打开一个命令行终端.
2. 在命令行里输入 `cd 1_算法示例` 并按 `ENTER` 进入"算法示例"目录。
3. 在命令行里输入 `python solution.py` 按 `ENTER` 运行示例程序。
> `python solution.py`实际是默认执行了`python solution.py --mode evaluation`命令，此处默认执行了evaluation命令行参数
一共有以下三类参数操作：
- 在命令行里输入`python solution.py --mode preprocessing`按 `ENTER` 运行示例程序，会把1_算法示例/raw_data/chinese下的数据集解析成1_算法示例/data/chinese下的数据。
- 在命令行里输入`python solution.py --mode train`按 `ENTER` 运行示例程序，会用1_算法示例/data/chinese下的数据进行模型训练，生成的模型文件存储在1_算法示例/saved_models下。
> 配置文件详见1_算法示例/experiments/chinese_selection_re.json，默认训练十个epoch，使用chinese_selection_re_9作为最终模型。
- 在命令行里输入 `python solution.py --mode evaluation` 按 `ENTER` 运行示例程序，会得到一句话的实体关系抽取结果。
> 运行命令后，所有数据都写入到了1_算法示例/data/writeDate/triples.csv中，可以打开查看详细结果。


## 目录指南

- data:data目录存储了数据集文件。
- experiments:experiments目录存储了配置文件chinese_selection_re.json。
- lib:lib目录存储了自定义的库文件。
- saved_models:saved_models目录存储了训练好的模型文件。
