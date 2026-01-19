# # 方法1：导入sklearn后查看__version__属性（推荐）
# import sklearn
# print(sklearn.__version__)

# # 方法2：也可以导入sklearn的子模块查看（效果相同）
# from sklearn import datasets
# print(sklearn.__version__)

# 核心：查看torch基础版本
import torch
print(torch.__version__)

# 扩展：查看torch是否支持CUDA，以及CUDA版本（GPU用户必看）
print("CUDA是否可用：", torch.cuda.is_available())
print("CUDA版本：", torch.version.cuda)  # 仅当CUDA可用时显示有效版本