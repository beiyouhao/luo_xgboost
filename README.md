# luo_xgboost
基于Android声音传感器3种不同频率的数据进行的位置预测


采用numpy和pandas等模块对数据进行预处理，清理脏数据和空值，同时采用最大最小归一化方法，缩放数据维度，保证了模型收敛，进而使用xgboost进行位置的分类，针对5个位置的5个不同朝向，完成该基于有监督学习的多分类问题。
