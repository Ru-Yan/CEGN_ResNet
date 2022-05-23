本坐姿识别网络采用RESNET-50结构，基于tensorflow2.0实现.


##目录
* [验证集错误率](#验证集错误率)
* [训练曲线](#训练曲线)
* [使用指导](#使用指导)
   * [需求](#需求)
   * [残差网络结构](#残差网络结构)


## 验证集错误率
最低的错误率的残差网络为ResNet-32, ResNet-56 and ResNet-110，错误率分别是 6.7%, 6.5% and 6.2%
总层数 = 6 * 残差块数 + 2

网络 | 最低错误率
------- | -----------------------
ResNet-32 | 6.7%
ResNet-56 | 6.5%
ResNet-110 | 6.2%

## 训练曲线
参考train_curve2.png

## 使用指导
`python cifar10_train.py --version='test'`
tensorboard --logdir "C:/Users/lenovo/Desktop/demo/resnet-in-tensorflow-master/logs_test_110"'

tensorboard --logdir "C:/Users/lenovo/Desktop/论文模型/cross-resnet-50/roc"

### 使用条件
pandas, numpy , opencv, tensorflow(1.0.0)

### 残差网络结构
参考appendix/Residual_block.png

   
