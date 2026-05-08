# pytorch_distilling

 本仓库代码主要用于使用pytorch演示teacher模型，student模型和蒸馏模型构建和性能评估

 # distilling model.py
 ## teacher大模型的构建，training和evalution
 ## student小模型的构建，training和evalution
 ## distilling model的构建，training和 evalution,其中主要涉及参数有：
 ### temperature的设置， tem越大，差异越不明显，选择合适的值，能够蒸馏到性能优越的模型
 ### hard_loss
 ### alpha 
 ### soft_loss，其中需要关注soft_loss中，student的logits要经过log_softmax,teacher的logits是经过softmax,最后的soft_loss要乘以tmp **2，因为 softmax 的输入除以了 T，KL loss 关于 logits 的梯度幅值会缩小约 1/T²。Hinton 论文要求在 soft loss 上乘 T² 来补偿

 
 
