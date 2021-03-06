# Fund-classification-based-text-data
## 数据描述
本文选取了12672个基金包括'fund_name', 'investment_target', 'investment_scope', 'investment_strategy', 'risk_return_character', 'comparison_criterion','tracking_benchmark'共7个特征的文本表述。相应的基金类型由人工标注，称为标签（type_name_x,stype_name_x,type_nmae_y,stype_nmae_y）。
### 数据处理
1.对人工标注错误的标签进行了修正。  

2.删除了多个特征缺失的基金。
## 分类原则
1. 人工校核成本尽可能低（准确率尽可能高，判别率尽可能高）。  
2. 大样本采用机器学习判别，小样本利用关键词提取。  
3. 区分度低采用机器学习判别，区分度高利用关键词提取（逻辑复杂或模糊）。  
## 模型方法
考虑到关键词匹配和机器学习的优缺点（见表2），单一方法的分类错误较高，本文通过组合两种方法对基金进行分类。图1显示了两种方法的基金分类流程。
表2 关键词匹配和机器学习的优缺点

基于关键词匹配	高效，计算成本小	匹配逻辑的设计复杂；
泛化能力差
基于机器学习模型判定	处理复杂逻辑的能力；
泛化能力；
具有概率输出能力	训练较复杂；
计算成本大；


图1 关键词匹配和机器学习分类流程
具体地说，对于某一维度的基金一级分类，本文的一般分类流程如下：首先根据由关键词设定的判定规则优先确定基金类别，进而对剩余基金进行机器学习判定。对于可直接应用机器学习的分类任务，则无需进行关键词匹配。二级分类由一级分类结果决定，分类顺序与一级分类相似。更多细节见附录A、B。
### 关键词匹配
关键词匹配方法利用某一类型基金的专有标志词、高频词或表述规则进行匹配以确定类别。关键词匹配包括以下细节：
1. 判别特征、字段设计（特征对的结果的影响较大）。  
2. 判别顺序设计（特征差异）。  
3. 判别逻辑设计（关键词类间尽可能独立，区分度尽可能高）。  
### Word2vect+Classifier
机器学习对基金进行分类的主要包括5个步骤：  
1.利用分词模型将文本分成若干词，得到每个基金对应的词列表；  
2.利用了word2vec模型对金融领域语料库进行训练。考虑到时间成本和模型构建的复杂性，本文利用了https://github.com/Embedding/Chinese-Word-Vectors已训练得到的50万词向量库，该词库对本项目的词覆盖率为80%；  
3.利用word2vec模型将词转化为向量，并平均该基金的所有词向量得到一个300维词向量；  
4.利用PCA降维；  
5.结合相应标签利用模型训练和测试。  
训练和测试过程如下：  
为避免类不平衡引起的模型训练，本文从每个基金类别中选取相同数量的样本用于训练，其他样本作为测试；为避免训练样本的随机性，本文重复30次训练，最终测试结果通过平均得到。另外本文测试了常用的21个分类模型，最终选用的模型为ExtraTrees。  
## 评价指标
TP（True Positive），TN（True Positive）：预测答案正确；FP（False Positive）：错将其他类预测为本类；FN（False Negative）：本类标签预测为其他类标。  
1.准确率（Accuracy）：分类正确的样本占总样本的比例。  
acc=\frac{TP+TN}{TP+TN+FP+FN}  
2.精确率（Precision）：对于某一实际类别，预测正确的样本占比。  
precision=\frac{TP}{TP+FP}  
3.召回率（Recall）：对于某一预测类别，预测正确的样本占比。  
recall=\frac{TP}{TP+FN}  
4.F1-score：兼顾精确率和召回率，两者的调和平均。  
F1_{score}=\frac{2precision\times recall}{precision+recall}  
5.混淆矩阵。  
![Image text](https://github.com/hyliush/Fund-classification-based-text-data/blob/main/%E5%81%8F%E8%82%A1%E5%81%8F%E5%80%BA.png)
