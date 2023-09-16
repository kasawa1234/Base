# KOSMOS-2

一种支持**perceiving object detection**的多模态LLM（grounding text to visual world？）

* 重点：多模态的基础上加入了bounding box，可以从圈住的里面提取信息也可以给prompt让他圈
* bound格式：[text span]/(bounding box)
* 与多模态语料库一起构建GRIT



优势：框框是一个很好的交互工具（空间上考虑）

* 可以输入输出一个框
* visual answer更准确且解决了coreference ambiguity（因为不是语言输出 自然没歧义）
* 可以将自由文本相应的回答链接到图像区域上（上面例子中有讲），回答更准确全面



transformer based，因果，下个单词预测任务训练的模型

训练数据集：grounded pairs + KOSMOS-1中的集合



GRIT创建：生成词汇块 - 边界块pair + 生成引用表达式-边界块pair

* 第一步：
  * spaCy切割caption得到noun chunks + 去掉部分对图像识别而言很困难的词汇（降噪）
  * 使用训练好的grounding模型
    * non-maximum suppression算法：去掉高度重叠的框（即便不是一个词汇产生的）
    * 保留confidence 0.65以上的bounding boxes
    * 如果没有剩下的，这组数据去掉
* 第二步：
  * 拓展词块以达到对更复杂描述的ground能力
  * spaCy获取句子的独立关系，递归遍历其依赖树中的子节点并连接起来
    * 有并列的不expand
  * 剔除被他者包含的内容
  * 剩下的bounding box -- 展开的reference



ground的表示：

* 将图像分为PxP个块（可能是这样更泛化？）并给予位置标记（怎么排序？）
* 有一定表达 连接 接入文本的表达方式（左上右下定位）
* grounding相当于就是一个特殊的token，和正常的文字表述是同级的

相当于就是用了KOSMOS-1的结构，只是加了个ground





三个都用作分类

多模态任务？

Kosmos-2

Classification + grounding