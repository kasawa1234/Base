# minigpt

视觉方面（pretrained）：BLIP-2 + Q-Former

* 加入一层投影层（linear）
* Vicuna和visual全部frozen

训练：20k 对齐

问题：简单对齐不具有视觉对话能力（高性能模型），原始数据对中的噪声可能导致不连贯语言输出

* 用3500对高质量对齐数据对 + 设计好的对话模板做fine tune

三个点：

* 视觉align to语言模型可以达到gpt效果（甚至有一些新东西）	
* 训练更少
* 只用公开数据训练可能出问题（需要更好的数据对（高质量、对齐很好））



related：

* visualgpt frozen 预训练模型
* flamingo：vision encoder + lm gated cross-attention 上下文小样本学习
* BLIP-2：flan-t5 q-former 对齐（也可以减少训练次数）
* PaLM-E：向LLM中融入连续的现实传感器 实现现实感知和语言的连接
* GPT4：更广泛 深刻的 语言图像连接



LLM：中心协调（？）各种神奇的 使用其他模块+LLM达到的功能

* 加强视觉模型
* 用于prompt生成（可以提取图片内容）（？）
* 视觉查询



minigpt只用了vision encoder，没有使用训练好的视觉模型



方法：

* encoder：blip2同款
  * ViT + Q-Former
  * ViT：分成PxP的patches并直接拉平 + position embedding + class token
    * position embedding：permutation-invariant，理论上输入顺序不改变输出结果（动态输入？），所以不能再加入额外学习成本
    * 根据一定算法产生一个与每个patch拉平后长度一样的不同的向量加上去
    * class token：使用linear分类，产出一个分类向量加到头部（position固定）
  * Q-Former：（不完全了解）
* encoder后接一个linear 对接LLM

两步走：大量粗糙训练+少量fine tune

第一步：20000步，batchsize 256，约5m对数据，10h，4 A100

第二步：为vicuna设计一个template，将encoder + linear产出的feature输入，强制输入80tokens以上 -- 生成更详细的文本-图片对 -- 还不好？chatgpt改一改（或许这也是一种模型接入？hhh）再人工检查一下 -- 随机instruction训练

400步，batch size 12，3500对数据，8min，1 A100



实现功能：

* 图片介绍
* meme介绍
* 发现不寻常（可能性低）的内容
* 通过手写文字生成网站
* 识别图片问题并提供解决方案
* 以图片为启发创作诗歌和rap
* 为图片写一段故事
* 打广告
* 识别图片内个体
* insight
* 相关事实
* 怎么做饭



limitations：

* 语言幻觉：不可靠的推理 + 臆想存在的知识
  * 更高质量的数据？
  * 更高端的LLM？
* 感知能力不足：难以识别详细文本信息及区分空间定位
  * 缺少相应训练数据
  * Q- Former本身缺少信息
  * 一层投影层可能不充分



vision encoder

* 24层



MML：24层MAGNETO Transformer

* 2048隐藏维度
* 32attention头



图像：224x224，patch size 14x14（图像划分），P=7（32分，32x32）（位置划分）



instruct tuning：基于相关指令学习

* expression：提expression instruction，让模型补全后面的bounding
* bounding：给bounding box，生成信息