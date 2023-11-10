# LLMQA
天池比赛-AI大模型检索问答baseline 代码
说明
- retriver.py  读取pdf索引和检索
- reader.py  利用大模型读取文章回答问题，prompt没怎么优化
#过程
1. 处理PDF先转换成html，然后再分栏处理
2. 使用Langchain 对content 分块，利用Faiss 索引BGE-large表示
3. 检索回来top5使用BGE-rerank (可以直接提交这个结果，记得65左右)
4. 利用大模型如baichuan/Qwen/Chatglm 进行阅读理解提前答案，最优模型合并三个模型的结果到answer1,2,3 ,结果可以到69左右
