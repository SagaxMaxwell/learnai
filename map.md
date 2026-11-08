---
date created: 2026-10-07
date updated: 2026-11-06
---

# AI 知识地图

模型内部解释“文字怎样变成输出”，应用工程解释“怎样利用模型完成任务”。这里只保留关系图，概念与示例放在对应笔记。

## 总关系图

```mermaid
flowchart TD
    A[Python 与编程基础]
    B[函数、向量、矩阵、概率]
    C[损失、梯度、优化]
    D[传统机器学习<br/>回归、分类、聚类]
    E[神经网络]
    F[前向传播与反向传播]
    G[Tokenizer 与 Token 词表]
    H[Token ID]
    I[Embedding 表<br/>Token ID → 基础向量]
    J[位置表示<br/>位置向量或 RoPE]
    K[Transformer 层]
    L[Q、K、V 投影]
    M[Attention<br/>Q 与 K 打分，按权重读取 V]
    N[前馈网络、残差、归一化]
    O[多层 Transformer]
    P[输出层 Logits]
    Q[Softmax / 解码]
    R[下一个 Token]
    S[追加到上下文]
    T[KV Cache<br/>缓存历史 Token 的 K、V]

    U[用户问题]
    V[Prompt 与 Context]
    W[RAG 检索]
    X[外部文档片段]
    Y[Tool Calling]
    Z[MCP<br/>统一连接工具、资源、提示]
    AA[Skill<br/>任务方法与操作说明]
    AB[Workflow<br/>固定步骤]
    AC[Agent<br/>模型 + 工具 + 状态 + 循环]
    AD[执行结果 / 最终回答]

    A --> B --> C --> D --> E --> F
    F --> G
    G --> H --> I --> J --> K
    K --> L --> M --> N --> O --> P --> Q --> R --> S
    S --> K
    K -.每层产生.-> T
    T -.历史 K/V 供新 Query 查询.-> M

    U --> V
    V --> W
    X --> W
    W --> V
    V --> K
    V --> Y
    Y --> Z
    AA --> V
    AA --> AC
    V --> AB
    AB --> AD
    AC --> Y
    AC --> W
    Y --> AD
    O --> AD
```

## 阅读入口

- [[math|数学与数据]]：向量、矩阵、概率、损失与梯度。
- [[ml|机器学习]]：任务、训练集划分、评估和常见算法。
- [[neural-networks|神经网络]]：前向传播、反向传播和训练。
- [[attention|Token 与 Attention]]：分词、Embedding、位置和 QKV。
- [[transformer|Transformer]]：模型结构、训练、生成和 KV Cache。
- [[apps|LLM 应用]]：Prompt、RAG、MCP、Skill、Workflow 与 Agent。
