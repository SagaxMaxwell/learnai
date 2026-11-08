---
date created: 2026-10-07T01:28
date updated: 2026-11-06T18:26
---

# AI 学习笔记

数学基础、机器学习、深度学习与 LLM 应用工程。

```base
filters:
  and:
    - 'file.inFolder("learnai")'
    - '["md", "ipynb"].contains(file.ext)'
    - 'file.path != "learnai/README.md"'
    - '!file.path.contains("/.")'
    - '!file.path.contains("/backups/")'
    - 'file.size > 0'
formulas:
  outline: |
    [
      ["map.md","AI 知识地图"],
      ["math.md","数学与数据表示"],
      ["ml.md","机器学习"],
      ["neural-networks.md","神经网络基础"],
      ["attention.md","Token、Embedding 与 Attention"],
      ["transformer.md","Transformer 与大语言模型"],
      ["apps.md","LLM 应用"]
    ]
  position: 'formula.outline.reduce(if(file.path == "learnai/" + value[0], index, acc), 999)'
  note: 'file.asLink(if(formula.position < formula.outline.length, formula.outline[formula.position][1], file.name))'
  topic: 'if(file.name == "map", "概览", if(file.name == "math", "数学基础", if(file.name == "ml", "机器学习", if(["neural-networks", "attention", "transformer"].contains(file.name), "深度学习", if(file.name == "apps", "LLM 工程", "其他")))))'
properties:
  formula.note:
    displayName: 笔记
views:
  - type: kanban
    name: 笔记
    groupBy:
      property: formula.topic
      direction: ASC
    groupOrder:
      - 概览
      - 数学基础
      - 机器学习
      - 深度学习
      - LLM 工程
      - 其他
    order:
      - formula.note
    sort:
      - property: formula.position
        direction: ASC
      - property: file.path
        direction: ASC
    columnWidth: 260
```

## 学习方式

先读概念，再看公式，最后用小实验验证。比喻只用于建立直觉。
