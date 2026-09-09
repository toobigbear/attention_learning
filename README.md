# 注意力机制学习工程（通过代码学概念）

> 本工程把七张图片整理成包括 5 个知识点，每个知识点一个 `.py` 文件，
> 每个文件都带**超详细注释 + 可运行代码**，运行就能看到每一步的数值变化。

## 工程结构（按学习顺序）

| 文件 | 对应图片 | 知识点 | 难度 |
|---|---|---|---|
| `01_agent_workflow.py` | 图1 (CRUB0369) | Agent 是怎么工作的（任务→规划→工具→检查→输出循环） | ★ 入门 |
| `02_self_attention.py` | 图6 (RJJO0109) | 自注意力机制核心公式（Q/K/V、打分、softmax、加权求和） | ★★ 基础 |
| `03_multi_head_attention.py` | 图2/4/5 (JAPW7556/MHJJ1323/MXLS4500) | 多头注意力全流程（分头→各头算注意力→拼接） | ★★★ 核心 |
| `04_attention_family.py` | 图7 (SRDN7754) | 注意力家族：MHA / MQA / GQA / MLA / DSA / CSA / KDA | ★★★ 进阶 |
| `05_spotify_agent_router.py` | 图3 (JGJT5976) | Spotify 真实 Agent 架构：用便宜模型拦截昂贵的文件读取 | ★★★ 工程 |

## 学习方法建议

1. **一个文件一个知识点**，从 `01` 开始顺序学。
2. 每个文件先读文件头的"这张图讲了啥"，再看代码。
3. **运行代码**，观察打印出的每一步矩阵变化：
   ```bash
   python 01_agent_workflow.py
   python 02_self_attention.py
   python 03_multi_head_attention.py
   python 04_attention_family.py
   python 05_spotify_agent_router.py
   ```
4. 看完打印结果，再回到图片对照：**图片上每个框，对应代码里哪几行**。
5. 试着改代码里的数字（比如改成 4 个头、换一段输入文本），看结果怎么变。

## 依赖

只需要 numpy：
```bash
pip install numpy
```
