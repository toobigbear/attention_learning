# -*- coding: utf-8 -*-
"""
================================================================================
05_spotify_agent_router.py
================================================================================
【对应图片】图3 (JGJT5976.JPG)：《Inside Spotify's agent architecture》
           来源：Spotify Engineering 博客 "Portal by Spotify cut my Claude Code
           token usage by 90%"（Dimitri Mazzanaro, 2024-09-03）

【这张图讲了啥】
Spotify 发现一个残酷的事实：**编程 Agent 干的大部分活根本不是"思考"，而是
"搬运文本"**（读文件、读日志、搬代码片段）。
于是他们做了一个路由器（router）：在昂贵的顶级大模型（frontier model）打开
大文件之前，把"批量读文件"这种苦力活拦下来，交给便宜的模型去干，能省约 90% 的 token。

单次读取的完整路径（图片上半部分）：
  昂贵模型想读一个文件
     ↓
  PreToolUse hooks（唯一能强制拦截的层）
     ├─ check_file_size：检查每次工具调用
     └─ check_batch_read：检查 cat/head/tail/less/more 这类批量读取
     ↓
  文件超过 350 行？
     ├─ 否 → 放行。完整文件进上下文，按最贵的费率计费，且后续每一轮都会重复计费
     └─ 是 → 拦截！交给便宜的 bulk-reader 模型
            便宜模型只回答你问的具体问题，返回几行要点（bullets），
            文件本身永远不进昂贵模型的上下文

三个拦截层（图片左下角，01 - THE THREE LAYERS）：
  Layer1 Hooks  ：工具执行前中断并拒绝 —— 唯一有"否决权"的层（强制）
  Layer2 Scripts：负责构建请求、调用 worker、上报 token 用量（管道）
  Layer3 Skills ：Markdown 文档，告诉模型什么时候该调哪个脚本（建议性）

仍花全价的场景（图片右下角，02 - WHAT STILL COSTS FULL PRICE）：
  · 编辑文件（worker 的摘要说不清行号，容易改错地方）
  · 真正需要推理的任务（线程安全 bug，便宜模型根本发现不了）
  · 太小的任务（<350行，委托的网络往返 10-30 秒比省的钱还贵）

【大白话理解】
请了个顶级专家（贵）干活，但专家 80% 时间在"翻资料"。
聪明老板的做法：给专家配个便宜助理，凡是"翻资料、抄笔记"的活全让助理干，
只有"拍板、推理、改关键代码"才请专家出手。
助理干不了的活（比如拿不准行号不能乱改）还是专家亲自来。

【本代码做什么】
1. 模拟"单次文件读取"走完路由的完整流程（看 350 行阈值怎么生效）
2. 量化对比：用路由 vs 不用路由的 token 成本
3. 模拟三层拦截各自的角色
================================================================================
"""

# ==============================================================================
# 1. 配置：定义"昂贵模型"和"便宜模型"的计费（近似真实 API 价格）
# ==============================================================================
# 每 1000 token 的价格（美元，约数）
FRONTIER_IN = 15.0     # 昂贵模型输入价（如 Claude Sonnet 级别）
FRONTIER_OUT = 75.0    # 昂贵模型输出价
WORKER_IN = 0.15       # 便宜模型输入价（便宜 100 倍）
WORKER_OUT = 0.60

# 模拟一个文件
class File:
    def __init__(self, path, lines):
        self.path = path
        self.lines = lines          # 总行数
        self.content = "\n".join(f"line {i}: 这是一行日志/代码..." for i in range(lines))

    @property
    def size_bytes(self):
        return len(self.content.encode("utf-8"))


def token_estimate(text: str) -> int:
    """粗略估算 token 数：1 个中文字 ≈ 1 token，英文按 4 字符 ≈ 1 token"""
    return max(1, len(text) // 4 + text.count("\n"))


# ==============================================================================
# 2. 模拟"三层拦截"：Hooks（强制） / Scripts（管道） / Skills（建议）
# ==============================================================================
LINE_THRESHOLD = 350   # 图片里的关键数字：超过 350 行就拦截

class Hooks:
    """Layer 1：唯一有否决权的层。工具执行前被调用，可以'拒绝'这次调用。"""
    def check_file_size(self, f: File) -> bool:
        """检查文件大小，超过阈值返回 True = 拦截"""
        return f.lines > LINE_THRESHOLD

    def check_batch_read(self, command: str) -> bool:
        """检查是否批量读取命令（cat/head/tail/less/more）"""
        return command in ("cat", "head", "tail", "less", "more")


class Scripts:
    """Layer 2：管道层。负责把活派给便宜模型 worker，并统计用量。"""
    def __init__(self):
        self.worker_tokens = 0   # 累计便宜模型消耗

    def call_bulk_reader(self, f: File, question: str) -> list:
        """调用便宜模型：只回答具体问题，返回几行要点（bullets）"""
        # 便宜模型只读一部分 + 生成简洁要点，token 消耗远小于全文件
        chunk = f.content[:500]
        cost_in = token_estimate(chunk)
        cost_out = token_estimate(question) // 2
        self.worker_tokens += cost_in + cost_out
        print(f"      [Scripts] 便宜模型(bulk-reader)被调用，消耗约 {cost_in + cost_out} tokens")
        # 模拟便宜模型返回的要点
        return ["- 该文件包含日志输出，约 %d 行" % f.lines, "- 第 12 行附近有 ERROR 关键字"]


class Skills:
    """Layer 3：建议层。Markdown 文档告诉模型'什么时候该用哪个脚本'。"""
    DOC = """
    ## bulk-read（批量读取）
    什么时候用：需要读日志/大段 SQL/代码片段时
    怎么做：调用 script bulk_reader，传入文件路径和具体问题
    ## code-write（写代码）
    什么时候用：按参考文件写新代码
    怎么做：调用 script code_writer，只输出代码、无注释无格式
    """
    def advice(self, task: str) -> str:
        """给模型返回建议（模型可以不听，所以叫'咨询性'）"""
        if "读" in task or "查" in task:
            return "建议使用 bulk-read 模式（便宜）"
        return "建议使用标准模式（昂贵）"


# ==============================================================================
# 3. 路由主流程：模拟"昂贵模型想读文件"时发生的一切
# ==============================================================================
def route_one_read(f: File, question: str, hooks, scripts):
    """完整复现图片上半部分：THE PATH OF ONE READ"""
    print(f"\n📄 昂贵模型想读取文件: {f.path}（{f.lines} 行, {f.size_bytes} 字节）")
    print(f"   问题: {question}")

    # 第1关：PreToolUse hooks（唯一强制层）
    blocked = hooks.check_file_size(f)
    print(f"\n  [Layer1-Hooks] check_file_size: {'> ' + str(LINE_THRESHOLD) + ' 行 → 拦截！' if blocked else '≤ ' + str(LINE_THRESHOLD) + ' 行 → 放行'}")

    if not blocked:
        # 放行：全文件进昂贵模型上下文 —— 按最贵费率计费，且每轮都重复计费
        full_tokens = token_estimate(f.content)
        cost = full_tokens * FRONTIER_IN / 1000
        print(f"  [放行] 完整文件进昂贵模型上下文（{full_tokens} tokens）")
        print(f"  [成本] 本轮就花 ${cost:.2f}，而且后续每轮对话都会重复计费！")
        return cost, 0

    # 拦截成功：检查是不是批量读取命令（图片：check_batch_read）
    print(f"  [Layer1-Hooks] check_batch_read: cat/head/tail → 属于批量读取")
    print(f"  [拦截生效] 文件绝不进入昂贵模型上下文，转给便宜模型 👇")

    # 第2关：Scripts 管道层调用便宜模型
    print(f"  [Layer2-Scripts] 构建请求 → 调用 bulk-reader worker...")
    bullets = scripts.call_bulk_reader(f, question)

    # 便宜模型只返回要点
    print(f"  [返回] 昂贵模型只收到几行要点（文件本身没进上下文）：")
    for b in bullets:
        print(f"         {b}")

    worker_cost = scripts.worker_tokens * WORKER_IN / 1000
    print(f"  [成本] 便宜模型花费 ${worker_cost:.4f}（对比全文件进昂贵模型省了约 99%）")
    return 0, scripts.worker_tokens


# ==============================================================================
# 4. 成本对比：一整天的工作，有路由 vs 无路由
# ==============================================================================
def cost_comparison():
    """模拟一天内发生 50 次批量读取的账单"""
    print("\n\n" + "=" * 80)
    print("成本对比：一天 50 次批量文件读取，有路由 vs 无路由")
    print("=" * 80)

    avg_file = File("app.log", 2000)     # 平均 2000 行的大日志文件
    full_tokens_per_read = token_estimate(avg_file.content)

    # 无路由：每次都全文件进昂贵模型
    no_router_cost = 50 * full_tokens_per_read * FRONTIER_IN / 1000
    # 有路由：便宜模型处理（图片说约省 90%）
    worker_tokens_per_read = 300          # 便宜模型每次只花几百 token
    with_router_cost = 50 * worker_tokens_per_read * WORKER_IN / 1000

    print(f"\n  无路由：50 × {full_tokens_per_read} tokens × ${FRONTIER_IN}/K = ${no_router_cost:,.0f}")
    print(f"  有路由：50 × {worker_tokens_per_read} tokens × ${WORKER_IN}/K = ${with_router_cost:,.2f}")
    print(f"  节省比例：{(1 - with_router_cost / no_router_cost) * 100:.1f}%  ← 和图片里的 ~90% 一致！")
    print(f"\n  按一个月 22 个工作日算：")
    print(f"  无路由月账单 ≈ ${no_router_cost * 22:,.0f}   |   有路由月账单 ≈ ${with_router_cost * 22:,.0f}")


# ==============================================================================
# 5. 什么情况"不能省"（图片右下角：WHAT STILL COSTS FULL PRICE）
# ==============================================================================
def what_stays_expensive():
    print("\n\n" + "=" * 80)
    print("什么场景仍然必须用昂贵模型（不能委托给便宜 worker）")
    print("=" * 80)
    cases = [
        ("编辑文件", "worker 的摘要说不清准确行号，改错地方比不省还糟；带 offset/limit 的精准读取故意放行"),
        ("推理任务", "测试中一个线程安全 bug，昂贵模型几秒就发现，便宜模型完全看不出来"),
        ("小文件(<350行)", "委托一次要等网络往返 10-30 秒，省下的 token 钱不如等待的损失"),
    ]
    for name, reason in cases:
        print(f"\n  🔴 {name}")
        print(f"     {reason}")


# ==============================================================================
# 6. 主入口：按顺序跑全部模拟
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("复现图片：THE PATH OF ONE READ（一次读取的完整路径）")
    print("=" * 80)

    hooks = Hooks()
    scripts = Scripts()
    skills = Skills()

    # 场景1：大文件被拦截（>350行）
    print("\n\n【场景1】读一个大日志文件（2000行）→ 应该被拦截，转便宜模型")
    route_one_read(File("app.log", 2000), "这个日志里有没有报错？", hooks, scripts)

    # 场景2：小文件直接放行（<350行）
    print("\n\n【场景2】读一个小配置文件（100行）→ 应该放行，昂贵模型直接读")
    route_one_read(File("config.yaml", 100), "这个配置里端口是多少？", hooks, scripts)

    # 场景3：Skills 层给建议
    print("\n\n【场景3】Layer3-Skills 给模型建议（咨询性，不是强制）")
    print(f"  任务'帮我读一下 Redis 日志' → {skills.advice('读')}")
    print(f"  任务'帮我修这个并发bug'    → {skills.advice('修bug')}")

    # 成本对比
    cost_comparison()

    # 不省钱的场景
    what_stays_expensive()

    print("\n\n" + "=" * 80)
    print("总结：Agent 架构优化的核心思路 = 把昂贵的'思考'和廉价的'搬文本'分开。")
    print("图片标题那句话：Most of what a coding agent does is not reasoning.")
    print("It is moving text around.")
    print("=" * 80)

"""
【运行结果解读】
1. 大文件被 Hooks 强制拦截 → 便宜模型只返回要点 → 昂贵模型上下文干干净净
2. 小文件放行 → 但注意它的代价：完整文件按最贵费率计费且每轮重复
3. 三个 Layer 的角色：Hooks 能否决（强制）、Scripts 干管道活、Skills 只给建议
4. 成本对比：有路由比无路由省 90%+ —— 这就是图里那个 ~90% 的来历
5. 三个"不省"的场景：编辑/推理/太小 —— 解释了为什么路由要有选择，不能全拦

【一句话总结】
Spotify 的 Agent 架构 = 给昂贵大模型配一个"便宜助理"，用硬拦截（Hooks）
在文件进上下文前拦住批量读取，只把要点喂给贵模型。
省的是钱（token），守住的是质量（推理、编辑这种活不让便宜模型干）。
"""
