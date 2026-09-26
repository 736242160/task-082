#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nested_parse.py — 嵌套结构多规则解析器（纯 Python 标准库，单文件）

用法：
    python3 nested_parse.py                 # 运行内置自测样例
    python3 nested_parse.py rules.txt in.txt  # 用规则文件解析文本文件

规则文件格式（每行一条，# 开头为注释，字段可用引号包围）：
    规则名  匹配模式(正则)  动作  [类型名]
动作：
    入栈   命中后开启一个节点并压栈（类型名缺省=规则名）
    出栈   命中后关闭栈顶节点；类型名声明它期望闭合的类型
    收集   命中后生成一个叶子节点挂到当前栈顶节点下
    豁免   命中的整段区域（如字符串、注释）内的其它命中全部无效

优先级策略（同位置多规则命中时的取舍）：
    1. 豁免区最优先：扫描位置落在豁免区域内时，直接跳过整段，
       区域内任何正常规则的命中都不生效（字符串/注释是词法层约定）。
    2. 同一位置多条正常规则同时命中：匹配文本更长者优先
       （最长匹配 = 最具体，与 lex/flex 的惯例一致）；
       长度相同则规则表中书写靠前者优先（用户可用书写顺序微调）。
    3. 歧义不静默：同一位置若正常规则与豁免规则同时命中，
       豁免规则生效，但会报告一条 AMBIGUITY 错误提示用户检查规则。

错误报告（均含 行:列 位置）：
    POP_EMPTY      出栈时栈空
    TYPE_MISMATCH  出栈类型不符（期望 / 实际）
    UNCLOSED       文件结束仍有未闭合节点（报告入栈起始位置）
    AMBIGUITY      同位置正常规则与豁免规则同时适用
"""

import re

import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

ACTIONS = ("入栈", "出栈", "收集", "豁免")


# ---------------------------------------------------------------- 数据结构

@dataclass
class Rule:
    name: str
    pattern: str
    action: str
    type: str
    order: int
    regex: "re.Pattern"


@dataclass
class Node:
    type: str
    start: int                      # 起始偏移
    end: int = -1                   # 结束偏移（-1 = 未闭合）
    text: str = ""                  # 收集节点的文本
    children: List["Node"] = field(default_factory=list)


@dataclass
class ParseError:
    kind: str
    pos: int
    message: str


# ---------------------------------------------------------------- 工具函数

def linecol(text: str, pos: int) -> Tuple[int, int]:
    """偏移 -> (行, 列)，均从 1 开始。"""
    line = text.count("\n", 0, pos) + 1
    col = pos - text.rfind("\n", 0, pos)
    return line, col


def load_rules(source: str) -> List[Rule]:
    """从规则文本解析规则列表；格式错误抛 ValueError。"""
    rules = []
    for lineno, raw in enumerate(source.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # 首尾切分：第一个空白分隔出规则名，末尾 1~2 个字段是 动作 [类型]，
        # 中间部分整体作为正则模式（允许模式内含空格与引号）。
        head = line.split(None, 1)
        if len(head) != 2:
            raise ValueError(f"规则第 {lineno} 行格式错误（应为 3 或 4 个字段）: {raw!r}")
        name, rest = head
        tail = rest.rsplit(None, 2)
        if len(tail) == 3:
            pattern, action, rtype = tail
        elif len(tail) == 2:
            pattern, action = tail
            rtype = name
        else:
            raise ValueError(f"规则第 {lineno} 行格式错误（应为 3 或 4 个字段）: {raw!r}")
        if action not in ACTIONS:
            raise ValueError(f"规则第 {lineno} 行动作未知: {action!r}（可选: {ACTIONS}）")
        try:
            regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
        except re.error as exc:
            raise ValueError(f"规则第 {lineno} 行正则错误: {exc}") from exc
        rules.append(Rule(name, pattern, action, rtype, len(rules), regex))
    return rules


# ---------------------------------------------------------------- 核心解析

def find_exempt_regions(text: str, exempt_rules: List[Rule]) -> List[Tuple[int, int, str]]:
    """收集所有豁免区域并合并重叠段，返回按起点排序的 (start, end, rule_name)。"""
    regions = []
    for rule in exempt_rules:
        for m in rule.regex.finditer(text):
            if m.end() > m.start():          # 忽略零宽匹配
                regions.append((m.start(), m.end(), rule.name))
    regions.sort(key=lambda r: (r[0], -(r[1] - r[0])))
    merged: List[List] = []
    for start, end, name in regions:
        if merged and start < merged[-1][1]:     # 与上一段重叠/包含
            if end > merged[-1][1]:
                merged[-1][1] = end
            continue
        merged.append([start, end, name])
    return [tuple(r) for r in merged]


def parse(text: str, rules: List[Rule]):
    """解析文本，返回 (根节点, 错误列表)。"""
    exempt_rules = [r for r in rules if r.action == "豁免"]
    normal_rules = [r for r in rules if r.action != "豁免"]
    regions = find_exempt_regions(text, exempt_rules)

    root = Node("ROOT", 0, len(text))
    stack: List[Node] = [root]
    errors: List[ParseError] = []

    pos, ri, n = 0, 0, len(text)
    while pos < n:
        # 推进到当前位置之后最近的豁免区
        while ri < len(regions) and pos >= regions[ri][1]:
            ri += 1
        # 情况 1：当前位置是豁免区起点 —— 检查歧义后跳过整段
        if ri < len(regions) and regions[ri][0] == pos:
            hits = [(r, r.regex.match(text, pos)) for r in normal_rules]
            hits = [(r, m) for r, m in hits if m and m.end() > pos]
            if hits:
                names = "、".join(sorted({r.name for r, _ in hits}))
                errors.append(ParseError(
                    "AMBIGUITY", pos,
                    f"歧义：正常规则 [{names}] 与豁免规则 [{regions[ri][2]}] "
                    f"在同一位置同时命中，豁免规则生效"))
            pos = regions[ri][1]
            ri += 1
            continue
        # 情况 2：当前位置在豁免区内部 —— 命中无效，跳过
        if ri < len(regions) and regions[ri][0] < pos < regions[ri][1]:
            pos = regions[ri][1]
            ri += 1
            continue

        # 收集当前位置所有正常规则命中
        hits = []
        for rule in normal_rules:
            m = rule.regex.match(text, pos)
            if m and m.end() > pos:
                hits.append((rule, m))
        if not hits:
            pos += 1
            continue

        # 优先级：最长匹配优先；等长则规则表靠前者优先
        hits.sort(key=lambda hm: (-(hm[1].end() - pos), hm[0].order))
        rule, m = hits[0]

        if rule.action == "入栈":
            node = Node(rule.type, pos)
            stack[-1].children.append(node)
            stack.append(node)
        elif rule.action == "出栈":
            if len(stack) == 1:
                errors.append(ParseError(
                    "POP_EMPTY", pos,
                    f"出栈规则 [{rule.name}] 命中时栈为空，无处可弹"))
            else:
                top = stack[-1]
                if top.type != rule.type:
                    errors.append(ParseError(
                        "TYPE_MISMATCH", pos,
                        f"出栈类型不符：期望闭合 [{rule.type}]，"
                        f"实际栈顶为 [{top.type}]（入栈于 "
                        f"{linecol(text, top.start)[0]}:{linecol(text, top.start)[1]}）"))
                stack.pop().end = m.end()      # 弹出以恢复，继续解析
        elif rule.action == "收集":
            stack[-1].children.append(Node(rule.type, pos, m.end(), m.group()))
        pos = m.end()

    # 文件结束仍未出栈的节点
    while len(stack) > 1:
        node = stack.pop()
        ln, cl = linecol(text, node.start)
        errors.append(ParseError(
            "UNCLOSED", node.start,
            f"节点 [{node.type}] 自 {ln}:{cl} 入栈后直到文件结束未闭合"))
        node.end = n
    return root, errors


# ---------------------------------------------------------------- 输出

def render_tree(node: Node, text: str, depth: int = 0, out: Optional[List[str]] = None) -> str:
    if out is None:
        out = []
    indent = "  " * depth
    ln, cl = linecol(text, node.start)
    if depth == 0:
        out.append(f"{indent}ROOT")
    elif node.text:
        snippet = node.text if len(node.text) <= 30 else node.text[:27] + "..."
        out.append(f"{indent}{node.type} @{ln}:{cl} 文本={snippet!r}")
    else:
        state = f"@{ln}:{cl}" if node.end >= 0 else f"@{ln}:{cl} (未闭合)"
        out.append(f"{indent}{node.type} {state}")
    for child in node.children:
        render_tree(child, text, depth + 1, out)
    return "\n".join(out)


def render_errors(errors: List[ParseError], text: str) -> str:
    if not errors:
        return "（无错误）"
    lines = []
    for e in sorted(errors, key=lambda e: e.pos):
        ln, cl = linecol(text, e.pos)
        lines.append(f"  [{e.kind}] {ln}:{cl}  {e.message}")
    return "\n".join(lines)


def run(text: str, rule_source: str, title: str = "") -> Tuple[Node, List[ParseError]]:
    rules = load_rules(rule_source)
    root, errors = parse(text, rules)
    if title:
        print("=" * 64)
        print(f"【{title}】")
        print("-- 输入文本 " + "-" * 40)
        print(text)
        print("-- 结构树 " + "-" * 42)
    print(render_tree(root, text))
    print("-- 错误报告 " + "-" * 40)
    print(render_errors(errors, text))
    return root, errors


# ---------------------------------------------------------------- 自测样例

DEMO_RULES = r"""
# 规则名      匹配模式                动作    类型
LINE_COMMENT \#[^\n]*                豁免
STRING      "(?:\\.|[^"\\])*"        豁免
LBRACE      \{                       入栈    brace
RBRACE      \}                       出栈    brace
LPAREN      \(                       入栈    paren
RPAREN      \)                       出栈    paren
NUMBER      \d+(?:\.\d+)?            收集    number
IDENT       [A-Za-z_]\w*             收集    ident
"""

def self_test() -> int:
    failures = 0

    def check(title, text, expect_kinds):
        nonlocal failures
        print("=" * 64)
        print(f"【{title}】")
        print("-- 输入文本 " + "-" * 40)
        print(text)
        root, errors = run(text, DEMO_RULES)
        kinds = sorted(e.kind for e in errors)
        ok = kinds == sorted(expect_kinds)
        print(f"-- 断言: 期望错误 {sorted(expect_kinds)}，实际 {kinds} -> "
              + ("通过" if ok else "失败"))
        if not ok:
            failures += 1

    # 1. 正常嵌套 + 字符串/注释豁免（字符串里的 { } 不算）
    check("正常嵌套 + 豁免",
          'foo { bar (1, 2.5) s = "{ not a brace }" } # comment ( )\n',
          [])

    # 2. 出栈类型不符：{(})
    check("出栈类型不符",
          "{ ( }",
          ["TYPE_MISMATCH", "UNCLOSED"])

    # 3. 出栈时栈空
    check("出栈时栈空",
          "a ) b",
          ["POP_EMPTY"])

    # 4. 文件结束未闭合（报告入栈起始位置）
    check("未闭合",
          "x { y ( 1",
          ["UNCLOSED", "UNCLOSED"])

    # 5. 歧义：正常规则 QUOTE 与豁免规则 STRING 在同一位置同时命中
    ambiguous_rules = DEMO_RULES + 'QUOTE      "                       收集    quote\n'
    print("=" * 64)
    print("【歧义：正常规则与豁免规则同位置同时命中】")
    text = 'a "str" { b }'
    print("-- 输入文本 " + "-" * 40)
    print(text)
    root, errors = run(text, ambiguous_rules)
    kinds = sorted(e.kind for e in errors)
    ok = kinds == ["AMBIGUITY"]
    print(f"-- 断言: 期望 ['AMBIGUITY']，实际 {kinds} -> " + ("通过" if ok else "失败"))
    if not ok:
        failures += 1

    # 6. 同位置多规则优先级：==（长）应胜过 =（短）
    prio_rules = (
        "EQEQ      ==                      收集    eq_op\n"
        "EQ        =                       收集    assign\n"
        "IDENT     [A-Za-z_]\\w*           收集    ident\n"
    )
    print("=" * 64)
    print("【优先级：最长匹配优先（== 胜过 =）】")
    text = "a == b = c"
    print("-- 输入文本 " + "-" * 40)
    print(text)
    root, errors = run(text, prio_rules)
    kinds = [c.type for c in root.children]
    ok = kinds == ["ident", "eq_op", "ident", "assign", "ident"] and not errors
    print(f"-- 断言: 期望 [ident, eq_op, ident, assign, ident]，实际 {kinds} -> "
          + ("通过" if ok else "失败"))
    if not ok:
        failures += 1

    print("=" * 64)
    print(f"自测结果：{'全部通过' if failures == 0 else f'{failures} 项失败'}")
    return 1 if failures else 0


def main(argv: List[str]) -> int:
    if len(argv) == 1:
        return self_test()
    if len(argv) == 3:
        with open(argv[1], encoding="utf-8") as f:
            rule_source = f.read()
        with open(argv[2], encoding="utf-8") as f:
            text = f.read()
        run(text, rule_source, title=f"{argv[1]} -> {argv[2]}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
