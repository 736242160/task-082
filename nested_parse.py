#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nested_parse.py — 嵌套结构多规则解析器（纯 Python 标准库，单文件）

用法:
    python3 nested_parse.py 规则文件 输入文本文件
    python3 nested_parse.py --selftest        # 运行内置自测样例

规则文件格式（每行一条，空行与 # 开头的行忽略）:
    规则名  匹配模式(正则, 不含空白)  动作

动作:
    push[:类型]   入栈（省略类型时类型 = 规则名）；也接受 入栈[:类型]
    pop[:类型]    出栈；也接受 出栈[:类型]
    collect       把命中文本收集为叶子；也接受 收集

优先级规则（自定，理由如下）:
    1. 同一位置多条规则命中时，规则文件中先定义者优先；
       理由：顺序由用户显式控制，行为确定、可预期、可调试。
    2. 同一规则在同一位置有多种匹配时，取最长匹配；
       理由：最长匹配最具体，信息损失最小（与词法分析惯例一致）。
    3. 规则命中被某条规则消费后，其覆盖区间内的其他命中不再处理。

豁免与歧义:
    - 字符串（"..." / '...'，支持 \\ 转义）与注释（#...、//...、/* ... */）
      区间内部的命中一律豁免（不算）。
    - 若某位置既是字符串/注释界定符的起点、又有规则命中，则报“歧义”，
      并按豁免优先处理（字符串/注释是词法层结构，应优先于语法层规则）。

错误报告（均含 行:列 位置）:
    - 出栈时栈顶类型不符（期望 / 实际）
    - 出栈时栈为空
    - 文件结束仍有未闭合的入栈（报告入栈起始位置）
    - 未闭合的字符串 / 块注释
"""

import re
import sys
from dataclasses import dataclass, field


# ---------- 数据结构 ----------

@dataclass
class Rule:
    name: str
    pattern: str
    action: str   # 'push' | 'pop' | 'collect'
    type: str     # 节点类型（push/pop 配对依据）
    order: int    # 定义顺序 = 优先级（小者优先）
    regex: object


@dataclass
class Leaf:
    rule: str
    text: str
    start: int
    end: int


@dataclass
class Node:
    type: str
    rule: str
    start: int
    end: int = -1  # -1 表示未闭合
    children: list = field(default_factory=list)


@dataclass
class Issue:
    pos: int
    kind: str
    detail: str


# ---------- 规则加载 ----------

ACTIONS = {
    'push': 'push', '入栈': 'push',
    'pop': 'pop', '出栈': 'pop',
    'collect': 'collect', '收集': 'collect',
}


def load_rules(lines):
    rules, problems = [], []
    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) != 3:
            problems.append(f'第 {lineno} 行格式错误（应为: 规则名 模式 动作）: {raw!r}')
            continue
        name, pattern, action = parts
        if ':' in action:
            act_raw, typ = action.split(':', 1)
        else:
            act_raw, typ = action, name
        act = ACTIONS.get(act_raw)
        if act is None:
            problems.append(f'第 {lineno} 行动作未知: {act_raw!r}')
            continue
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            problems.append(f'第 {lineno} 行正则无效: {exc}')
            continue
        rules.append(Rule(name, pattern, act, typ, len(rules), rx))
    return rules, problems


# ---------- 豁免区间扫描（字符串 / 注释） ----------

def scan_exempt(text):
    """返回 (豁免区间列表, 问题列表)。区间 = (start, end, 种类)。"""
    spans, issues = [], []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ''
        if ch in ('"', "'"):
            j = i + 1
            while j < n:
                if text[j] == '\\':
                    j += 2
                    continue
                if text[j] == ch:
                    break
                j += 1
            end = j + 1 if j < n else n
            if j >= n:
                issues.append(Issue(i, '未闭合字符串', '字符串开始后未找到结束引号'))
            spans.append((i, end, '字符串'))
            i = end
        elif ch == '#' or (ch == '/' and nxt == '/'):
            j = text.find('\n', i)
            end = n if j == -1 else j
            spans.append((i, end, '注释'))
            i = end
        elif ch == '/' and nxt == '*':
            j = text.find('*/', i + 2)
            if j == -1:
                issues.append(Issue(i, '未闭合注释', '块注释未找到 */'))
                end = n
            else:
                end = j + 2
            spans.append((i, end, '注释'))
            i = end
        else:
            i += 1
    return spans, issues


# ---------- 主解析 ----------

def parse(text, rules):
    spans, issues = scan_exempt(text)
    notes = []  # (pos, 说明) —— 优先级决策等提示信息
    span_starts = {s: (s, e, k) for s, e, k in spans}

    def inside_span(p):
        return any(s < p < e for s, e, _ in spans)

    hits = {}
    for r in rules:
        for m in r.regex.finditer(text):
            hits.setdefault(m.start(), []).append((r, m.end()))

    root = Node('ROOT', '', 0, len(text))
    stack = []
    consumed = -1  # 已被消费的位置下界

    for pos in sorted(hits):
        cands = hits[pos]
        if pos < consumed:
            continue  # 已被上一条命中消费
        if inside_span(pos):
            continue  # 豁免：字符串/注释内部命中不算
        if pos in span_starts:
            _, _, kind = span_starts[pos]
            names = '、'.join(r.name for r, _ in cands)
            issues.append(Issue(pos, '歧义',
                f'规则 [{names}] 与{kind}界定符同时适用于此位置，按豁免优先处理'))
            continue
        # 优先级：先定义者优先；同规则取最长匹配
        cands.sort(key=lambda c: (c[0].order, pos - c[1]))
        rule, end = cands[0]
        if len(cands) > 1:
            losers = '、'.join(r.name for r, _ in cands[1:])
            notes.append((pos, f'多条规则命中，按优先级选择 [{rule.name}]，放弃 [{losers}]'))
        consumed = max(consumed, end)

        if rule.action == 'push':
            node = Node(rule.type, rule.name, pos)
            (stack[-1].children if stack else root.children).append(node)
            stack.append(node)
        elif rule.action == 'pop':
            if not stack:
                issues.append(Issue(pos, '出栈落空',
                    f'规则 [{rule.name}] 欲出栈类型 [{rule.type}]，但栈为空'))
            else:
                top = stack[-1]
                if top.type != rule.type:
                    issues.append(Issue(pos, '栈顶类型不符',
                        f'期望出栈 [{top.type}]（规则 {top.rule}，入栈于位置 {top.start}），'
                        f'实际遇到 [{rule.type}]（规则 {rule.name}）；已按出栈恢复'))
                top.end = end
                stack.pop()
        else:  # collect
            leaf = Leaf(rule.name, text[pos:end], pos, end)
            (stack[-1].children if stack else root.children).append(leaf)

    for node in stack:
        issues.append(Issue(node.start, '未闭合',
            f'类型 [{node.type}]（规则 {node.rule}）在此入栈，文件结束仍未出栈'))
    return root, issues, notes


# ---------- 输出 ----------

def linecol(text, pos):
    line = text.count('\n', 0, pos) + 1
    col = pos - text.rfind('\n', 0, pos)
    return line, col


def render(node, text, depth=0, out=None):
    out = out if out is not None else []
    for child in node.children:
        ln, col = linecol(text, child.start)
        if isinstance(child, Leaf):
            out.append('  ' * depth + f'收集[{child.rule}] {child.text!r}  @{ln}:{col}')
        else:
            end_s = str(child.end) if child.end >= 0 else '未闭合'
            out.append('  ' * depth + f'<{child.type}> 规则={child.rule}  @{ln}:{col} 结束={end_s}')
            render(child, text, depth + 1, out)
    return out


def print_report(root, text, issues, notes):
    print('== 结构树 ==')
    for line in render(root, text):
        print(line)
    if notes:
        print('\n== 决策说明 ==')
        for pos, msg in notes:
            ln, col = linecol(text, pos)
            print(f' - {ln}:{col} {msg}')
    print('\n== 错误报告 ==')
    if not issues:
        print('（无错误）')
    for it in sorted(issues, key=lambda x: x.pos):
        ln, col = linecol(text, it.pos)
        print(f' - [{it.kind}] {ln}:{col} {it.detail}')


# ---------- 自测样例 ----------

SELFTESTS = [
    ('1. 正常嵌套与收集',
     r'''
block_begin  \{          push:block
block_end    \}          pop:block
word         [A-Za-z]+   collect
''',
     'a { b { c } d } e'),

    ('2. 同位置多规则命中，先定义者优先',
     r'''
double_open   \{\{   push:double
double_close  \}\}   pop:double
open          \{     push:block
close         \}     pop:block
word          [A-Za-z]+  collect
''',
     '{{ x }}'),

    ('3. 出栈时栈顶类型不符',
     r'''
open     \{     push:block
close    \}     pop:block
lclose   \]     pop:list
''',
     '{ ]'),

    ('4. 出栈时栈为空',
     r'''
open     \{     push:block
close    \}     pop:block
''',
     'a }'),

    ('5. 文件结束仍未闭合',
     r'''
open     \{          push:block
close    \}          pop:block
word     [A-Za-z]+   collect
''',
     '{ a { b }'),

    ('6. 字符串与注释中的命中被豁免',
     r'''
open     \{          push:block
close    \}          pop:block
word     [A-Za-z]+   collect
''',
     '{ a } # { fake } 且 " { 也不算 } " ok'),

    ('7. 正常规则与豁免界定符同位置（歧义）',
     r'''
quote    "           collect
word     [A-Za-z]+   collect
''',
     'say "hi"'),
]


def run_selftest():
    for title, rules_text, text in SELFTESTS:
        print('=' * 60)
        print(title)
        print('-' * 60)
        print('输入:', repr(text))
        rules, problems = load_rules(rules_text.splitlines())
        for p in problems:
            print('规则错误:', p)
        root, issues, notes = parse(text, rules)
        print_report(root, text, issues, notes)
        print()


# ---------- 入口 ----------

def main(argv):
    if '--selftest' in argv[1:]:
        run_selftest()
        return 0
    if len(argv) != 3:
        print(__doc__)
        return 2
    with open(argv[1], encoding='utf-8') as f:
        rule_lines = f.read().splitlines()
    with open(argv[2], encoding='utf-8') as f:
        text = f.read()
    rules, problems = load_rules(rule_lines)
    for p in problems:
        print('规则错误:', p, file=sys.stderr)
    if not rules:
        return 1
    root, issues, notes = parse(text, rules)
    print_report(root, text, issues, notes)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
