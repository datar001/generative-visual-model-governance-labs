"""Render only exercises/answers; never regenerate practices or evidence.

The question tuple is [specific title, question, answer]. Default: check only.
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = r"\section*{习题}"


def load_curriculum():
    data = json.loads((ROOT / 'content/curriculum.json').read_text(encoding='utf-8'))
    assert [c['chapter'] for c in data] == list(range(1, 9))
    for chapter in data:
        questions = chapter['questions']
        assert len(questions) == 12
        assert len({q[0] for q in questions}) == 12
        for title, question, answer in questions:
            assert title and question and answer
            assert title not in {'基础知识', '原理分析', '实践设计'}
    return data


def render_markdown(chapter, answers=False):
    kind = '参考答案' if answers else '习题'
    intro = ('逐题给出判断依据、计算过程或方案示例；开放题允许有充分依据的其他解法。'
             if answers else '共12题。题目中的教学假设数据用于说明计算和推理，不代表模型实测性能。先独立作答，再核对参考答案。')
    lines = [f"# 第{chapter['chapter']}章 {chapter['title']}：{kind}", '', intro, '']
    for i, (title, question, answer) in enumerate(chapter['questions'], 1):
        lines += [f"## {chapter['chapter']}-{i:02} {title}", '', question, '']
        if answers:
            lines += ['**参考答案**', '', answer, '']
    lines += ['[返回本章](README.md)', '']
    return '\n'.join(lines)


def tex(text):
    """Escape prose and render the Unicode math used in questions explicitly."""
    symbols = {
        '×': r'\ensuremath{\times}', '÷': r'\ensuremath{\div}',
        '−': r'\ensuremath{-}', '≈': r'\ensuremath{\approx}',
        '≥': r'\ensuremath{\geq}', '≤': r'\ensuremath{\leq}',
        '≠': r'\ensuremath{\neq}', '→': r'\ensuremath{\rightarrow}',
        'ε': r'\ensuremath{\varepsilon}', 'α': r'\ensuremath{\alpha}',
        'δ': r'\ensuremath{\delta}', 'σ': r'\ensuremath{\sigma}',
        '∞': r'\ensuremath{\infty}', 'Δ': r'\ensuremath{\Delta}',
        'λ': r'\ensuremath{\lambda}', '±': r'\ensuremath{\pm}',
        '∩': r'\ensuremath{\cap}', '′': r'\ensuremath{\prime}',
        '\\': r'\textbackslash{}', '%': r'\%', '&': r'\&',
        '_': r'\_', '#': r'\#', '$': r'\$', '{': r'\{', '}': r'\}',
        '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
        '=': r'=\allowbreak{}', '+': r'+\allowbreak{}', '/': r'/\allowbreak{}',
    }
    # Preserve the scope of bars, hats, roots and exponents.
    patterns = [
        (r'w＝1＋s', r'\ensuremath{w=1+s}'),
        (r'x_(noise|data)', lambda m: r'\ensuremath{x_{\mathrm{' + m[1] + '}}}'),
        (r'ε_(cond|uncond)', lambda m: r'\ensuremath{\varepsilon_{\mathrm{' + m[1] + '}}}'),
        (r'\(([−－-]?\d+)\)²', lambda m: r'\ensuremath{(' + m[1].replace('－','-').replace('−','-') + ')^2}'),
        (r'log₁₀', r'\ensuremath{\log_{10}}'),
        (r'\(1/2\)⁶', r'\ensuremath{(1/2)^6}'),
        (r'√ᾱₜ', r'\ensuremath{\sqrt{\bar{\alpha}_t}}'),
        (r'√\(1−ᾱₜ\)', r'\ensuremath{\sqrt{1-\bar{\alpha}_t}}'),
        (r'ᾱₜ', r'\ensuremath{\bar{\alpha}_t}'),
        (r'ε̂', r'\ensuremath{\hat{\varepsilon}}'),
        (r'([A-Za-z])([₀₁₂ₜ]+)', lambda m: r'\ensuremath{' + m[1] + '_{' + m[2].translate(str.maketrans('₀₁₂ₜ','012t')) + '}}'),
        (r'e⁻ᶻ', r'\ensuremath{e^{-z}}'),
        (r'([A-Za-z0-9]+)([⁰¹²³⁴⁵⁶⁷⁸⁹]+)', lambda m: r'\ensuremath{' + m[1] + '^{' + m[2].translate(str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹','0123456789')) + '}}'),
        (r'√(\d+(?:\.\d+)?)', lambda m: r'\ensuremath{\sqrt{' + m[1] + '}}'),
    ]
    # Protected tokens avoid re-escaping the TeX produced by substitutions.
    protected = []
    for pattern, replacement in patterns:
        def replace(match):
            value = replacement(match) if callable(replacement) else replacement
            protected.append(value)
            return '\x00' + str(len(protected)-1) + '\x00'
        text = re.sub(pattern, replace, text)
    parts = re.split(r'(\x00\d+\x00)', text)
    out = []
    for part in parts:
        if re.fullmatch(r'\x00\d+\x00', part):
            out.append(protected[int(part[1:-1])])
        else:
            out.append(''.join(symbols.get(ch, ch) for ch in part))
    result = ''.join(out)
    assert not re.search('[\u0300-\u036f\u2070-\u209f\u1dbb]', result), result
    return result.replace('\n\n', '\n\\par\\smallskip\n').replace('\n', '\n')


def render_tex(chapter):
    lines = [MARKER + r'\addcontentsline{toc}{section}{习题}',
             r'\begingroup\emergencystretch=2em',
             '本章共12题，题号与配套仓库一致。题目中的教学假设数据用于说明计算和推理，不代表模型实测性能。', '']
    for i, (title, question, answer) in enumerate(chapter['questions'], 1):
        lines += [r'\noindent\textbf{' + f"{chapter['chapter']}-{i:02}　" + tex(title) + r'}\par\nobreak',
                  r'\noindent ' + tex(question) + r'\par\medskip', '']
    lines += [r'\subsection*{参考答案}',
              '以下逐题说明判断依据、计算过程或方案示例；开放题允许有充分依据的其他解法。', '']
    for i, (title, question, answer) in enumerate(chapter['questions'], 1):
        lines += [r'\noindent\textbf{' + f"{chapter['chapter']}-{i:02}　" + tex(title) + r'}\par\nobreak',
                  r'\noindent ' + tex(answer) + r'\par\medskip', '']
    lines += [r'\par\endgroup', '']
    return '\n'.join(lines).encode('utf-8')


def replace_exercise_tail(raw, chapter):
    marker = MARKER.encode('utf-8')
    assert raw.count(marker) == 1, chapter['file']
    prefix, tail = raw.split(marker, 1)
    # Abort rather than remove any newly added content after the exercises.
    assert not re.search(rb'\\(?:chapter|section|appendix|end\{document\})', tail), chapter['file']
    assert tail.count(r'\noindent\textbf{'.encode()) == 24, chapter['file']
    return prefix + render_tex(chapter)


def run(write=False, book_root=None):
    data = load_curriculum()
    targets = {}
    for chapter in data:
        folder = ROOT / 'chapters' / chapter['slug']
        targets[folder / 'exercises.md'] = render_markdown(chapter).encode('utf-8')
        targets[folder / 'answers.md'] = render_markdown(chapter, True).encode('utf-8')
        if book_root:
            source = book_root / 'section_0802' / chapter['file']
            targets[source] = replace_exercise_tail(source.read_bytes(), chapter)
    changed = []
    for path, expected in targets.items():
        if path.read_bytes() != expected:
            changed.append(str(path))
            if write:
                path.write_bytes(expected)
    if changed and not write:
        raise SystemExit('Out of date exercise files:\n' + '\n'.join(changed))
    print(json.dumps({'mode': 'write' if write else 'check', 'chapters': 8,
                      'questions': 96, 'answers': 96, 'changed_files': len(changed)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Explicitly update exercise and answer files only.')
    parser.add_argument('--book-root', type=Path, help='Optional local book root containing section_0802; exercise tails only.')
    args = parser.parse_args()
    run(args.write, args.book_root)
