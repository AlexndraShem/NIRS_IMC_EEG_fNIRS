"""Markdown -> PDF без внешних зависимостей (markdown_it + matplotlib PdfPages).

Мини-вёрстка A4: заголовки, абзацы с **жирным** / *курсивом* / `кодом`, списки (вложенные),
таблицы (ширины колонок по содержимому, перенос текста в ячейках, перенос таблицы между
страницами с повтором шапки), блоки кода (моноширинный шрифт на сером фоне, авто-уменьшение
кегля), картинки (по ширине полосы), горизонтальные линейки, титульная шапка, оглавление
по H2/H3 с номерами страниц, нумерация страниц. Шрифты DejaVu встраиваются (Type 42).

Запуск:  python -m src.scripts.md2pdf <in.md> <out.pdf> [--team "Команда HORMIGAS"] [--no-toc]
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from functools import lru_cache

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.ft2font import FT2Font, LOAD_NO_HINTING  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from markdown_it import MarkdownIt  # noqa: E402
from markdown_it.tree import SyntaxTreeNode  # noqa: E402
from PIL import Image  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["pdf.compression"] = 9

# ------------------------------------------------------------------ геометрия и стиль
PAGE_W, PAGE_H = 595.28, 841.89  # A4, pt
MARGIN_L = MARGIN_R = 56.7  # 2 см
MARGIN_T = 56.7
MARGIN_B = 62.0
TEXT_W = PAGE_W - MARGIN_L - MARGIN_R
BOTTOM = PAGE_H - MARGIN_B

BODY = 9.5
BODY_LH = 1.38
TABLE_SIZES = (7.8, 7.2, 6.6, 6.0)
CODE_MAX, CODE_MIN = 8.0, 5.8
H_SIZES = {1: 18.0, 2: 14.5, 3: 11.5, 4: 10.5, 5: 10.0, 6: 10.0}
IMG_DPI_GUESS = 150.0  # картинки не растягиваем сильнее, чем ~1.15 × их «родной» размер

C_TEXT = "#1f1f1f"
C_MUTED = "#6b6b6b"
C_RULE = "#9aa4b1"
C_CODE_BG = "#f3f4f6"
C_CODE_BORDER = "#d6d9de"
C_INLINE_CODE_BG = "#eef0f3"
C_TH_BG = "#e6ebf2"
C_ZEBRA = "#f8f9fb"
C_GRID = "#b9c0c9"
C_HEAD = "#16325c"

_FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
FONT_FILES = {
    ("sans", False, False): "DejaVuSans.ttf",
    ("sans", True, False): "DejaVuSans-Bold.ttf",
    ("sans", False, True): "DejaVuSans-Oblique.ttf",
    ("sans", True, True): "DejaVuSans-BoldOblique.ttf",
    ("mono", False, False): "DejaVuSansMono.ttf",
    ("mono", True, False): "DejaVuSansMono-Bold.ttf",
    ("mono", False, True): "DejaVuSansMono-Oblique.ttf",
    ("mono", True, True): "DejaVuSansMono-BoldOblique.ttf",
}
FONT_PATHS = {k: os.path.join(_FONT_DIR, v) for k, v in FONT_FILES.items()}
NORMAL = ("sans", False, False)
BOLD = ("sans", True, False)
ITALIC = ("sans", False, True)
MONO = ("mono", False, False)


@lru_cache(maxsize=None)
def _ft(style):
    f = FT2Font(FONT_PATHS[style])
    f.set_size(100.0, 72)
    return f


@lru_cache(maxsize=None)
def _adv100(ch: str, style) -> float:
    """Ширина глифа (advance) при кегле 100 pt."""
    f = _ft(style)
    if ord(ch) not in f.get_charmap():
        ch = "?"
    g = f.load_char(ord(ch), flags=LOAD_NO_HINTING)
    return g.linearHoriAdvance / 65536.0


@lru_cache(maxsize=200000)
def _w100(s: str, style) -> float:
    return sum(_adv100(c, style) for c in s)


def text_w(s: str, style, size: float) -> float:
    return _w100(s, style) * size / 100.0


@lru_cache(maxsize=None)
def fprop(style) -> FontProperties:
    return FontProperties(fname=FONT_PATHS[style])


# ------------------------------------------------------------------ inline -> атомы
@dataclass
class Atom:
    parts: list  # [(text, style)]
    space: bool = False  # перед атомом пробел
    br: bool = False  # принудительный перевод строки

    def width(self, size):
        return sum(text_w(t, s, size) for t, s in self.parts)


def inline_chunks(node, bold=False, italic=False, out=None):
    """Разворачивает inline-узел в список (text, style) / ('\n', None) для hardbreak."""
    if out is None:
        out = []
    for ch in node.children or []:
        t = ch.type
        if t == "text":
            out.append((ch.content, ("sans", bold, italic)))
        elif t == "code_inline":
            out.append((ch.content, ("mono", bold, italic)))
        elif t == "softbreak":
            out.append((" ", ("sans", bold, italic)))
        elif t == "hardbreak":
            out.append(("\n", None))
        elif t == "strong":
            inline_chunks(ch, True, italic, out)
        elif t == "em":
            inline_chunks(ch, bold, True, out)
        elif t == "s":
            inline_chunks(ch, bold, italic, out)
        elif t == "link":
            inline_chunks(ch, bold, italic, out)
            href = ch.attrs.get("href", "")
            label = "".join(c.content for c in ch.walk() if c.type in ("text", "code_inline"))
            if href and not href.startswith("#") and href != label:
                out.append((f" ({href})", ("sans", bold, italic)))
        elif t == "image":
            out.append((f"[рисунок: {ch.attrs.get('alt', '') or ch.content}]", ("sans", bold, True)))
        elif t in ("html_inline",):
            out.append((ch.content, ("sans", bold, italic)))
        elif ch.children:
            inline_chunks(ch, bold, italic, out)
        else:
            out.append((ch.content or "", ("sans", bold, italic)))
    return out


def chunks_to_atoms(chunks):
    atoms: list[Atom] = []
    cur: Atom | None = None
    pending_space = False
    for text, style in chunks:
        if style is None:  # hardbreak
            if cur:
                atoms.append(cur)
                cur = None
            atoms.append(Atom([], br=True))
            pending_space = False
            continue
        text = text.replace("\t", "    ").replace("\n", " ")
        i = 0
        n = len(text)
        while i < n:
            if text[i] == " ":
                if cur:
                    atoms.append(cur)
                    cur = None
                pending_space = True
                while i < n and text[i] == " ":
                    i += 1
                continue
            j = i
            while j < n and text[j] != " ":
                j += 1
            word = text[i:j]
            if cur is None:
                cur = Atom([(word, style)], space=pending_space and bool(atoms) and not atoms[-1].br)
                pending_space = False
            else:
                if cur.parts[-1][1] == style:
                    cur.parts[-1] = (cur.parts[-1][0] + word, style)
                else:
                    cur.parts.append((word, style))
            i = j
    if cur:
        atoms.append(cur)
    return atoms


@dataclass
class Line:
    runs: list = field(default_factory=list)  # [(x, text, style)]
    width: float = 0.0


def _split_atom(atom: Atom, maxw: float, size: float):
    """Режет слишком длинный атом по символам на куски ≤ maxw."""
    pieces, cur, w = [], [], 0.0
    for text, style in atom.parts:
        for c in text:
            cw = text_w(c, style, size)
            if cur and w + cw > maxw:
                pieces.append(cur)
                cur, w = [], 0.0
            if cur and cur[-1][1] == style:
                cur[-1] = (cur[-1][0] + c, style)
            else:
                cur.append((c, style))
            w += cw
    if cur:
        pieces.append(cur)
    return [Atom(p) for p in pieces]


def _space_style(prev, nxt):
    """Пробел между разными стилями — пропорциональный, между двумя mono — моноширинный."""
    return prev if prev == nxt else ("sans", prev[1], prev[2])


def wrap(atoms, width: float, size: float) -> list[Line]:
    lines: list[Line] = []
    cur = Line()

    def add(atom: Atom, space: bool):
        x = cur.width
        if space and cur.runs:
            x += text_w(" ", _space_style(cur.runs[-1][2], atom.parts[0][1]), size)
        first = True
        for text, style in atom.parts:
            if cur.runs and cur.runs[-1][2] == style:
                px, ptxt, pst = cur.runs[-1]
                gap = " " if (first and space) else ""
                cur.runs[-1] = (px, ptxt + gap + text, pst)
            else:
                cur.runs.append((x, text, style))
            x += text_w(text, style, size)
            first = False
        cur.width = x

    for a in atoms:
        if a.br:
            lines.append(cur)
            cur = Line()
            continue
        aw = a.width(size)
        sp = text_w(" ", _space_style(cur.runs[-1][2], a.parts[0][1]), size) if (a.space and cur.runs) else 0.0
        if cur.runs and cur.width + sp + aw > width:
            lines.append(cur)
            cur = Line()
            sp = 0.0
        if not cur.runs and aw > width:
            pieces = _split_atom(a, width, size)
            for p in pieces[:-1]:
                add(p, False)
                lines.append(cur)
                cur = Line()
            add(pieces[-1], False)
        else:
            add(a, bool(sp))
    if cur.runs or not lines:
        lines.append(cur)
    return lines


# ------------------------------------------------------------------ страницы и операции
class Doc:
    def __init__(self):
        self.pages: list[list] = []
        self.y = 0.0
        self.pending_marker = None  # (x, text, style, size)
        self.headings: list = []  # (level, text, page_idx)
        self.new_page()

    # --- базовое
    def new_page(self):
        self.pages.append([])
        self.y = MARGIN_T

    @property
    def ops(self):
        return self.pages[-1]

    def room(self):
        return BOTTOM - self.y

    def ensure(self, h):
        if self.y + h > BOTTOM and self.y > MARGIN_T + 1:
            self.new_page()

    def space(self, h):
        if self.y > MARGIN_T + 0.5:
            self.y = min(self.y + h, BOTTOM)

    def text(self, x, y, s, style, size, color=C_TEXT):
        self.ops.append(("text", x, y, s, style, size, color))

    def rect(self, x, y, w, h, fc="none", ec="none", lw=0.0):
        self.ops.append(("rect", x, y, w, h, fc, ec, lw))

    def line(self, x0, y0, x1, y1, color=C_RULE, lw=0.6, dash=None):
        self.ops.append(("line", x0, y0, x1, y1, color, lw, dash))

    def image(self, x, y, w, h, path):
        self.ops.append(("image", x, y, w, h, path))

    # --- строки текста
    def emit_line(self, ln: Line, x0, size, lh, color=C_TEXT, code_bg=True):
        self.ensure(lh)
        base = self.y + size * 1.0
        if self.pending_marker:
            mx, mt, mst, msz = self.pending_marker
            self.text(mx, base, mt, mst, msz, color)
            self.pending_marker = None
        for x, t, st in ln.runs:
            if st[0] == "mono" and code_bg and t.strip():
                w = text_w(t, st, size)
                self.rect(x0 + x - 1.0, base - size * 0.86, w + 2.0, size * 1.12, fc=C_INLINE_CODE_BG)
            self.text(x0 + x, base, t, st, size, color)
        self.y += lh

    def flush_marker(self, size=BODY):
        if self.pending_marker:
            self.ensure(size * BODY_LH)
            mx, mt, mst, msz = self.pending_marker
            self.text(mx, self.y + size, mt, mst, msz)
            self.pending_marker = None


# ------------------------------------------------------------------ рендер блоков
class Renderer:
    def __init__(self, doc: Doc, base_dir: str, skip_first_h1=True, alt_captions=False):
        self.d = doc
        self.alt_captions = alt_captions
        self.base_dir = base_dir
        self.skip_first_h1 = skip_first_h1
        self.title = None

    def blocks(self, nodes, x0, width, tight=False, depth=0):
        for i, n in enumerate(nodes):
            self.block(n, x0, width, tight, depth, last=(i == len(nodes) - 1))

    def block(self, n, x0, width, tight, depth, last):
        t = n.type
        d = self.d
        if t == "heading":
            self.heading(n, x0, width)
        elif t == "paragraph":
            inl = n.children[0] if n.children else None
            kids = [c for c in (inl.children if inl else []) if not (c.type == "text" and not c.content.strip())]
            if kids and all(c.type in ("image", "softbreak", "hardbreak") for c in kids):
                d.flush_marker()
                for c in kids:
                    if c.type == "image":
                        self.image(c.attrs.get("src", ""), c.attrs.get("alt", "") or c.content, x0, width)
                return
            self.paragraph(inl, x0, width)
            gap = BODY * BODY_LH * (0.12 if tight else 0.5)
            if not (tight and last):
                d.space(gap)
        elif t in ("bullet_list", "ordered_list"):
            self.lst(n, x0, width, depth)
            if not tight:
                d.space(BODY * 0.5)
        elif t in ("fence", "code_block"):
            d.flush_marker()
            self.code(n.content, x0, width)
        elif t == "table":
            d.flush_marker()
            self.table(n, x0, width)
        elif t == "hr":
            d.ensure(14)
            d.space(4)
            d.line(x0, d.y + 3, x0 + width, d.y + 3, color=C_GRID, lw=0.5)
            d.y += 10
        elif t == "blockquote":
            y_start, p_start = d.y, len(d.pages)
            self.blocks(n.children, x0 + 12, width - 12, tight, depth)
            if len(d.pages) == p_start:
                d.line(x0 + 3, y_start, x0 + 3, d.y - 4, color=C_GRID, lw=2)
        elif t == "html_block":
            atoms = chunks_to_atoms([(n.content.strip(), MONO)])
            for ln in wrap(atoms, width, BODY):
                d.emit_line(ln, x0, BODY, BODY * BODY_LH, code_bg=False)
        else:
            if n.children:
                self.blocks(n.children, x0, width, tight, depth)

    # --- заголовки
    def heading(self, n, x0, width):
        d = self.d
        lvl = int(n.tag[1])
        inl = n.children[0]
        plain = "".join(c.content for c in inl.walk() if c.type in ("text", "code_inline"))
        if lvl == 1 and self.skip_first_h1 and self.title is None:
            self.title = plain
            return
        size = H_SIZES[lvl]
        lh = size * 1.25
        before = {1: 18, 2: 16, 3: 11, 4: 8}.get(lvl, 6)
        chunks = [(t, (st[0], True, st[2]) if st else st) for t, st in inline_chunks(inl)]
        lines = wrap(chunks_to_atoms(chunks), width, size)
        need = before + len(lines) * lh + 8 + 4 * BODY * BODY_LH
        if d.y + need > BOTTOM:
            d.new_page()
        else:
            d.space(before)
        if lvl <= 3:
            d.headings.append((lvl, plain, len(d.pages) - 1))
        color = C_HEAD if lvl <= 3 else C_TEXT
        for ln in lines:
            d.emit_line(ln, x0, size, lh, color=color, code_bg=False)
        if lvl <= 2:
            d.line(x0, d.y + 1, x0 + width, d.y + 1, color=C_RULE, lw=0.8 if lvl == 2 else 1.2)
            d.y += 7
        else:
            d.y += 3

    # --- абзац
    def paragraph(self, inl, x0, width, size=BODY):
        if inl is None:
            return
        atoms = chunks_to_atoms(inline_chunks(inl))
        lines = wrap(atoms, width, size)
        lh = size * BODY_LH
        # не оставляем одну строку внизу страницы (сироты)
        if len(lines) >= 2 and self.d.y + 2 * lh > BOTTOM:
            self.d.new_page()
        for ln in lines:
            self.d.emit_line(ln, x0, size, lh)

    # --- списки
    def lst(self, n, x0, width, depth):
        d = self.d
        ordered = n.type == "ordered_list"
        num = int(n.attrs.get("start", 1)) if ordered else 0
        indent = 16 if not ordered else 20
        for item in n.children:
            if ordered:
                mark = f"{num}."
                num += 1
                mst = NORMAL
            else:
                mark = "•" if depth % 2 == 0 else "–"
                mst = NORMAL
            mw = text_w(mark, mst, BODY)
            d.pending_marker = (x0 + indent - mw - 4, mark, mst, BODY)
            tight = bool(item.children) and all(
                getattr(c.nester_tokens.opening, "hidden", False) if c.type == "paragraph" else True
                for c in item.children)
            self.blocks(item.children, x0 + indent, width - indent, tight=tight, depth=depth + 1)
            d.flush_marker()
            d.space(BODY * BODY_LH * 0.15)

    # --- код
    def code(self, text, x0, width):
        d = self.d
        lines = text.rstrip("\n").split("\n")
        pad = 5.0
        avail = width - 2 * pad
        maxw100 = max((_w100(l.expandtabs(4), MONO) for l in lines), default=0.0)
        size = CODE_MAX
        # схемы псевдографикой уменьшаем до влезания; обычный код — не мельче 7 pt, дальше перенос
        is_diagram = any(0x2500 <= ord(c) <= 0x257F for c in text)
        min_size = CODE_MIN if is_diagram else 7.0
        if maxw100 > 0:
            size = max(min_size, min(CODE_MAX, avail * 100.0 / maxw100))
        lh = size * 1.3
        out = []
        for l in lines:
            l = l.expandtabs(4)
            if text_w(l, MONO, size) <= avail:
                out.append(l)
            else:  # перенос по символам с отступом-маркером
                cur = ""
                for c in l:
                    if text_w(cur + c, MONO, size) > avail:
                        out.append(cur)
                        cur = "  " + c
                    else:
                        cur += c
                out.append(cur)
        d.space(2)
        i = 0
        if d.room() < min(len(out), 4) * lh + 2 * pad:
            d.new_page()
        while i < len(out):
            n_fit = max(1, int((d.room() - 2 * pad) // lh))
            chunk = out[i:i + n_fit]
            h = len(chunk) * lh + 2 * pad
            d.rect(x0, d.y, width, h, fc=C_CODE_BG, ec=C_CODE_BORDER, lw=0.5)
            y = d.y + pad
            for l in chunk:
                d.text(x0 + pad, y + size * 0.95, l, MONO, size)
                y += lh
            d.y += h
            i += len(chunk)
            if i < len(out):
                d.new_page()
        d.space(BODY * 0.8)

    # --- картинка
    def image(self, src, alt, x0, width):
        d = self.d
        path = src if os.path.isabs(src) else os.path.join(self.base_dir, src)
        if not os.path.exists(path):
            atoms = chunks_to_atoms([(f"[нет файла картинки: {src}]", ITALIC)])
            for ln in wrap(atoms, width, BODY):
                d.emit_line(ln, x0, BODY, BODY * BODY_LH, color=C_MUTED)
            return
        with Image.open(path) as im:
            pw, ph = im.size
        w = min(width, pw / IMG_DPI_GUESS * 72.0 * 1.15)
        h = w * ph / pw
        if not self.alt_captions:
            alt = ""
        cap_h = BODY * 1.6 if alt else 0.0
        max_h = (BOTTOM - MARGIN_T) - cap_h - 10
        if h > max_h:
            h = max_h
            w = h * pw / ph
        d.space(4)
        if d.y + h + cap_h > BOTTOM:
            d.new_page()
        x = x0 + (width - w) / 2
        d.image(x, d.y, w, h, path)
        d.y += h + 3
        if alt:
            sz = BODY * 0.85
            cw = text_w(alt, ITALIC, sz)
            d.text(x0 + (width - cw) / 2, d.y + sz, alt, ITALIC, sz, C_MUTED)
            d.y += cap_h
        d.space(BODY * 0.6)

    # --- таблица
    def table(self, n, x0, width):
        d = self.d
        head_rows, body_rows, aligns = [], [], []
        for sec in n.children:
            for tr in sec.children:
                cells = []
                for td in tr.children:
                    inl = td.children[0] if td.children else None
                    chunks = inline_chunks(inl) if inl else []
                    if td.type == "th":
                        chunks = [(t, (st[0], True, st[2]) if st else st) for t, st in chunks]
                    cells.append(chunks_to_atoms(chunks))
                    if sec.type == "thead":
                        style = td.attrs.get("style", "") or ""
                        aligns.append("right" if "right" in style else "center" if "center" in style else "left")
                (head_rows if sec.type == "thead" else body_rows).append(cells)
        ncol = max(len(r) for r in head_rows + body_rows)
        for r in head_rows + body_rows:
            while len(r) < ncol:
                r.append([])
        aligns += ["left"] * (ncol - len(aligns))
        pad_x, pad_y = 3.2, 2.4

        def natural(size):
            mx, mn = [0.0] * ncol, [0.0] * ncol
            for r in head_rows + body_rows:
                for j, atoms in enumerate(r):
                    tot = 0.0
                    for k, a in enumerate(atoms):
                        aw = a.width(size)
                        if a.space and k:
                            tot += text_w(" ", _space_style(atoms[k - 1].parts[-1][1], a.parts[0][1]), size)
                        tot += aw
                        mn[j] = max(mn[j], min(aw, width * 0.35))
                    mx[j] = max(mx[j], tot)
            mx = [v + 2 * pad_x + 0.5 for v in mx]
            mn = [v + 2 * pad_x + 0.5 for v in mn]
            return mx, mn

        size = TABLE_SIZES[-1]
        for s in TABLE_SIZES:
            mx, mn = natural(s)
            if sum(mn) <= width:
                size = s
                break
        mx, mn = natural(size)
        if sum(mx) <= width:
            widths = mx
        elif sum(mn) >= width:
            widths = [v * width / sum(mn) for v in mn]
        else:
            extra = width - sum(mn)
            flex = [a - b for a, b in zip(mx, mn)]
            # sqrt-смягчение: узкие колонки не сжимаются в «столбик» слов
            import math
            wts = [math.sqrt(f) * math.sqrt(max(a, 1)) for f, a in zip(flex, mx)]
            tw = sum(wts) or 1.0
            widths = [b + extra * w_ / tw for b, w_ in zip(mn, wts)]
            # не выдаём колонке больше её натуральной ширины — излишек перераспределяем
            for _ in range(5):
                over = sum(max(0.0, w_ - a) for w_, a in zip(widths, mx))
                if over < 0.5:
                    break
                widths = [min(w_, a) for w_, a in zip(widths, mx)]
                need = [j for j in range(ncol) if widths[j] < mx[j] - 0.5]
                tw = sum(wts[j] for j in need) or 1.0
                for j in need:
                    widths[j] += over * wts[j] / tw
        lh = size * 1.3

        def layout_row(r):
            cells = [wrap(atoms, widths[j] - 2 * pad_x, size) for j, atoms in enumerate(r)]
            h = max(len(c) for c in cells) * lh + 2 * pad_y
            return cells, h

        heads = [layout_row(r) for r in head_rows]
        bodies = [layout_row(r) for r in body_rows]
        head_h = sum(h for _, h in heads)

        def draw_row(cells, h, bg, color=C_TEXT):
            x = x0
            if bg:
                d.rect(x0, d.y, sum(widths), h, fc=bg)
            for j, lines in enumerate(cells):
                y = d.y + pad_y
                for ln in lines:
                    if aligns[j] == "right":
                        off = widths[j] - 2 * pad_x - ln.width
                    elif aligns[j] == "center":
                        off = (widths[j] - 2 * pad_x - ln.width) / 2
                    else:
                        off = 0.0
                    base = y + size * 0.98
                    for rx, t, st in ln.runs:
                        d.text(x + pad_x + off + rx, base, t, st, size, color)
                    y += lh
                d.line(x, d.y, x, d.y + h, color=C_GRID, lw=0.4)
                x += widths[j]
            d.line(x, d.y, x, d.y + h, color=C_GRID, lw=0.4)
            d.line(x0, d.y + h, x, d.y + h, color=C_GRID, lw=0.4)
            d.y += h

        def draw_head():
            d.line(x0, d.y, x0 + sum(widths), d.y, color=C_GRID, lw=0.4)
            for cells, h in heads:
                draw_row(cells, h, C_TH_BG)

        d.space(2)
        first_h = bodies[0][1] if bodies else 0
        if d.y + head_h + first_h > BOTTOM:
            d.new_page()
        draw_head()
        for i, (cells, h) in enumerate(bodies):
            if d.y + h > BOTTOM:
                d.new_page()
                draw_head()
            draw_row(cells, h, C_ZEBRA if i % 2 else None)
        d.space(BODY * 0.9)


# ------------------------------------------------------------------ титул и оглавление
def build_front(title: str, subtitle: str | None, headings, offset: int, toc=True, k=1.0) -> Doc:
    """Титул + оглавление; k < 1 — сжатие интерлиньяжа оглавления, чтобы влезть на страницу."""
    f = Doc()
    f.y = MARGIN_T + 30 * k
    size = 20.0
    lines = wrap(chunks_to_atoms([(title, BOLD)]), TEXT_W, size)
    for ln in lines:
        f.emit_line(ln, MARGIN_L, size, size * 1.3, color=C_HEAD, code_bg=False)
    f.y += 6
    if subtitle:
        f.emit_line(wrap(chunks_to_atoms([(subtitle, ("sans", False, False))]), TEXT_W, 13)[0],
                    MARGIN_L, 13, 18, color=C_TEXT)
    f.y += 4
    f.line(MARGIN_L, f.y + 2, MARGIN_L + TEXT_W, f.y + 2, color=C_HEAD, lw=1.4)
    f.y += 24
    if not toc or not headings:
        return f
    f.emit_line(wrap(chunks_to_atoms([("Содержание", BOLD)]), TEXT_W, 13)[0], MARGIN_L, 13, 20,
                color=C_HEAD, code_bg=False)
    f.y += 4
    for lvl, text, pidx in headings:
        size = 10.0 if lvl == 2 else 8.8
        style = BOLD if lvl == 2 else NORMAL
        indent = 0.0 if lvl == 2 else 16.0
        lh = size * (1.55 if lvl == 2 else 1.4) * k
        if lvl == 2:
            f.ensure(lh + 3 * k)
            f.y += 3 * k
        num = str(pidx + 1 + offset)
        nw = text_w(num, style, size)
        avail = TEXT_W - indent - nw - 24
        lines = wrap(chunks_to_atoms([(text, style)]), avail, size)
        f.ensure(lh * len(lines))
        for li, ln in enumerate(lines):
            if li == len(lines) - 1:
                base = f.y + size
                x_end = MARGIN_L + indent + ln.width + 4
                x_num = MARGIN_L + TEXT_W - nw
                f.line(x_end, base, x_num - 4, base, color="#9a9a9a", lw=0.7, dash=(0, (0.6, 2.6)))
                f.text(x_num, base, num, style, size)
            f.emit_line(ln, MARGIN_L + indent, size, lh, code_bg=False)
    return f


# ------------------------------------------------------------------ вывод в PDF
def render_pdf(pages, out_path, footer_title="", first_numbered=1, meta_title=""):
    total = len(pages)
    with PdfPages(out_path, metadata={"Title": meta_title, "Creator": "src.scripts.md2pdf"}) as pdf:
        for pi, ops in enumerate(pages):
            fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72), dpi=72)
            for op in ops:
                kind = op[0]
                if kind == "text":
                    _, x, y, s, st, size, color = op
                    if not s:
                        continue
                    fig.text(x / PAGE_W, 1 - y / PAGE_H, s, fontproperties=fprop(st), fontsize=size,
                             color=color, va="baseline", ha="left", parse_math=False)
                elif kind == "rect":
                    _, x, y, w, h, fc, ec, lw = op
                    fig.add_artist(Rectangle((x / PAGE_W, 1 - (y + h) / PAGE_H), w / PAGE_W, h / PAGE_H,
                                             transform=fig.transFigure, facecolor=fc, edgecolor=ec,
                                             linewidth=lw, zorder=-1))
                elif kind == "line":
                    _, x0, y0, x1, y1, color, lw, dash = op
                    ln = Line2D([x0 / PAGE_W, x1 / PAGE_W], [1 - y0 / PAGE_H, 1 - y1 / PAGE_H],
                                transform=fig.transFigure, color=color, linewidth=lw)
                    if dash:
                        ln.set_linestyle(dash)
                    fig.add_artist(ln)
                elif kind == "image":
                    _, x, y, w, h, path = op
                    ax = fig.add_axes([x / PAGE_W, 1 - (y + h) / PAGE_H, w / PAGE_W, h / PAGE_H])
                    ax.imshow(plt.imread(path), interpolation="none", aspect="auto")
                    ax.set_axis_off()
            if pi + 1 >= first_numbered + 1:
                num = f"{pi + 1} / {total}"
                nw = text_w(num, NORMAL, 8.5)
                fig.text((PAGE_W - nw) / 2 / PAGE_W, 28 / PAGE_H, num, fontproperties=fprop(NORMAL),
                         fontsize=8.5, color=C_MUTED, parse_math=False)
                if footer_title:
                    ft = footer_title if len(footer_title) < 90 else footer_title[:87] + "…"
                    fig.text(MARGIN_L / PAGE_W, 1 - 32 / PAGE_H, ft, fontproperties=fprop(NORMAL),
                             fontsize=7.5, color=C_MUTED, parse_math=False)
                    fig.add_artist(Line2D([MARGIN_L / PAGE_W, 1 - MARGIN_R / PAGE_W],
                                          [1 - 37 / PAGE_H] * 2, transform=fig.transFigure,
                                          color=C_GRID, linewidth=0.4))
            pdf.savefig(fig)
            plt.close(fig)


def convert(md_path: str, out_path: str, subtitle: str | None = "Команда HORMIGAS", toc: bool = True):
    with open(md_path, encoding="utf-8") as fh:
        src = fh.read()
    md = MarkdownIt("commonmark").enable("table").enable("strikethrough")
    tree = SyntaxTreeNode(md.parse(src))
    body = Doc()
    r = Renderer(body, os.path.dirname(os.path.abspath(md_path)))
    r.blocks(tree.children, MARGIN_L, TEXT_W)
    if not body.pages[-1]:
        body.pages.pop()
    title = r.title or os.path.splitext(os.path.basename(md_path))[0]
    heads = [h for h in body.headings if h[0] == 2 or (h[0] == 3)]
    k = 1.0
    front = build_front(title, subtitle, heads, 0, toc, k)
    while len(front.pages) > 1 and k > 0.8:
        k -= 0.05
        front = build_front(title, subtitle, heads, 0, toc, k)
    front = build_front(title, subtitle, heads, len(front.pages), toc, k)
    pages = front.pages + body.pages
    render_pdf(pages, out_path, footer_title=title, first_numbered=1, meta_title=title)
    return len(pages)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--team", default="Команда HORMIGAS", help="подзаголовок титула ('' — без него)")
    ap.add_argument("--no-toc", action="store_true")
    a = ap.parse_args()
    n = convert(a.inp, a.out, a.team or None, not a.no_toc)
    print(f"{a.out}: {n} страниц")


if __name__ == "__main__":
    main()
