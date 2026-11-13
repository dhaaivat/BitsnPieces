"""Draw the 'how a number is stored' figure (fp32 / fp16 / bf16) as SVG. Stdlib only.

Usage:
    python3 docs/figures/number_formats.py number_formats.svg
    rsvg-convert -z 1.5 number_formats.svg -o docs/figures/number_formats.png

Every value is computed from the bit patterns and checked against the notebook's Part A output.
"""
import struct, math, sys
from html import escape

# ---------- encoding (matches torch's round-to-nearest-even) ----------
def fp32_bits(x):
    return struct.unpack(">I", struct.pack(">f", x))[0], 32

def fp16_bits(x):
    try:
        return struct.unpack(">H", struct.pack(">e", x))[0], 16
    except OverflowError:
        return (0x7C00 if x > 0 else 0xFC00), 16

def bf16_bits(x):
    u = struct.unpack(">I", struct.pack(">f", x))[0]
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF, 16

FMT = {  # name: (encoder, exp bits, mantissa bits, bias)
    "fp32": (fp32_bits, 8, 23, 127),
    "fp16": (fp16_bits, 5, 10, 15),
    "bf16": (bf16_bits, 8, 7, 127),
}

def g(v, n=10):
    return "inf" if math.isinf(v) else f"{v:.{n}g}"

def sup(n):
    return str(n).replace("-", "−")

def decode(x, fmt):
    """Return bit string, fields, stored value and a list of calculation lines."""
    enc, e, m, bias = FMT[fmt]
    u, n = enc(x)
    b = format(u, f"0{n}b")
    s, E, M = int(b[0]), int(b[1:1 + e], 2), int(b[1 + e:], 2)
    if E == (1 << e) - 1:
        val = math.inf
        mx = (2 - 2 ** -m) * 2 ** ((1 << e) - 2 - bias)
        calc = [f"exponent = all 1s (= {E})  →  reserved for inf / NaN  →  stored as inf",
                f"largest finite {fmt} = 2^{(1 << e) - 2 - bias} × (2 − 2^−{m}) = {g(mx)}  <  {g(x)}  → OVERFLOW"]
    elif E == 0:
        val = M * 2.0 ** (1 - bias - m)
        if M == 0:
            calc = [f"exponent = 0 and mantissa = 0  →  stored as 0",
                    f"smallest {fmt} subnormal = 2^{sup(1 - bias - m)} ≈ {g(2.0 ** (1 - bias - m), 3)};  "
                    f"{g(x, 3)} is below half of it → UNDERFLOW (rounds to 0)"]
        else:
            calc = [f"exponent = 0  →  SUBNORMAL: no hidden 1, fixed scale 2^{sup(1 - bias)}",
                    f"2^{sup(1 - bias)} × ({M} / 2^{m}) = {M} × 2^{sup(1 - bias - m)} = {g(val)}"
                    f"   (only {M.bit_length()} significant bits left)"]
    else:
        frac = 1 + M / 2 ** m
        val = (-1) ** s * 2.0 ** (E - bias) * frac
        calc = [f"2^({E} − {bias}) × (1 + {M} / 2^{m})  =  2^{sup(E - bias)} × {frac:.10g}  =  {g(val)}"]
    return b, (s, E, M), val, calc

# Check against the notebook output (Part A) before drawing anything.
EXPECTED = {
    0.1: ("0.1000000015", "0.09997558594", "0.1000976562"),
    1 / 3: ("0.3333333433", "0.3332519531", "0.333984375"),
    3.14159265: ("3.141592741", "3.140625", "3.140625"),
    70000.0: ("70000", "inf", "70144"),
    1e-6: ("9.999999975e-07", "1.013278961e-06", "9.983778e-07"),
    1e-8: ("9.999999939e-09", "0", "1.001171768e-08"),
}
for x, exp in EXPECTED.items():
    got = tuple(g(decode(x, f)[2]) for f in ("fp32", "fp16", "bf16"))
    assert got == exp, (x, got, exp)

# ---------- drawing ----------
W = 1400
C_S, C_E, C_M = "#E45756", "#4C78A8", "#54A24B"         # sign / exponent / mantissa
INK, MUTED, RULE, PANEL = "#1F2328", "#57606A", "#D0D7DE", "#F6F8FA"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
SANS = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
out = []

def text(x, y, s, size=15, color=INK, weight="normal", font=SANS, anchor="start"):
    out.append(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" fill="{color}" '
               f'font-weight="{weight}" text-anchor="{anchor}" xml:space="preserve">{escape(s)}</text>')

def rect(x, y, w, h, fill, stroke="none", rx=0):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" rx="{rx}"/>')

CELL, CH, X0 = 30, 32, 270

def bit_row(y, bits, e):
    for i, ch in enumerate(bits):
        col = C_S if i == 0 else C_E if i <= e else C_M
        x = X0 + i * CELL
        rect(x, y, CELL - 2, CH, col, rx=3)
        text(x + (CELL - 2) / 2, y + 22, ch, 16, "#FFFFFF", "bold", MONO, "middle")

def fmt_row(y, x, fmt, note=None):
    enc, e, m, bias = FMT[fmt]
    b, (s, E, M), val, calc = decode(x, fmt)
    text(40, y + 15, fmt.upper(), 20, INK, "bold")
    text(40, y + 34, f"1 · {e} · {m} bits, bias {bias}", 13, MUTED)
    bit_row(y, b, e)
    if note:
        text(X0 + len(b) * CELL + 16, y + 22, note[0], 15, note[1], "bold")
    text(X0, y + CH + 22, f"sign {s}   exponent {b[1:1 + e]} = {E}   mantissa {b[1 + e:]} = {M}", 13, MUTED, font=MONO)
    for k, line in enumerate(calc):
        text(X0, y + CH + 44 + 20 * k, ("→ " if k == 0 else "  ") + line, 15, INK, font=MONO)
    return y + CH + 44 + 20 * len(calc) + 18

def section(y, title, sub=None):
    out.append(f'<line x1="40" y1="{y}" x2="{W - 40}" y2="{y}" stroke="{RULE}"/>')
    text(40, y + 34, title, 21, INK, "bold")
    if sub: text(40, y + 58, sub, 14, MUTED)
    return y + (76 if sub else 56)

# Header
text(40, 52, "How a number is stored: FP32 vs FP16 vs BF16", 30, INK, "bold")
text(40, 82, "Same value, three formats: the bits, the formula, and the value that actually gets stored.  "
             "(BitsnPieces, Part A)", 15, MUTED)
rect(40, 102, W - 80, 78, PANEL, RULE, 8)
text(60, 132, "normal:      value = (−1)^sign × 2^(exponent − bias) × (1 + mantissa / 2^m)", 17, INK, font=MONO)
text(60, 162, "subnormal:   value = (−1)^sign × 2^(1 − bias) × (mantissa / 2^m)        "
              "(exponent bits all 0)", 17, INK, font=MONO)
lx = 40
for lab, col in [("sign", C_S), ("exponent: range", C_E), ("mantissa: precision", C_M)]:
    rect(lx, 198, 18, 18, col, rx=3); text(lx + 26, 212, lab, 14, MUTED); lx += 200

# 1. Worked example
y = section(236, "1 · Worked example: x = 0.1",
            "All three land on the same power of two (2^−4 = 1/16). Only the number of mantissa bits differs.")
y = fmt_row(y, 0.1, "fp32", ("error ≈ 1.5e−9", MUTED))
y = fmt_row(y, 0.1, "fp16", ("error ≈ −2.4e−5", MUTED))
y = fmt_row(y, 0.1, "bf16", ("error ≈ +9.8e−5", MUTED))

# 2. Edge cases
y = section(y + 4, "2 · Where the formats break: range (exponent bits) vs precision (mantissa bits)",
            "fp16 has only 5 exponent bits, so it fails at the extremes. bf16 keeps fp32's 8, so it never runs "
            "out of range, but it is coarse everywhere.")
for x, label, n16 in [(70000.0, "x = 70000", ("OVERFLOW → inf", C_S)),
                      (1e-6, "x = 1e−6", ("SUBNORMAL", "#B07A00")),
                      (1e-8, "x = 1e−8", ("UNDERFLOW → 0", C_S))]:
    y += 14; text(40, y + 2, label, 18, INK, "bold"); y += 22
    y = fmt_row(y, x, "fp16", n16)
    nb = {70000.0: ("70000 → 70144", MUTED), 1e-6: ("ok", C_M), 1e-8: ("ok", C_M)}[x]
    y = fmt_row(y, x, "bf16", nb)

# 3. Tables
y = section(y + 4, "3 · Stored values and format limits")
cols = [40, 300, 560, 820]
text(cols[0], y, "x", 14, MUTED, "bold")
for c, f in zip(cols[1:], ["fp32", "fp16", "bf16"]):
    text(c, y, f + " stores", 14, MUTED, "bold")
y += 10
for x, lab in [(0.1, "0.1"), (1 / 3, "1/3"), (3.14159265, "π"), (70000.0, "70000"), (1e-6, "1e−6"), (1e-8, "1e−8")]:
    y += 28
    text(cols[0], y, lab, 15, INK, font=MONO)
    for c, f in zip(cols[1:], ["fp32", "fp16", "bf16"]):
        v = decode(x, f)[2]
        bad = math.isinf(v) or v == 0
        text(c, y, g(v), 15, C_S if bad else INK, "bold" if bad else "normal", MONO)
text(1080, y - 140, "π: fp16 and bf16 both store 3.140625.", 14, MUTED)
text(1080, y - 118, "More mantissa bits don't help", 14, MUTED)
text(1080, y - 96, "every number equally.", 14, MUTED)

y += 46
heads = ["format", "max = 2^(emax) × (2 − 2^−m)", "smallest normal = 2^(1 − bias)", "eps = 2^−m (gap after 1.0)", "bytes"]
hx = [40, 200, 560, 900, 1230]
for hxi, h in zip(hx, heads): text(hxi, y, h, 14, MUTED, "bold")
for f, mx, mn, ep, by in [
    ("fp32", "2^127 × (2 − 2^−23) ≈ 3.40e38", "2^−126 ≈ 1.18e−38", "2^−23 ≈ 1.19e−7", "4"),
    ("fp16", "2^15 × (2 − 2^−10) = 65504", "2^−14 ≈ 6.10e−5", "2^−10 ≈ 9.77e−4", "2"),
    ("bf16", "2^127 × (2 − 2^−7) ≈ 3.39e38", "2^−126 ≈ 1.18e−38", "2^−7 ≈ 7.81e−3", "2"),
]:
    y += 28
    for hxi, v in zip(hx, [f.upper(), mx, mn, ep, by]):
        text(hxi, y, v, 15, INK, "bold" if hxi == 40 else "normal", MONO if hxi != 40 else SANS)

y += 44
rect(40, y - 26, W - 80, 40, PANEL, RULE, 8)
text(60, y, "Exponent bits → range (fp16 overflows past 65,504).   Mantissa bits → precision "
            "(bf16's steps are 8× coarser than fp16's).   bf16 = fp32 with the last 16 bits cut off.",
     15, INK)
H = y + 40

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
       f'<rect width="100%" height="100%" fill="#FFFFFF"/>' + "".join(out) + "</svg>")
open(sys.argv[1], "w").write(svg)
print("ok", W, H)
