"""
Gera os gráficos da apresentação (SVG) a partir do CSV, usando a mesma limpeza de analise.py.

Uso (rodar DEPOIS de analise.py, na mesma pasta):
    python graficos.py "Práticas de desenvolvimento de software com e sem IA.csv"

Saídas em saida/figuras: likert.svg, heatmap.svg, dotplot.svg, forest.svg
"""
import io, sys, contextlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analise as A

OUT = Path("saida") / "figuras"; OUT.mkdir(parents=True, exist_ok=True)
INK, BODY, MUTED, LINE, PAPER = "#14213D", "#4A5568", "#6B7280", "#DAD7CE", "#F7F6F2"
ORANGE, BLUE = "#C2551A", "#3B5B92"
FONT = "Inter, Arial, sans-serif"

with contextlib.redirect_stdout(io.StringIO()):
    df, _ = A.limpar(A.renomear(pd.read_csv(sys.argv[1])))
usu = df[df["U1"] == "Sim"]
raw = usu[A.LIKERT].replace(A.ESCALA).astype(int)          # respostas como marcadas
rec = raw.astype(float).copy()
for it in A.INVERTIDOS: rec[it] = 6 - rec[it]               # recodificadas
anos = usu["D1"].astype(float)
faixa = pd.cut(anos, **A.FAIXAS)
n = len(usu)

def t(x, y, s, size=22, fill=BODY, anchor="start", weight=400, extra=""):
    s = str(s).replace("&", "&amp;").replace("<", "&lt;")
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" {extra}>{s}</text>'

def svg(w, h, body, label):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" aria-label="{label}">' + "".join(body) + "</svg>"

# ---------------------------------------------------------------- 1. Likert divergente
ROT = {
    "P1": "P1 Concluo tarefas mais rápido", "P2": "P2 Reduz código repetitivo",
    "P3": "P3 Mantenho a concentração", "P4": "P4 Perco tempo corrigindo (inv.)",
    "C1": "C1 Confio que o código está correto", "C2": "C2 Aceito sem modificar",
    "C3": "C3 Desconfio em trechos críticos (inv.)", "V1": "V1 Leio linha a linha",
    "V2": "V2 Escrevo ou executo testes", "V3": "V3 Submeto à revisão de código",
}
CORES = {1: BLUE, 2: "#9DB0D1", 3: "#D9D6CE", 4: "#E8A27B", 5: ORANGE}
W, H = 1664, 640
_pc = {it: [(raw[it] == v).mean() * 100 for v in range(1, 6)] for it in A.LIKERT}
_L = max(p[0] + p[1] + p[2] / 2 for p in _pc.values()) + 7   # % máximo à esquerda (+ rótulo)
_R = max(p[3] + p[4] + p[2] / 2 for p in _pc.values()) + 10
x_ini, x_fim = 500, W
esc = (x_fim - x_ini) / (_L + _R)
cx = x_ini + _L * esc
b = []
rowh, top = 52, 70
grupos = [("Produtividade", ["P1", "P2", "P3", "P4"]), ("Confiança", ["C1", "C2", "C3"]), ("Verificação", ["V1", "V2", "V3"])]
y = top
for gname, itens in grupos:
    b.append(t(0, y + 6, gname.upper(), 18, ORANGE, weight=600, extra='letter-spacing="2"'))
    y += 18
    for it in itens:
        pct = {v: (raw[it] == v).mean() * 100 for v in range(1, 6)}
        x = cx - (pct[1] + pct[2] + pct[3] / 2) * esc
        for v in range(1, 6):
            w = pct[v] * esc
            if w > 0:
                b.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{rowh-14}" fill="{CORES[v]}"/>')
            x += w
        yc = y + (rowh - 14) / 2 + 8
        b.append(t(x_ini - 20, yc, ROT[it], 22, INK, "end"))
        neg, pos = pct[1] + pct[2], pct[4] + pct[5]
        left_edge = cx - (pct[1] + pct[2] + pct[3] / 2) * esc
        right_edge = cx + (pct[4] + pct[5] + pct[3] / 2) * esc
        b.append(t(left_edge - 10, yc, f"{neg:.0f}%", 20, BLUE, "end", 600))
        b.append(t(right_edge + 10, yc, f"{pos:.0f}%", 20, ORANGE, "start", 600))
        y += rowh
    y += 10
b.append(f'<line x1="{cx}" y1="{top-10}" x2="{cx}" y2="{y-14}" stroke="{INK}" stroke-width="2"/>')
# legenda
leg = [(1, "Discordo totalmente / Nunca"), (2, "Discordo / Raramente"), (3, "Neutro / Às vezes"), (4, "Concordo / Frequentemente"), (5, "Concordo totalmente / Sempre")]
lxp = 0
for v, s in leg:
    b.append(f'<rect x="{lxp}" y="8" width="22" height="22" fill="{CORES[v]}"/>')
    b.append(t(lxp + 30, 27, s, 19, BODY))
    lxp += 30 + len(s) * 10.4 + 34
H = int(y + 4)
(OUT / "likert.svg").write_text(svg(W, H, b, "Barras divergentes com a distribuição das respostas dos 10 itens Likert, centradas no neutro"))

# ---------------------------------------------------------------- 2. Mapa de calor
items = A.LIKERT
R = rec[items].corr(method="spearman")
cell, ox, oy = 62, 70, 10
b = []
def cor(r):
    a = min(abs(r), 1.0)
    base = np.array([194, 85, 26]) if r > 0 else np.array([59, 91, 146])
    c = (np.array([247, 246, 242]) * (1 - a) + base * a).astype(int)
    return "#%02x%02x%02x" % tuple(c)
for i, a in enumerate(items):
    for j, bb in enumerate(items):
        if j > i: continue
        x, yy = ox + j * cell, oy + i * cell
        r = 1.0 if i == j else R.loc[a, bb]
        fill = "#ECEAE4" if i == j else cor(r)
        b.append(f'<rect x="{x}" y="{yy}" width="{cell-3}" height="{cell-3}" fill="{fill}" rx="4"/>')
        if i != j:
            txt = f"{r:.2f}".replace("-", "−").replace(".", ",")
            b.append(t(x + (cell - 3) / 2, yy + cell / 2 + 6, txt, 18, "#FBF3EC" if abs(r) > 0.45 else INK, "middle", 600 if abs(r) > 0.45 else 400))
    b.append(t(ox - 12, oy + i * cell + cell / 2 + 7, a, 20, INK, "end", 600))
    b.append(t(ox + i * cell + (cell - 3) / 2, oy + len(items) * cell + 24, a, 20, INK, "middle", 600))
# contorno dos blocos de construto
for i0, k in [(0, 4), (4, 3), (7, 3)]:
    b.append(f'<rect x="{ox + i0*cell - 3}" y="{oy + i0*cell - 3}" width="{k*cell+3}" height="{k*cell+3}" fill="none" stroke="{INK}" stroke-width="3" rx="6"/>')
Wh = ox + len(items) * cell + 10; Hh = oy + len(items) * cell + 40
(OUT / "heatmap.svg").write_text(svg(Wh, Hh, b, "Mapa de calor das correlações de Spearman entre os 10 itens, com os blocos de cada construto contornados"))

# ---------------------------------------------------------------- 3. Dot plot PROD × faixa
labs = A.FAIXAS["labels"]
PW, PH, gap = 380, 470, 28
b = []
titulos = {"P1": "P1 · mais rápido", "P2": "P2 · menos repetitivo", "P3": "P3 · concentração", "P4": "P4 · pouco retrabalho (recod.)"}
for k, it in enumerate(["P1", "P2", "P3", "P4"]):
    x0 = k * (PW + gap) + (60 if k == 0 else 0) + (k > 0) * 60
    b.append(t(x0, 26, titulos[it], 24, INK, weight=600))
    y_top, y_bot = 70, 70 + 360
    ystep = (y_bot - y_top) / 4
    colw = (PW - 20) / 4
    for v in range(1, 6):
        yy = y_bot - (v - 1) * ystep
        b.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+PW-20}" y2="{yy:.1f}" stroke="{LINE}" stroke-width="1.5"/>')
        if k == 0: b.append(t(x0 - 14, yy + 7, v, 20, MUTED, "end"))
    for g, lab in enumerate(labs):
        xc = x0 + colw * g + colw / 2
        vals = rec[it][faixa == lab]
        if len(vals):
            md = vals.median(); ym = y_bot - (md - 1) * ystep
            b.append(f'<line x1="{xc-36:.1f}" y1="{ym:.1f}" x2="{xc+36:.1f}" y2="{ym:.1f}" stroke="{ORANGE}" stroke-width="6" stroke-linecap="round"/>')
        for v in range(1, 6):
            c = int((vals == v).sum())
            yy = y_bot - (v - 1) * ystep
            per = 6
            for d in range(c):
                rr, cc = divmod(d, per)
                inrow = min(per, c - rr * per)
                dx = (cc - (inrow - 1) / 2) * 12
                dy = (rr - 0) * -11 + (0 if c <= per else 5)
                b.append(f'<circle cx="{xc+dx:.1f}" cy="{yy+dy:.1f}" r="5" fill="{BLUE}" fill-opacity="0.85"/>')
        b.append(t(xc, y_bot + 40, lab, 20, INK, "middle", 600))
        b.append(t(xc, y_bot + 64, f"n={len(vals)}", 18, MUTED, "middle"))
Wd = 60 + 4 * PW + 3 * gap; Hd = 70 + 360 + 76
(OUT / "dotplot.svg").write_text(svg(Wd, Hd, b, "Gráfico de pontos dos itens P1 a P4 por faixa de experiência, um ponto por respondente e a mediana em laranja"))

# ---------------------------------------------------------------- 4. Forest plot
fam = []
for it in ["P1", "P2", "P3", "P4"]:
    r, p = stats.spearmanr(anos, rec[it]); fam.append(("H2", f"anos × {it}", r, p, len(rec)))
conf = rec[["C1", "C2"]].mean(axis=1)
for it in ["V1", "V2", "V3"]:
    r, p = stats.spearmanr(conf, rec[it]); fam.append(("H3", f"CONF × {it}", r, p, len(rec)))
import pingouin as pg
import json
_fam = {f["teste"]: f["p_holm"] for f in json.load(open(Path("saida") / "resultados.json"))["familia"]}
ph = [_fam[("Spearman " + l.replace("CONF", "CONF (sem C3)"))] for (_, l, *_r) in fam]  # Holm da família de 11 testes
# sensibilidade (escores originais)
conf3, verif3 = rec[["C1", "C2", "C3"]].mean(axis=1), rec[["V1", "V2", "V3"]].mean(axis=1)
rs, ps = stats.spearmanr(conf3, verif3)
def ci(r, n):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
rows = [(h, l, r, p, q, *ci(r, nn), False) for (h, l, r, p, nn), q in zip(fam, ph)]
rows.append(("Sens.", "CONF (c/ C3) × VERIF", rs, ps, None, *ci(rs, n), True))
FW = 1664; ax0, ax1 = 560, 1360
def X(r): return ax0 + (r + 1) / 2 * (ax1 - ax0)
rh, ytop = 58, 70
b = []
thr = np.tanh((stats.norm.ppf(.975) + stats.norm.ppf(.8)) / np.sqrt(n - 3))
yb = ytop + len(rows) * rh + 22
b.append(f'<rect x="{X(-thr):.1f}" y="{ytop-20}" width="{X(thr)-X(-thr):.1f}" height="{yb-ytop+20}" fill="#ECEAE4"/>')
b.append(t((X(-thr) + X(thr)) / 2, ytop - 30, f"|ρ| < {thr:.2f}: fora do alcance com 80% de poder (n = {n})".replace(".", ","), 19, MUTED, "middle"))
for v in [-1, -0.5, 0, 0.5, 1]:
    b.append(f'<line x1="{X(v):.1f}" y1="{ytop-20}" x2="{X(v):.1f}" y2="{yb}" stroke="{INK if v == 0 else LINE}" stroke-width="{2 if v == 0 else 1.5}"/>')
    b.append(t(X(v), yb + 28, f"{v:g}".replace(".", ",").replace("-", "−"), 20, MUTED, "middle"))
b.append(t((ax0 + ax1) / 2, yb + 58, "ρ de Spearman com IC 95%", 20, INK, "middle", 600))
b.append(t(FW, ytop - 30, "p Holm", 19, MUTED, "end", 600))
for i, (h, l, r, p, q, lo, hi, sens) in enumerate(rows):
    yc = ytop + i * rh + rh / 2
    if sens:
        b.append(f'<line x1="0" y1="{yc-rh/2:.1f}" x2="{FW}" y2="{yc-rh/2:.1f}" stroke="{LINE}" stroke-width="2" stroke-dasharray="6 6"/>')
    colr = ORANGE if sens else BLUE
    b.append(t(0, yc + 7, h, 20, ORANGE if sens else MUTED, weight=600))
    b.append(t(90, yc + 7, l, 22, INK))
    b.append(f'<line x1="{X(lo):.1f}" y1="{yc:.1f}" x2="{X(hi):.1f}" y2="{yc:.1f}" stroke="{colr}" stroke-width="4" stroke-linecap="round"/>')
    if sens:
        b.append(f'<rect x="{X(r)-9:.1f}" y="{yc-9:.1f}" width="18" height="18" fill="{colr}" transform="rotate(45 {X(r):.1f} {yc:.1f})"/>')
    else:
        b.append(f'<circle cx="{X(r):.1f}" cy="{yc:.1f}" r="9" fill="{colr}"/>')
    lab = f"{(0.0 if abs(r) < 0.005 else r):+.2f}".replace(".", ",").replace("-", "−").replace("+", "")
    b.append(t(ax1 + 30, yc + 7, lab, 22, INK, weight=600))
    b.append(t(FW, yc + 7, "—" if q is None else f"{q:.2f}".replace(".", ","), 22, BODY, "end"))
Hf = int(yb + 70)
(OUT / "forest.svg").write_text(svg(FW, Hf, b, "Forest plot com o rho de Spearman e intervalo de confiança de 95% de cada teste de H2 e H3, mais a análise de sensibilidade"))

print("Gráficos gerados em", OUT)
