"""
Análise da survey "Assistentes de IA no desenvolvimento de software"
Segue o protocolo pré-registrado (seções 5.1 a 5.4) e aplica os desvios
declarados no relatório (ver DESVIOS abaixo).

Uso:
    python analise.py "Práticas de desenvolvimento de software com e sem IA.csv"

Saídas (pasta ./saida):
    dados_limpos.csv        dataset válido e anonimizado (sem textos abertos)
    codificacao_A1.csv      respostas abertas válidas para codificação temática
    resultados.txt          relatório completo da análise
    resultados.json         números principais (para o dashboard)

Dependências: pandas, numpy, scipy, pingouin, scikit-posthocs
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
from scipy import stats

ALPHA = 0.05
SAIDA = Path("saida")

# ----------------------------------------------------------------------------
# Mapeamento das colunas do Google Forms para os IDs do protocolo
# (busca por trecho do enunciado, para não depender da ordem das colunas)
# ----------------------------------------------------------------------------
TRECHOS = {
    "TS": "Carimbo de data/hora",
    "T0": "aceita participar",
    "F1": "18 anos ou mais",
    "F2": "matriculado(a) na disciplina",
    "D1": "Há quantos anos",
    "D2": "papel principal",
    "D3": "porte da empresa",
    "D4": "tipo da empresa",
    "D5": "política da sua empresa",
    "U1": "Nos últimos 30 dias",
    "U2": "Com que frequência",
    "U3": "Quais ferramentas",
    "U4": "Para quais tarefas",
    "P1": "concluo minhas tarefas mais rápido",
    "P2": "código repetitivo",
    "AT": "Para mostrar que está lendo",
    "P3": "manter a concentração",
    "P4": "Perco tempo corrigindo",
    "C1": "Confio que o código sugerido",
    "C2": "aceitar as sugestões sem modificá-las",
    "C3": "Desconfio das sugestões",
    "V1": "Leio linha a linha",
    "V2": "Escrevo ou executo testes",
    "V3": "Submeto código gerado por IA",
    "A1": "atrapalhou mais do que ajudou",
    "N1": "Por que você não usa",
}

ESCALA = {
    "Discordo totalmente": 1, "Discordo": 2, "Neutro": 3, "Concordo": 4, "Concordo totalmente": 5,
    "Nunca": 1, "Raramente": 2, "Às vezes": 3, "Frequentemente": 4, "Sempre": 5,
}
INVERTIDOS = ["P4", "C3"]
CONSTRUTOS = {"PROD": ["P1", "P2", "P3", "P4"], "CONF": ["C1", "C2", "C3"], "VERIF": ["V1", "V2", "V3"]}
LIKERT = [i for itens in CONSTRUTOS.values() for i in itens]
FAIXAS = {"bins": [-1, 2, 5, 10, np.inf], "labels": ["≤2", "3–5", "6–10", ">10"]}

out_linhas = []


def log(texto=""):
    print(texto)
    out_linhas.append(str(texto))


def renomear(df):
    mapa = {}
    for cod, trecho in TRECHOS.items():
        achadas = [c for c in df.columns if trecho.lower() in c.lower()]
        if len(achadas) != 1:
            raise ValueError(f"Coluna de {cod} não encontrada sem ambiguidade: {achadas}")
        mapa[achadas[0]] = cod
    return df.rename(columns=mapa)[list(TRECHOS)]


# ----------------------------------------------------------------------------
# 5.1 Limpeza (na ordem do protocolo, com contagem por motivo)
# ----------------------------------------------------------------------------
def limpar(df):
    log("=" * 70)
    log("5.1 LIMPEZA E DESCARTES")
    log("=" * 70)
    contagem = {"respostas_brutas": len(df)}

    ok = df["T0"].fillna("").str.startswith("Li e aceito")
    contagem["sem_consentimento"] = int((~ok).sum())
    df = df[ok]

    fora = (df["F1"] != "Sim") | (df["F2"] != "Não")
    contagem["fora_da_populacao"] = int(fora.sum())
    contagem["  F1_nao"] = int((df["F1"] != "Sim").sum())
    contagem["  F2_sim (da turma)"] = int(((df["F1"] == "Sim") & (df["F2"] != "Não")).sum())
    df = df[~fora]

    # a pergunta de atenção só aparece para usuários (U1 = Sim)
    usuario = df["U1"] == "Sim"
    falhou = usuario & (df["AT"] != "Discordo totalmente")
    contagem["falha_atencao"] = int(falhou.sum())
    df = df[~falhou]

    num = df[LIKERT].replace(ESCALA).astype(float)
    reto = (df["U1"] == "Sim") & (num.nunique(axis=1) == 1) & num.notna().all(axis=1)
    contagem["straightlining"] = int(reto.sum())
    df = df[~reto]

    ts = pd.to_datetime(df["TS"].str.replace(" GMT-3", "", regex=False), format="%Y/%m/%d %I:%M:%S %p")
    df = df.assign(_ts=ts).sort_values("_ts")
    campos = [c for c in df.columns if c not in ("TS", "_ts")]
    dup = df.duplicated(subset=campos, keep="first")
    # só conta como duplicata se enviada < 5 min após a resposta idêntica anterior
    if dup.any():
        primeira = df.groupby(campos, dropna=False)["_ts"].transform("min")
        dup = dup & ((df["_ts"] - primeira) < pd.Timedelta(minutes=5))
    contagem["duplicatas"] = int(dup.sum())
    df = df[~dup]

    contagem["respostas_validas"] = len(df)
    for k, v in contagem.items():
        log(f"  {k:<28}{v:>5}")
    log(f"  Período das respostas: {ts.min():%d/%m/%Y %H:%M} a {ts.max():%d/%m/%Y %H:%M}")
    return df.drop(columns="_ts"), contagem


def margem(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n)


def wilson(k, n):
    z = 1.96
    centro = (k + z**2 / 2) / (n + z**2)
    meia = z * np.sqrt(k * (n - k) / n + z**2 / 4) / (n + z**2)
    return centro - meia, centro + meia


def multipla(serie):
    itens = serie.dropna().str.split(";").explode().str.strip()
    return itens.value_counts()


# ----------------------------------------------------------------------------
# 5.2 Descritiva e confiabilidade
# ----------------------------------------------------------------------------
def descritiva(df, num):
    log()
    log("=" * 70)
    log("5.2 DESCRITIVA")
    log("=" * 70)
    n = len(df)
    res = {}

    for cod, nome in [("D2", "Papel"), ("D3", "Porte"), ("D4", "Tipo de empresa"), ("D5", "Política de IA")]:
        log(f"\n{nome} ({cod}), n = {n}")
        for cat, k in df[cod].value_counts(dropna=False).items():
            log(f"  {str(cat):<40}{k:>4}  {k/n:6.1%}")

    anos = df["D1"]
    log(f"\nExperiência (D1): mediana {anos.median():.0f} anos, IIQ {anos.quantile(.25):.0f}–{anos.quantile(.75):.0f}, "
        f"mín {anos.min():.0f}, máx {anos.max():.0f}")
    faixa = pd.cut(anos, **FAIXAS)
    log("  Faixas: " + ", ".join(f"{f}: {k}" for f, k in faixa.value_counts(sort=False).items()))

    # QP1 — M1 sobre todos os válidos
    k = int((df["U1"] == "Sim").sum())
    p = k / n
    lo, hi = wilson(k, n)
    log(f"\nQP1 — M1 uso de IA nos últimos 30 dias: {k}/{n} = {p:.1%} "
        f"(margem do protocolo ±{margem(p, n)*100:.1f} p.p.; IC 95% de Wilson {lo:.1%}–{hi:.1%})")
    log(f"  Margem de erro no pior caso (p = 0,5) com n = {n}: ±{margem(0.5, n)*100:.1f} p.p.")
    res["M1"] = {"k": k, "n": n, "p": p, "wilson": [lo, hi], "margem_pior_caso_pp": margem(0.5, n) * 100}

    usu = df[df["U1"] == "Sim"]
    nu = len(usu)
    log(f"\nM2 frequência (usuários, n = {nu})")
    ordem = ["Menos de 1 vez por semana", "1 a 2 vezes por semana", "Quase todos os dias", "Várias vezes ao dia"]
    for cat in ordem:
        kk = int((usu["U2"] == cat).sum())
        log(f"  {cat:<30}{kk:>4}  {kk/nu:6.1%} ±{margem(kk/nu, nu)*100:4.1f} p.p.")
    for cod, nome in [("U3", "Ferramentas"), ("U4", "M3 tarefas")]:
        log(f"\n{nome} ({cod}, múltipla escolha, % sobre {nu} usuários)")
        for cat, kk in multipla(usu[cod]).items():
            log(f"  {cat:<40}{kk:>4}  {kk/nu:6.1%}")

    log(f"\nItens Likert (usuários; invertidos já recodificados como 6 − x)")
    log(f"  {'item':<5}{'n':>4}{'mediana':>9}{'IIQ':>10}   1    2    3    4    5")
    for it in LIKERT:
        s = num[it].dropna()
        dist = " ".join(f"{int((s == v).sum()):>4}" for v in range(1, 6))
        log(f"  {it:<5}{len(s):>4}{s.median():>9.1f}{s.quantile(.25):>6.1f}–{s.quantile(.75):<4.1f}{dist}")
    return res


def confiabilidade(num):
    log()
    log("Confiabilidade (alfa de Cronbach, aceitável ≥ 0,70)")
    decisao = {}
    for nome, itens in CONSTRUTOS.items():
        X = num[itens].dropna()
        a = pg.cronbach_alpha(X)[0]
        sem = {i: pg.cronbach_alpha(X.drop(columns=i))[0] for i in itens}
        log(f"  {nome}: α = {a:.2f} (n = {len(X)}) | α sem cada item: "
            + ", ".join(f"{i} {v:.2f}" for i, v in sem.items()))
        if a >= 0.70:
            decisao[nome] = {"alfa": a, "regra": "escore", "itens": itens}
        else:
            melhor = max(sem, key=sem.get)
            if sem[melhor] >= 0.70:
                fica = [i for i in itens if i != melhor]
                decisao[nome] = {"alfa": a, "regra": "escore sem item", "removido": melhor,
                                 "alfa_novo": sem[melhor], "itens": fica}
                log(f"    → DESVIO: {melhor} removido (α sobe para {sem[melhor]:.2f}); escore = média de {fica}")
            else:
                decisao[nome] = {"alfa": a, "regra": "itens isolados", "itens": itens}
                log(f"    → nenhum item eleva α a 0,70: itens analisados isoladamente")
    return decisao


def variaveis(num, decisao, nome):
    """Retorna as variáveis de análise de um construto conforme a decisão de confiabilidade."""
    d = decisao[nome]
    if d["regra"] == "itens isolados":
        return {it: num[it] for it in d["itens"]}
    rotulo = nome if d["regra"] == "escore" else f"{nome} (sem {d['removido']})"
    return {rotulo: num[d["itens"]].mean(axis=1, skipna=False)}


# ----------------------------------------------------------------------------
# 5.3 Testes por questão de pesquisa
# ----------------------------------------------------------------------------
def testes(df, num, decisao):
    log()
    log("=" * 70)
    log("5.3 TESTES CONFIRMATÓRIOS (bilaterais, α = 0,05, Holm-Bonferroni na família)")
    log("=" * 70)
    familia = []
    anos = df["D1"]
    faixa = pd.cut(anos, **FAIXAS)

    # H2 — PROD × experiência
    log("\nQP2 / H2 — produtividade percebida × experiência")
    for rot, y in variaveis(num, decisao, "PROD").items():
        dd = pd.DataFrame({"y": y, "faixa": faixa, "anos": anos}).dropna()
        grupos = [g["y"].values for _, g in dd.groupby("faixa", observed=True)]
        H, p = stats.kruskal(*grupos)
        eps2 = H / (len(dd) - 1)
        med = dd.groupby("faixa", observed=False)["y"].agg(["count", "median"])
        log(f"  {rot}: Kruskal-Wallis H = {H:.2f}, p = {p:.3f}, ε² = {eps2:.3f} | "
            + "; ".join(f"{f}: n={int(r['count'])}, Md={r['median']:.2f}" for f, r in med.iterrows()))
        familia.append({"hip": "H2", "teste": f"KW {rot} × faixa", "est": H, "p": p, "efeito": f"ε²={eps2:.3f}",
                        "dd": dd})
        rho, p2 = stats.spearmanr(dd["anos"], dd["y"])
        log(f"  {rot}: Spearman anos × {rot}: ρ = {rho:.2f}, p = {p2:.3f}")
        familia.append({"hip": "H2", "teste": f"Spearman anos × {rot}", "est": rho, "p": p2, "efeito": f"ρ={rho:.2f}"})

    # H3 — CONF × VERIF (somente usuários)
    log("\nQP3 / H3 — confiança × verificação (usuários)")
    for rc, xc in variaveis(num, decisao, "CONF").items():
        for rv, xv in variaveis(num, decisao, "VERIF").items():
            dd = pd.DataFrame({"c": xc, "v": xv}).dropna()
            rho, p = stats.spearmanr(dd["c"], dd["v"])
            log(f"  {rc} × {rv}: ρ = {rho:.2f}, p = {p:.3f}, n = {len(dd)}")
            familia.append({"hip": "H3", "teste": f"Spearman {rc} × {rv}", "est": rho, "p": p,
                            "efeito": f"ρ={rho:.2f}"})

    # H4 — uso × política / porte
    log("\nQP4 / H4a, H4b — uso × política da empresa / porte")
    if df["U1"].nunique() < 2:
        log(f"  INVIÁVEL: U1 é constante entre os válidos ({df['U1'].iloc[0]} = {len(df)}). "
            "χ² e Fisher exigem variação em ambas as variáveis. Tabelas apenas descritivas:")
        for cod in ["D5", "D3"]:
            log(f"  {cod}: " + ", ".join(f"{k}: {v}" for k, v in df[cod].value_counts().items()))
        h4 = "inviável"
    else:
        h4 = "testável"
        for cod, nome in [("D5", "H4a"), ("D3", "H4b")]:
            sub = df[~df[cod].isin(["Não sei", "Sou autônomo(a) / freelancer"])]
            tab = pd.crosstab(sub["U1"], sub[cod])
            chi2, p, dof, esp = stats.chi2_contingency(tab)
            v = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
            nota = " (frequência esperada < 5: interpretar com cautela / usar Fisher)" if (esp < 5).any() else ""
            log(f"  {nome}: χ² = {chi2:.2f}, gl = {dof}, p = {p:.3f}, V = {v:.2f}{nota}")
            familia.append({"hip": nome, "teste": f"χ² U1 × {cod}", "est": chi2, "p": p, "efeito": f"V={v:.2f}"})

    # Holm-Bonferroni sobre a família efetivamente testada
    ps = [f["p"] for f in familia]
    rej, p_holm = pg.multicomp(ps, alpha=ALPHA, method="holm")
    log(f"\nCorreção de Holm-Bonferroni (família com {len(familia)} testes)")
    log(f"  {'hip':<4}{'teste':<46}{'efeito':>10}{'p':>8}{'p Holm':>9}  decisão")
    for f, ph, r in zip(familia, p_holm, rej):
        f["p_holm"], f["rejeita"] = float(ph), bool(r)
        log(f"  {f['hip']:<4}{f['teste']:<46}{f['efeito']:>10}{f['p']:>8.3f}{ph:>9.3f}  "
            f"{'rejeita H0' if r else 'não rejeita H0'}")

    # Dunn com Holm só para Kruskal-Wallis significativo após correção
    for f in familia:
        if f["teste"].startswith("KW") and f["rejeita"]:
            log(f"\n  Pós-teste de Dunn (Holm) para {f['teste']}:")
            log(sp.posthoc_dunn(f["dd"], val_col="y", group_col="faixa", p_adjust="holm").round(3).to_string())
    return familia, h4


def sensibilidade(df, num):
    log()
    log("=" * 70)
    log("ANÁLISE DE SENSIBILIDADE (não confirmatória): construtos como pré-registrados")
    log("=" * 70)
    esc = {n: num[i].mean(axis=1, skipna=False) for n, i in CONSTRUTOS.items()}
    dd = pd.DataFrame({**esc, "anos": df["D1"], "faixa": pd.cut(df["D1"], **FAIXAS)}).dropna()
    H, p = stats.kruskal(*[g["PROD"] for _, g in dd.groupby("faixa", observed=True)])
    log(f"  PROD (4 itens) × faixa: H = {H:.2f}, p = {p:.3f}")
    r, p = stats.spearmanr(dd["anos"], dd["PROD"])
    log(f"  Spearman anos × PROD: ρ = {r:.2f}, p = {p:.3f}")
    r, p = stats.spearmanr(dd["CONF"], dd["VERIF"])
    log(f"  Spearman CONF (3 itens, com C3) × VERIF (3 itens): ρ = {r:.2f}, p = {p:.3f}, n = {len(dd)}")
    log("  ATENÇÃO: estes escores violam a regra de confiabilidade (5.2) e não sustentam conclusões.")


def proxy_nao_resposta(df, num):
    log()
    log("Proxy de não resposta (exploratório): primeiros 25% × últimos 25% dos respondentes")
    ordem = df.reset_index(drop=True)
    q = max(1, len(ordem) // 4)
    ini, fim = ordem.index[:q], ordem.index[-q:]
    n2 = num.reset_index(drop=True)
    for it in LIKERT:
        a, b = n2.loc[ini, it].dropna(), n2.loc[fim, it].dropna()
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        log(f"  {it}: Md primeiros {a.median():.1f} × últimos {b.median():.1f}, Mann-Whitney p = {p:.3f}")


def poder(n):
    """Menor |ρ| detectável com 80% de poder, α = 0,05 bilateral (aproximação de Fisher)."""
    z = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) / np.sqrt(n - 3)
    return np.tanh(z)


def main(caminho):
    SAIDA.mkdir(exist_ok=True)
    bruto = renomear(pd.read_csv(caminho))
    df, contagem = limpar(bruto)

    num = df[LIKERT].replace(ESCALA).astype(float)
    for it in INVERTIDOS:
        num[it] = 6 - num[it]
    num = num[df["U1"] == "Sim"]

    desc = descritiva(df, num)
    decisao = confiabilidade(num)
    familia, h4 = testes(df, num, decisao)
    sensibilidade(df, num)
    proxy_nao_resposta(df[df["U1"] == "Sim"], num)

    n_usu = len(num)
    log(f"\nPoder: com n = {n_usu}, o menor |ρ| detectável com 80% de poder é {poder(n_usu):.2f} "
        f"(o protocolo planejava detectar ρ = 0,30 com n ≈ 85).")

    # Saídas
    limpo = df.drop(columns=["T0", "F1", "F2", "AT", "A1", "N1"]).copy()
    limpo.to_csv(SAIDA / "dados_limpos.csv", index=False)
    a1 = df.loc[df["A1"].fillna("").str.strip().str.len() > 2, ["A1"]].reset_index(drop=True)
    a1.index.name = "id"
    a1.assign(codigo_avaliador1="", codigo_avaliador2="").to_csv(SAIDA / "codificacao_A1.csv")
    log(f"\nRespostas abertas A1 válidas para codificação: {len(a1)} (saida/codificacao_A1.csv)")

    resumo = {
        "limpeza": contagem, "descritiva": desc,
        "confiabilidade": decisao, "H4": h4,
        "familia": [{k: v for k, v in f.items() if k != "dd"} for f in familia],
        "poder_rho_minimo": poder(n_usu),
    }
    (SAIDA / "resultados.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=float))
    (SAIDA / "resultados.txt").write_text("\n".join(out_linhas), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Práticas de desenvolvimento de software com e sem IA.csv")
