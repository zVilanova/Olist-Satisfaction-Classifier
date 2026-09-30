"""
📦 Simulador de Satisfação do Cliente — Olist
Playground interativo de um modelo XGBoost treinado com o dataset público
"Brazilian E-Commerce" da Olist (Kaggle).

Estrutura do arquivo:
    1. Configuração        4. Gráficos
    2. Definição das features
    3. Modelo e predição   5. Interface (painel, resultado, abas)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH = Path(__file__).parent / "modelo_olist_xgb.json"

# Preencha com seus links: aparecem no rodapé da barra lateral.
GITHUB_URL = "https://github.com/zVilanova"
LINKEDIN_URL = "https://www.linkedin.com/in/leonardovilanova/"

# Faixas de probabilidade de satisfação (classe 1) usadas para classificar o risco.
HIGH_RISK_BELOW = 0.50  # abaixo disso o modelo prevê INSATISFAÇÃO
MODERATE_BELOW = 0.60  # entre 0.50 e 0.60: satisfeito, mas com margem estreita

GREEN, AMBER, RED = "#2ECC71", "#F39C12", "#E74C3C"
PURPLE, ORANGE = "#6C5CE7", "#E17055"

CSS = """
<style>
.stApp {background-color: #070B14;}
.block-container {padding-top: 2rem; max-width: 1250px;}

/* Cabeçalho */
.hero h1 {
    font-size: 2.6rem; font-weight: 800; margin-bottom: .2rem;
    background: linear-gradient(90deg, #6C5CE7 0%, #00B894 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p {font-size: 1.05rem; opacity: .75; margin-bottom: 1.2rem;}

/* Barra lateral com os parâmetros */
section[data-testid="stSidebar"] {background: #0A0F1C; border-right: 1px solid #1E2A44;}
section[data-testid="stSidebar"][aria-expanded="true"] {min-width: 400px;}
section[data-testid="stSidebar"] h2 {font-size: 1.75rem; line-height: 1.25; padding: 0 0 .3rem 0;}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {font-size: 1.1rem; line-height: 1.45;}
[class*="st-key-ctrl_"] {
    border: 1px solid rgba(128,128,128,.25); border-radius: 14px;
    padding: 1rem 1.3rem .6rem 1.3rem; background: rgba(128,128,128,.06);
}
[class*="st-key-ctrl_"] h4 {margin: 0 0 .4rem 0; padding: 0; font-size: 1.3rem;}
[class*="st-key-ctrl_"] label p {font-weight: 500; font-size: 1.05rem;}
[data-testid="stSliderThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {
    font-size: .95rem;
}

/* Resultado */
.card {
    border: 1px solid rgba(128,128,128,.25); border-radius: 14px;
    padding: 1.1rem 1.3rem; background: rgba(128,128,128,.06); margin-bottom: .8rem;
}
.card .verdict {font-size: 1.05rem; margin: .6rem 0 0 0;}
.badge {
    display: inline-block; padding: .25rem .8rem; border-radius: 999px;
    color: white; font-weight: 700; font-size: .85rem; letter-spacing: .3px;
}
.driver {font-size: 1.05rem; padding: .55rem .9rem; border-radius: 10px;
         background: rgba(128,128,128,.08); margin-bottom: .5rem;}

/* Cards de estatística com variação explícita (melhor / pior) */
.stat {border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
       padding: .8rem 1rem; background: rgba(128,128,128,.05); margin-bottom: .8rem;}
.stat .t {font-size: .85rem; opacity: .7;}
.stat .v {font-size: 2.1rem; font-weight: 700; line-height: 1.25;}
.stat .d {display: inline-block; font-size: .78rem; font-weight: 600;
          padding: .15rem .6rem; border-radius: 999px; margin-top: .2rem;}
.stat .d.good {color: #2ecc71; background: rgba(46,204,113,.16);}
.stat .d.bad {color: #ff6b5b; background: rgba(231,76,60,.18);}
.stat .d.neutral {color: #9aa3b5; background: rgba(128,128,128,.16);}

/* Botões de cenário maiores */
[class*="st-key-preset_"] button {
    min-height: 3.6rem; border-radius: 12px;
    border: 1px solid rgba(108,92,231,.55); background: rgba(108,92,231,.14);
    transition: background .15s, border-color .15s;
}
[class*="st-key-preset_"] button p {font-size: 1.1rem; font-weight: 600;}
[class*="st-key-preset_"] button:hover {border-color: #6C5CE7; background: rgba(108,92,231,.32);}

/* Abas: textos maiores */
button[data-baseweb="tab"] p {font-size: 1.15rem; font-weight: 600;}
[data-testid="stTabs"] [data-testid="stMarkdownContainer"] p,
[data-testid="stTabs"] [data-testid="stMarkdownContainer"] li {font-size: 1.1rem; line-height: 1.65;}
[data-testid="stTabs"] [data-testid="stMarkdownContainer"] h3 {font-size: 1.6rem;}
[data-testid="stTabs"] [data-testid="stCaptionContainer"] p {font-size: 1rem;}
[data-testid="stTabs"] label p {font-size: 1.05rem;}
[data-testid="stTabs"] [data-baseweb="select"] {font-size: 1.05rem;}
[data-testid="stTabs"] table td, [data-testid="stTabs"] table th {font-size: 1.05rem; padding: .6rem .8rem;}
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
# 2. DEFINIÇÃO DAS FEATURES (fonte única da verdade: widgets, gráficos e docs)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Feature:
    key: str  # nome da coluna no modelo
    label: str
    group: str
    help: str
    default: int | bool
    lo: int = 0
    hi: int = 1
    step: int = 1
    prefix: str = ""
    suffix: str = ""
    scale: float = 1.0  # valor_no_modelo = valor_no_widget * scale
    kind: str = "slider"  # "slider" | "toggle"

    @property
    def slider_format(self) -> str:
        return f"{self.prefix}%d{self.suffix.replace('%', '%%')}"

    def show(self, value: float) -> str:
        """Formata um valor (em unidades do widget) para exibição."""
        if self.kind == "toggle":
            return "Sim" if value else "Não"
        return f"{self.prefix}{value:,.0f}{self.suffix}".replace(",", ".")


G_LOG, G_PED, G_ANU = "📍 Logística", "🛒 Pedido", "🖼️ Anúncio e produto"

FEATURES: list[Feature] = [
    Feature(
        "tempo_entrega_dias",
        "Tempo total de entrega",
        G_LOG,
        "Dias entre a compra e a chegada do pedido ao cliente.",
        10,
        1,
        60,
        suffix=" dias",
    ),
    Feature(
        "dias_atraso",
        "Dias de atraso",
        G_LOG,
        "Entrega real menos a data prometida. Valores negativos = chegou adiantado.",
        0,
        -20,
        30,
        suffix=" dias",
    ),
    Feature(
        "tempo_postagem_vendedor",
        "Dias para o vendedor postar",
        G_LOG,
        "Tempo que o vendedor levou para despachar o pedido.",
        2,
        0,
        15,
        suffix=" dias",
    ),
    Feature(
        "proporcao_frete",
        "Peso do frete no valor total",
        G_LOG,
        "Frete ÷ valor total do pedido.",
        15,
        0,
        100,
        suffix="%",
        scale=0.01,
    ),
    Feature(
        "mesma_uf",
        "Cliente e vendedor na mesma UF",
        G_LOG,
        "Se comprador e vendedor estão no mesmo estado.",
        True,
        kind="toggle",
    ),
    Feature(
        "valor_total_pedido",
        "Valor total do pedido",
        G_PED,
        "Soma de produtos + frete.",
        150,
        10,
        5000,
        step=10,
        prefix="R$ ",
    ),
    Feature(
        "qtd_itens_pedido",
        "Quantidade de itens",
        G_PED,
        "Número de itens no pedido.",
        1,
        1,
        10,
    ),
    Feature(
        "qtd_parcelas",
        "Quantidade de parcelas",
        G_PED,
        "Parcelas do pagamento.",
        1,
        1,
        24,
        suffix="x",
    ),
    Feature(
        "qtd_fotos",
        "Fotos no anúncio",
        G_ANU,
        "Quantidade de fotos do produto.",
        3,
        1,
        15,
    ),
    Feature(
        "tamanho_descricao",
        "Tamanho da descrição",
        G_ANU,
        "Caracteres na descrição do anúncio.",
        500,
        0,
        4000,
        step=50,
        suffix=" car.",
    ),
    Feature(
        "volume_produto_cm3",
        "Volume do produto",
        G_ANU,
        "Volume da embalagem (comprimento × altura × largura).",
        2000,
        100,
        60000,
        step=100,
        suffix=" cm³",
    ),
]
FEATURE_BY_KEY = {f.key: f for f in FEATURES}
DEFAULTS = {f.key: f.default for f in FEATURES}

# Cenários prontos (sobrescrevem os padrões; valores nas unidades dos widgets).
PRESETS: dict[str, dict] = {
    "↺ Padrão": {},
    "🎯 Pedido ideal": dict(
        tempo_entrega_dias=5,
        dias_atraso=-8,
        tempo_postagem_vendedor=1,
        proporcao_frete=8,
        qtd_fotos=6,
        tamanho_descricao=900,
    ),
    "⏰ Entrega atrasada": dict(
        tempo_entrega_dias=35, dias_atraso=12, tempo_postagem_vendedor=6, mesma_uf=False
    ),
    "💳 Compra parcelada": dict(
        valor_total_pedido=2500, qtd_parcelas=10, tempo_entrega_dias=20, dias_atraso=0
    ),
    "🖼️ Anúncio fraco": dict(
        qtd_fotos=1, tamanho_descricao=50, tempo_postagem_vendedor=5, proporcao_frete=45
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODELO E PREDIÇÃO
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Carregando modelo…")
def load_model() -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def to_model_units(inputs: dict) -> dict[str, float]:
    return {k: float(v) * FEATURE_BY_KEY[k].scale for k, v in inputs.items()}


def build_frame(rows: list[dict], model: xgb.XGBClassifier) -> pd.DataFrame:
    """DataFrame com colunas na ordem e com os tipos exatos que o modelo aprendeu."""
    booster = model.get_booster()
    cols = booster.feature_names
    types = booster.feature_types or ["float"] * len(cols)
    dtypes = {c: "int64" if t == "int" else "float64" for c, t in zip(cols, types)}
    return pd.DataFrame(rows)[cols].astype(dtypes)


def predict_satisfaction(model: xgb.XGBClassifier, rows: list[dict]) -> np.ndarray:
    """Probabilidade (0-1) de satisfação (classe 1) para cada linha."""
    return model.predict_proba(build_frame(rows, model))[:, 1]


def shap_contributions(model: xgb.XGBClassifier, row: dict) -> tuple[pd.Series, float]:
    """Valores SHAP nativos do XGBoost (log-odds) + valor base do modelo."""
    df = build_frame([row], model)
    contribs = model.get_booster().predict(xgb.DMatrix(df), pred_contribs=True)[0]
    return pd.Series(contribs[:-1], index=df.columns), float(contribs[-1])


def risk_level(p_sat: float) -> tuple[str, str, str, str]:
    """(nível, cor, ícone, veredito)."""
    if p_sat < HIGH_RISK_BELOW:
        return (
            "ALTO",
            RED,
            "⚠️",
            "O modelo prevê que o cliente ficará <b>insatisfeito</b> com esta compra.",
        )
    if p_sat < MODERATE_BELOW:
        return (
            "MODERADO",
            AMBER,
            "🟡",
            "O modelo prevê <b>satisfação</b>, mas com margem estreita: pequenos problemas podem virar a previsão.",
        )
    return (
        "BAIXO",
        GREEN,
        "✅",
        "O modelo prevê que o cliente ficará <b>satisfeito</b> com esta compra.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════
def style_fig(fig: go.Figure) -> go.Figure:
    """Aplica o visual escuro/tecnológico a qualquer gráfico."""
    fig.update_layout(
        font=dict(size=15, color="#E6EDF7"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(
        gridcolor="rgba(148,163,184,.12)",
        zerolinecolor="rgba(148,163,184,.30)",
        title_font=dict(size=16),
        tickfont=dict(size=14),
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,.12)",
        zerolinecolor="rgba(148,163,184,.30)",
        title_font=dict(size=16),
        tickfont=dict(size=14),
    )
    return fig


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(style_fig(fig), width="stretch", config={"displayModeBar": False})


def gauge_chart(p_sat: float, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p_sat * 100,
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 46}},
            title={"text": "Probabilidade de satisfação", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": color, "thickness": 0.28},
                "steps": [
                    {
                        "range": [0, HIGH_RISK_BELOW * 100],
                        "color": "rgba(231,76,60,.25)",
                    },
                    {
                        "range": [HIGH_RISK_BELOW * 100, MODERATE_BELOW * 100],
                        "color": "rgba(243,156,18,.25)",
                    },
                    {
                        "range": [MODERATE_BELOW * 100, 100],
                        "color": "rgba(46,204,113,.25)",
                    },
                ],
                "threshold": {
                    "line": {"color": "gray", "width": 3},
                    "thickness": 0.85,
                    "value": HIGH_RISK_BELOW * 100,
                },
            },
        )
    )
    fig.update_layout(height=290, margin=dict(l=30, r=30, t=60, b=10))
    return fig


def contribution_chart(contribs: pd.Series, inputs: dict) -> go.Figure:
    data = contribs.reindex(contribs.abs().sort_values().index)  # maior impacto no topo
    labels = [
        f"{FEATURE_BY_KEY[k].label}: {FEATURE_BY_KEY[k].show(inputs[k])}"
        for k in data.index
    ]
    fig = go.Figure(
        go.Bar(
            x=data.values,
            y=labels,
            orientation="h",
            marker_color=[GREEN if v >= 0 else RED for v in data.values],
            text=[f"{v:+.2f}" for v in data.values],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=14),
            hovertemplate="%{y}<br>Contribuição: %{x:+.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.update_yaxes(automargin=True)
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=50, t=10, b=50),
        showlegend=False,
        xaxis_title="← empurra para INSATISFAÇÃO  |  empurra para SATISFAÇÃO →",
    )
    return fig


def sweep_values(f: Feature, points: int = 80) -> np.ndarray:
    return np.unique(np.linspace(f.lo, f.hi, points).round().astype(int))


def sensitivity_chart(model, inputs: dict, f: Feature, p_now: float) -> go.Figure:
    xs = sweep_values(f)
    base = to_model_units(inputs)
    ys = predict_satisfaction(model, [{**base, f.key: x * f.scale} for x in xs]) * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(width=3, color=PURPLE),
            fill="tozeroy",
            fillcolor="rgba(108,92,231,.15)",
            hovertemplate=f"{f.label}: %{{x}}<br>P(satisfação): %{{y:.1f}}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[inputs[f.key]],
            y=[p_now * 100],
            mode="markers",
            marker=dict(size=14, color=ORANGE, line=dict(width=2, color="white")),
            hovertemplate="Cenário atual<extra></extra>",
        )
    )
    fig.add_hline(
        y=HIGH_RISK_BELOW * 100,
        line_dash="dash",
        line_color="gray",
        annotation_text="limite de decisão (50%)",
        annotation_position="bottom right",
        annotation_font_size=14,
    )
    fig.update_layout(
        height=440,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=f"{f.label}{f' ({f.suffix.strip()})' if f.suffix else ''}",
        yaxis_title="Probabilidade de satisfação (%)",
        yaxis_range=[0, 100],
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. INTERFACE
# ══════════════════════════════════════════════════════════════════════════════
def init_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def apply_preset(overrides: dict) -> None:
    """Callback dos botões de cenário (roda antes do rerun, então é seguro alterar os widgets)."""
    for key, value in {**DEFAULTS, **overrides}.items():
        st.session_state[key] = value


def current_inputs() -> dict:
    """Valores atuais dos widgets (o session_state já vem atualizado no início de cada rerun)."""
    return {f.key: st.session_state[f.key] for f in FEATURES}


def render_group(group: str, key: str) -> None:
    with st.container(key=key):
        st.markdown(f"#### {group}")
        for f in (f for f in FEATURES if f.group == group):
            if f.kind == "toggle":
                st.toggle(f.label, key=f.key, help=f.help)
            else:
                st.slider(
                    f.label,
                    f.lo,
                    f.hi,
                    step=f.step,
                    key=f.key,
                    help=f.help,
                    format=f.slider_format,
                )


def render_controls() -> None:
    with st.sidebar:
        st.markdown("## 🎛️ Parâmetros do pedido")
        st.caption("Mexa nos controles: a previsão atualiza em tempo real.")
        render_group(G_LOG, "ctrl_log")
        render_group(G_PED, "ctrl_ped")
        render_group(G_ANU, "ctrl_anu")


def render_hero() -> None:
    st.markdown(
        '<div class="hero"><h1>📦 Simulador de Satisfação · Olist</h1>'
        "<p>Um modelo XGBoost treinado com pedidos reais de e-commerce. Ajuste logística, pedido e anúncio "
        "na barra lateral e veja, em tempo real, como cada detalhe muda o risco de o cliente ficar insatisfeito.</p></div>",
        unsafe_allow_html=True,
    )


def render_presets() -> None:
    st.markdown("**Comece por um cenário pronto:**")
    cols = st.columns(len(PRESETS))
    for col, (name, overrides) in zip(cols, PRESETS.items()):
        col.button(
            name,
            on_click=apply_preset,
            args=(overrides,),
            width="stretch",
            key=f"preset_{name}",
        )


def render_consistency_warnings(inputs: dict) -> None:
    if inputs["tempo_postagem_vendedor"] > inputs["tempo_entrega_dias"]:
        st.warning(
            "O vendedor demorou mais para postar do que o tempo total de entrega, o que é uma combinação "
            "incomum nos dados reais. O modelo responde mesmo assim, mas trate o resultado com cautela."
        )


def stat_card(
    title: str, value_pct: float, delta_pp: float, higher_is_better: bool
) -> str:
    """Card com a variação em relação ao pedido padrão, dizendo explicitamente se é melhor ou pior."""
    if abs(delta_pp) < 0.05:
        css, text = "neutral", "= igual ao pedido padrão"
    else:
        improved = (delta_pp > 0) == higher_is_better
        css = "good" if improved else "bad"
        arrow = "▲" if delta_pp > 0 else "▼"
        text = f"{arrow} {abs(delta_pp):.1f} p.p. {'melhor' if improved else 'pior'} que o padrão"
    return (
        f'<div class="stat"><div class="t">{title}</div><div class="v">{value_pct:.1f}%</div>'
        f'<span class="d {css}">{text}</span></div>'
    )


def render_result(inputs: dict, p_sat: float, p_ref: float) -> None:
    level, color, icon, verdict = risk_level(p_sat)
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        show_chart(gauge_chart(p_sat, color))

    with right:
        st.markdown(
            f'<div class="card" style="border-left: 6px solid {color}">'
            f'<span class="badge" style="background:{color}">{icon} RISCO {level}</span>'
            f'<p class="verdict">{verdict}</p></div>',
            unsafe_allow_html=True,
        )
        diff = (p_sat - p_ref) * 100
        c1, c2 = st.columns(2)
        c1.markdown(
            stat_card("Satisfação", p_sat * 100, diff, higher_is_better=True),
            unsafe_allow_html=True,
        )
        c2.markdown(
            stat_card("Insatisfação", (1 - p_sat) * 100, -diff, higher_is_better=False),
            unsafe_allow_html=True,
        )
        if level == "ALTO":
            st.info(
                "💡 **Ação sugerida:** enviar um cupom de desconto ou um e-mail de desculpas preventivo "
                "antes que o cliente reclame."
            )
        render_consistency_warnings(inputs)


def render_explanation_tab(model, inputs: dict) -> None:
    contribs, base_value = shap_contributions(model, to_model_units(inputs))
    st.write(
        "Cada barra mostra quanto uma característica **deste pedido** empurrou a previsão para "
        "**satisfação** (verde) ou **insatisfação** (vermelho), em relação à média do modelo."
    )

    neg, pos = contribs[contribs < 0], contribs[contribs > 0]
    c1, c2 = st.columns(2)
    if not neg.empty:
        k = neg.idxmin()
        c1.markdown(
            f'<div class="driver">🔻 <b>Maior fator de risco:</b> {FEATURE_BY_KEY[k].label} '
            f"({FEATURE_BY_KEY[k].show(inputs[k])})</div>",
            unsafe_allow_html=True,
        )
    if not pos.empty:
        k = pos.idxmax()
        c2.markdown(
            f'<div class="driver">🔺 <b>Maior fator de proteção:</b> {FEATURE_BY_KEY[k].label} '
            f"({FEATURE_BY_KEY[k].show(inputs[k])})</div>",
            unsafe_allow_html=True,
        )

    show_chart(contribution_chart(contribs, inputs))
    st.caption(
        f"Valores em log-odds (escala interna do XGBoost). Valor base do modelo: {base_value:+.2f}. "
        f"Base + soma das contribuições = saída bruta do modelo, que a função sigmoide converte em probabilidade."
    )


def render_sensitivity_tab(model, inputs: dict, p_sat: float) -> None:
    st.write(
        "Varia **um** parâmetro por vez, mantendo todos os outros como estão na barra lateral. "
        "O ponto laranja é o seu cenário atual."
    )
    options = [f for f in FEATURES if f.kind == "slider"]
    f = st.selectbox(
        "Parâmetro a variar",
        options,
        format_func=lambda x: x.label,
        key="sweep_feature",
    )
    show_chart(sensitivity_chart(model, inputs, f, p_sat))


def render_about_tab(model) -> None:
    st.markdown(
        "### Sobre o projeto\n"
        "Este simulador é a vitrine de um modelo de **classificação binária** (cliente satisfeito × insatisfeito) "
        "construído com **XGBoost** a partir do dataset público *Brazilian E-Commerce Public Dataset by Olist* (Kaggle). "
        "Foram criadas **11 features** combinando as tabelas de pedidos, itens, pagamentos, produtos e avaliações.\n\n"
        "### Dicionário de features"
    )
    doc = pd.DataFrame(
        {
            "Feature (modelo)": [f.key for f in FEATURES],
            "Descrição": [f.help for f in FEATURES],
            "Faixa no simulador": [
                "Sim / Não"
                if f.kind == "toggle"
                else f"{f.show(f.lo)} a {f.show(f.hi)}"
                for f in FEATURES
            ],
        }
    )
    st.table(doc.set_index("Feature (modelo)"))
    st.markdown(
        "### Como interpretar\n"
        f"- A previsão é **satisfeito** quando a probabilidade da classe 1 é ≥ {HIGH_RISK_BELOW:.0%}. "
        f"Entre {HIGH_RISK_BELOW:.0%} e {MODERATE_BELOW:.0%} o risco é classificado como *moderado*.\n"
        "- O modelo foi treinado com `scale_pos_weight` (≈ 0,30) para lidar com o desbalanceamento entre as classes. "
        "Por isso as probabilidades funcionam melhor como **índice de risco relativo** do que como frequência real.\n"
        "- A explicação usa valores **SHAP** calculados nativamente pelo XGBoost (`pred_contribs`).\n\n"
        "### Limitações\n"
        "- O modelo captura **correlações** nos dados de 2016–2018 do marketplace, não relações causais.\n"
        "- Combinações fora da distribuição real (ex.: entrega em 1 dia com 30 dias de atraso) são extrapolações.\n"
        "- Satisfação é medida pela nota da avaliação; fatores não observados (qualidade real do produto, expectativa "
        "do cliente) ficam de fora."
    )


def render_footer() -> None:
    links = [
        f"[GitHub]({GITHUB_URL})" if GITHUB_URL else "",
        f"[LinkedIn]({LINKEDIN_URL})" if LINKEDIN_URL else "",
    ]
    links = [l for l in links if l]
    if links:
        st.divider()
        st.caption(" · ".join(links))


def main() -> None:
    st.set_page_config(
        page_title="Simulador de Satisfação · Olist",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    if not MODEL_PATH.exists():
        st.error(
            f"Arquivo do modelo não encontrado: `{MODEL_PATH.name}`. "
            "Coloque-o na mesma pasta do `app.py`."
        )
        st.stop()

    model = load_model()
    mismatch = set(model.get_booster().feature_names) ^ set(FEATURE_BY_KEY)
    if mismatch:
        st.error(f"As features do modelo não batem com as do app: {sorted(mismatch)}")
        st.stop()

    init_state()
    render_hero()
    render_presets()

    inputs = current_inputs()
    p_sat = float(predict_satisfaction(model, [to_model_units(inputs)])[0])
    p_ref = float(predict_satisfaction(model, [to_model_units(DEFAULTS)])[0])

    render_controls()
    st.divider()
    render_result(inputs, p_sat, p_ref)

    st.divider()
    tab_why, tab_curve, tab_about = st.tabs(
        ["🧠 Por que essa previsão?", "📈 Curva de sensibilidade", "📚 Sobre o projeto"]
    )
    with tab_why:
        render_explanation_tab(model, inputs)
    with tab_curve:
        render_sensitivity_tab(model, inputs, p_sat)
    with tab_about:
        render_about_tab(model)

    render_footer()


main()
