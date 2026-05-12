"""
markov_animations.py
─────────────────────
Constrói figuras Plotly animadas para a Cadeia de Markov.
Sem código de interface (Streamlit).

Exporta:
  create_distribution_animation(P, state_names, dist_results, frame_duration)
  create_trajectory_animation(P, state_names, trajectory, max_steps, frame_duration)
  create_step_animation(P, state_names, trajectory, frame_duration)
"""

import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ═══════════════════════════════════════════════════════════════════
# Constantes visuais
# ═══════════════════════════════════════════════════════════════════
_BG        = "#0f0f13"
_CARD      = "#1c1c28"
_ACCENT    = "#4f46e5"
_NODE_DIM  = "#1c1c28"
_NODE_HI   = "#7f7fff"
_FONT_MONO = "Space Mono, monospace"


# ═══════════════════════════════════════════════════════════════════
# Helpers internos
# ═══════════════════════════════════════════════════════════════════

def _layout_positions(P: np.ndarray) -> dict[int, tuple[float, float]]:
    n = P.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(n):
            if i != j and P[i, j] > 1e-9:
                G.add_edge(i, j)
    if n == 2:
        return {0: (-0.6, 0.0), 1: (0.6, 0.0)}
    if n <= 7:
        return nx.circular_layout(G)
    return nx.spring_layout(G, seed=42, k=1.6)


def _arrow_annotations(P: np.ndarray, pos: dict,
                       xref="x", yref="y") -> list[dict]:
    n = P.shape[0]
    anns = []
    for i in range(n):
        for j in range(n):
            if i == j or P[i, j] <= 1e-9:
                continue
            x0, y0 = pos[i]
            x1, y1 = pos[j]
            alpha = 0.30 + 0.60 * P[i, j]
            width = 0.80 + 2.50 * P[i, j]
            anns.append(dict(
                x=x1, y=y1, ax=x0, ay=y0,
                xref=xref, yref=yref, axref=xref, ayref=yref,
                showarrow=True, arrowhead=3, arrowsize=1.0,
                arrowwidth=width,
                arrowcolor=f"rgba(127,127,255,{alpha:.2f})",
                startstandoff=22, standoff=22,
            ))
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            anns.append(dict(
                x=mx, y=my, text=f"{P[i, j]:.2f}",
                xref=xref, yref=yref,
                showarrow=False,
                font=dict(color="rgba(192,192,255,0.85)",
                          size=8, family="monospace"),
                bgcolor="rgba(15,15,19,0.65)", borderpad=2,
            ))
    return anns


def _invisible_edge_trace(P: np.ndarray, pos: dict) -> go.Scatter:
    n = P.shape[0]
    ex, ey = [], []
    for i in range(n):
        for j in range(n):
            if i != j and P[i, j] > 1e-9:
                ex += [pos[i][0], pos[j][0], None]
                ey += [pos[i][1], pos[j][1], None]
    return go.Scatter(x=ex, y=ey, mode="lines",
                      line=dict(color="rgba(0,0,0,0)", width=0),
                      hoverinfo="none", showlegend=False)


def _dist_to_colors(dist: np.ndarray) -> list[str]:
    n = len(dist)
    out = []
    for p in dist:
        t = min(p * n, 1.0)
        r = int(0x1c + (0x7f - 0x1c) * t)
        g = int(0x1c + (0x7f - 0x1c) * t)
        b = int(0x28 + (0xff - 0x28) * t)
        out.append(f"rgb({r},{g},{b})")
    return out


def _dist_to_sizes(dist: np.ndarray, lo=28.0, hi=56.0) -> list[float]:
    n = len(dist)
    return [lo + (hi - lo) * min(p * n, 1.0) for p in dist]


def _bar_colors(n: int, hot: int) -> list[str]:
    return [_ACCENT if i == hot else _NODE_DIM for i in range(n)]


def _base_layout(height=530) -> dict:
    return dict(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        height=height,
        margin=dict(l=10, r=10, t=95, b=75),
        font=dict(color="#e8e8f0"),
        hoverlabel=dict(bgcolor=_CARD, font_color="#e8e8f0",
                        font_family=_FONT_MONO),
    )


def _play_controls(frame_duration=650) -> list[dict]:
    return [dict(
        type="buttons", showactive=False, direction="left",
        bgcolor=_CARD, bordercolor="#3a3a52",
        font=dict(color="#e8e8f0", family=_FONT_MONO, size=12),
        x=0.50, y=1.14, xanchor="center", yanchor="top",
        pad=dict(r=8, t=0),
        buttons=[
            dict(label="▶  Play", method="animate",
                 args=[None, dict(
                     frame=dict(duration=frame_duration, redraw=True),
                     fromcurrent=True, mode="immediate",
                     transition=dict(duration=frame_duration // 3),
                 )]),
            dict(label="⏸  Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode="immediate")]),
            dict(label="⏮  Reset", method="animate",
                 args=[["0"], dict(frame=dict(duration=0, redraw=True),
                                   mode="immediate",
                                   transition=dict(duration=0))]),
        ],
    )]


def _build_slider(frames: list[go.Frame], prefix="t = ") -> list[dict]:
    return [dict(
        active=0, pad=dict(t=12, b=5),
        currentvalue=dict(
            prefix=prefix,
            font=dict(color="#a0a0cc", family=_FONT_MONO, size=14),
            xanchor="center", visible=True,
        ),
        steps=[dict(
            args=[[f.name], dict(frame=dict(duration=0, redraw=True),
                                  mode="immediate",
                                  transition=dict(duration=0))],
            label=str(i), method="animate",
        ) for i, f in enumerate(frames)],
        bgcolor="#16161e", activebgcolor=_ACCENT,
        bordercolor="#3a3a52", tickcolor="#3a3a52",
        font=dict(color="#555", size=7),
        len=0.96, x=0.02,
    )]


def _style_axes(fig: go.Figure):
    fig.update_xaxes(showgrid=False, zeroline=False,
                     showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False,
                     showticklabels=False, row=1, col=1)
    fig.update_xaxes(tickfont=dict(color="#888", family="monospace"),
                     showgrid=False, row=1, col=2)
    fig.update_yaxes(range=[0, 1.22], gridcolor="#2a2a38", zeroline=False,
                     tickfont=dict(color="#888", family="monospace"),
                     row=1, col=2)
    for ann in fig.layout.annotations:
        if hasattr(ann, "font"):
            ann.font = dict(color="#666", size=10, family=_FONT_MONO)


# ═══════════════════════════════════════════════════════════════════
# 1. Animação da evolução de π(t)
# ═══════════════════════════════════════════════════════════════════

def create_distribution_animation(
    P: np.ndarray,
    state_names: list[str],
    dist_results: list[np.ndarray],
    frame_duration: int = 650,
) -> go.Figure:
    """
    Grafo (esq.) + barras de π(t) (dir.) animados passo a passo.
    Nós mudam cor/tamanho proporcionalmente à probabilidade.
    """
    n  = P.shape[0]
    pos = _layout_positions(P)
    nx_arr = [pos[i][0] for i in range(n)]
    ny_arr = [pos[i][1] for i in range(n)]
    d0 = dist_results[0]

    # ── Traces iniciais ───────────────────────────────────────────
    edge_trace = _invisible_edge_trace(P, pos)           # idx 0 estático
    node_trace = go.Scatter(                              # idx 1 animado
        x=nx_arr, y=ny_arr, mode="markers+text",
        marker=dict(size=_dist_to_sizes(d0),
                    color=_dist_to_colors(d0),
                    line=dict(color=_NODE_HI, width=2.2)),
        text=state_names, textposition="middle center",
        textfont=dict(color="white", size=11, family="monospace"),
        hovertext=[f"<b>{state_names[i]}</b><br>π({i}) = {d0[i]:.4f}"
                   for i in range(n)],
        hoverinfo="text", showlegend=False,
    )
    bar_trace = go.Bar(                                   # idx 2 animado
        x=state_names, y=list(d0),
        marker=dict(color=_bar_colors(n, int(np.argmax(d0))),
                    line=dict(color=_NODE_HI, width=1.2)),
        text=[f"{v:.3f}" for v in d0], textposition="outside",
        textfont=dict(color="#c0c0ff", size=9, family="monospace"),
        showlegend=False,
    )

    # ── Frames ────────────────────────────────────────────────────
    frames = [
        go.Frame(
            name=str(step), traces=[1, 2],
            data=[
                go.Scatter(
                    marker=dict(size=_dist_to_sizes(dist),
                                color=_dist_to_colors(dist),
                                line=dict(color=_NODE_HI, width=2.2)),
                    hovertext=[f"<b>{state_names[i]}</b><br>π({i}) = {dist[i]:.4f}"
                               for i in range(n)],
                ),
                go.Bar(
                    y=list(dist),
                    marker=dict(color=_bar_colors(n, int(np.argmax(dist))),
                                line=dict(color=_NODE_HI, width=1.2)),
                    text=[f"{v:.3f}" for v in dist],
                ),
            ],
        )
        for step, dist in enumerate(dist_results)
    ]

    # ── Figura ────────────────────────────────────────────────────
    fig = make_subplots(rows=1, cols=2, column_widths=[0.57, 0.43],
                        subplot_titles=["Grafo de Transição", "Distribuição π(t)"])
    fig.add_trace(edge_trace, row=1, col=1)
    fig.add_trace(node_trace, row=1, col=1)
    fig.add_trace(bar_trace,  row=1, col=2)
    fig.frames = frames
    fig.update_layout(
        **_base_layout(),
        updatemenus=_play_controls(frame_duration),
        sliders=_build_slider(frames, prefix="t = "),
        annotations=_arrow_annotations(P, pos) + list(fig.layout.annotations),
    )
    _style_axes(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════
# 2. Animação de trajetória longa (frequências acumuladas)
# ═══════════════════════════════════════════════════════════════════

def create_trajectory_animation(
    P: np.ndarray,
    state_names: list[str],
    trajectory: list[int],
    max_steps: int = 120,
    frame_duration: int = 400,
) -> go.Figure:
    """
    Caminhada aleatória: nó atual em destaque, trail dos últimos visitados,
    barras de frequência acumulada à direita.
    """
    n     = P.shape[0]
    traj  = trajectory[:max_steps]
    TRAIL = max(3, n)
    pos   = _layout_positions(P)
    nx_arr = [pos[i][0] for i in range(n)]
    ny_arr = [pos[i][1] for i in range(n)]

    def _colors(step):
        cur    = traj[step]
        recent = set(traj[max(0, step - TRAIL):step])
        return ["#9f9fff" if i == cur
                else "#3a3a82" if i in recent
                else _NODE_DIM for i in range(n)]

    def _borders(step):
        return ["#ffffff" if i == traj[step] else _NODE_HI for i in range(n)]

    def _sizes(step):
        return [54.0 if i == traj[step] else 30.0 for i in range(n)]

    def _freq(step):
        c = np.zeros(n)
        for s in traj[: step + 1]:
            c[s] += 1
        return (c / (step + 1)).tolist()

    f0 = _freq(0)
    edge_trace = _invisible_edge_trace(P, pos)
    node_trace = go.Scatter(
        x=nx_arr, y=ny_arr, mode="markers+text",
        marker=dict(size=_sizes(0), color=_colors(0),
                    line=dict(color=_borders(0), width=2.5)),
        text=state_names, textposition="middle center",
        textfont=dict(color="white", size=11, family="monospace"),
        hovertext=[f"<b>{state_names[i]}</b>" for i in range(n)],
        hoverinfo="text", showlegend=False,
    )
    bar_trace = go.Bar(
        x=state_names, y=f0,
        marker=dict(color=_bar_colors(n, traj[0]),
                    line=dict(color=_NODE_HI, width=1.2)),
        text=[f"{v:.3f}" for v in f0], textposition="outside",
        textfont=dict(color="#c0c0ff", size=9, family="monospace"),
        showlegend=False,
    )

    frames = []
    for step in range(len(traj)):
        freq = _freq(step)
        frames.append(go.Frame(
            name=str(step), traces=[1, 2],
            data=[
                go.Scatter(
                    marker=dict(size=_sizes(step), color=_colors(step),
                                line=dict(color=_borders(step), width=2.5)),
                    hovertext=[f"<b>{state_names[i]}</b>" for i in range(n)],
                ),
                go.Bar(
                    y=freq,
                    marker=dict(color=_bar_colors(n, traj[step]),
                                line=dict(color=_NODE_HI, width=1.2)),
                    text=[f"{v:.3f}" for v in freq],
                ),
            ],
        ))

    fig = make_subplots(rows=1, cols=2, column_widths=[0.57, 0.43],
                        subplot_titles=["Trajetória — estado atual",
                                        "Freq. de visitas acumulada"])
    fig.add_trace(edge_trace, row=1, col=1)
    fig.add_trace(node_trace, row=1, col=1)
    fig.add_trace(bar_trace,  row=1, col=2)
    fig.frames = frames
    fig.update_layout(
        **_base_layout(),
        updatemenus=_play_controls(frame_duration),
        sliders=_build_slider(frames, prefix="passo = "),
        annotations=_arrow_annotations(P, pos) + list(fig.layout.annotations),
    )
    _style_axes(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════
# 3. Animação passo-a-passo (foco na transição entre estados)
# ═══════════════════════════════════════════════════════════════════

def create_step_animation(
    P: np.ndarray,
    state_names: list[str],
    trajectory: list[int],
    frame_duration: int = 800,
) -> go.Figure:
    """
    Animação detalhada passo a passo:
      - Esquerda : grafo com estado atual em destaque e aresta da transição piscando
      - Direita  : "fita" scrollável mostrando a sequência de estados visitados
                  (cada chip é colorido por recência, seta entre eles)

    Ideal para trajectórias curtas (10–40 passos) onde se quer ver
    cada transição individual.
    """
    n    = P.shape[0]
    traj = trajectory        # já vem limitado pelo chamador
    T    = len(traj)
    pos  = _layout_positions(P)
    nx_arr = [pos[i][0] for i in range(n)]
    ny_arr = [pos[i][1] for i in range(n)]

    WINDOW = 12   # quantos passos aparecem na fita de cada vez

    # ── helpers por frame ─────────────────────────────────────────
    def _node_colors(step: int) -> list[str]:
        cur  = traj[step]
        prev = traj[step - 1] if step > 0 else -1
        out  = []
        for i in range(n):
            if i == cur:
                out.append("#b0b0ff")   # estado atual — lilás claro
            elif i == prev:
                out.append("#5050aa")   # estado anterior — roxo médio
            else:
                out.append(_NODE_DIM)
        return out

    def _node_sizes(step: int) -> list[float]:
        cur  = traj[step]
        prev = traj[step - 1] if step > 0 else -1
        return [58.0 if i == cur
                else 38.0 if i == prev
                else 30.0 for i in range(n)]

    def _node_borders(step: int) -> list[str]:
        cur = traj[step]
        return ["#ffffff" if i == cur else "#7f7fff" for i in range(n)]

    def _transition_arrow(step: int) -> list[dict]:
        """Annotation extra que pisca na aresta atual (step>0)."""
        if step == 0:
            return []
        src, dst = traj[step - 1], traj[step]
        if src == dst:
            return []
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        return [dict(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.4,
            arrowwidth=3.5, arrowcolor="rgba(255,220,80,0.9)",
            startstandoff=22, standoff=22,
        )]

    # Fita de passos: Scatter horizontal mostrando a janela [step-W, step]
    def _ribbon_traces(step: int):
        """
        Retorna (x_vals, y_vals, texts, colors, sizes) para a fita.
        Cada ponto é um passo; o mais recente fica à direita.
        """
        start = max(0, step - WINDOW + 1)
        window = traj[start: step + 1]
        xs     = list(range(len(window)))
        ys     = [0.5] * len(window)
        texts  = [state_names[s] for s in window]
        # Cor: mais recente = mais brilhante
        L      = len(window)
        colors = []
        for k in range(L):
            t = k / max(L - 1, 1)
            r = int(0x2e + (0x9f - 0x2e) * t)
            g = int(0x2e + (0x9f - 0x2e) * t)
            b = int(0x52 + (0xff - 0x52) * t)
            colors.append(f"rgb({r},{g},{b})")
        sizes = [14 + 22 * (k / max(L - 1, 1)) for k in range(L)]
        # Bordas: último é branco
        borders = ["#ffffff" if k == L - 1 else "#7f7fff" for k in range(L)]
        return xs, ys, texts, colors, sizes, borders

    # ── Traces iniciais ───────────────────────────────────────────
    edge_trace = _invisible_edge_trace(P, pos)

    # Nós do grafo (subplot 1)
    node_trace = go.Scatter(
        x=nx_arr, y=ny_arr, mode="markers+text",
        marker=dict(size=_node_sizes(0), color=_node_colors(0),
                    line=dict(color=_node_borders(0), width=2.5)),
        text=state_names, textposition="middle center",
        textfont=dict(color="white", size=11, family="monospace"),
        hovertext=[f"<b>{state_names[i]}</b>" for i in range(n)],
        hoverinfo="text", showlegend=False,
    )

    # Fita de estados (subplot 2)
    rx0, ry0, rt0, rc0, rs0, rb0 = _ribbon_traces(0)
    ribbon_trace = go.Scatter(
        x=rx0, y=ry0, mode="markers+text",
        marker=dict(size=rs0, color=rc0,
                    line=dict(color=rb0, width=2),
                    symbol="circle"),
        text=rt0, textposition="top center",
        textfont=dict(color="#e8e8f0", size=10, family="monospace"),
        hoverinfo="skip", showlegend=False,
    )

    # ── Frames ────────────────────────────────────────────────────
    frames = []
    base_anns = _arrow_annotations(P, pos)

    for step in range(T):
        rx, ry, rt, rc, rs, rb = _ribbon_traces(step)

        # Conectores entre chips da fita
        conn_x, conn_y = [], []
        for k in range(len(rx) - 1):
            conn_x += [rx[k], rx[k + 1], None]
            conn_y += [ry[k], ry[k + 1], None]

        transition_ann = _transition_arrow(step)

        frames.append(go.Frame(
            name=str(step),
            traces=[1, 2, 3],
            data=[
                # Nós do grafo
                go.Scatter(
                    marker=dict(size=_node_sizes(step),
                                color=_node_colors(step),
                                line=dict(color=_node_borders(step), width=2.5)),
                    hovertext=[f"<b>{state_names[i]}</b>" for i in range(n)],
                ),
                # Fita de chips
                go.Scatter(
                    x=rx, y=ry,
                    marker=dict(size=rs, color=rc,
                                line=dict(color=rb, width=2)),
                    text=rt,
                ),
                # Conectores da fita
                go.Scatter(x=conn_x, y=conn_y),
            ],
            layout=go.Layout(
                annotations=base_anns + transition_ann + list(
                    _subplot_title_anns()
                ),
            ),
        ))

    # Conectores iniciais (passo 0 — apenas 1 chip, sem conectores)
    conn_trace = go.Scatter(
        x=[], y=[], mode="lines",
        line=dict(color="rgba(127,127,255,0.4)", width=1.5, dash="dot"),
        hoverinfo="none", showlegend=False,
    )

    # ── Figura ────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.52, 0.48],
        subplot_titles=["Grafo — transição atual", "Sequência de estados"],
    )
    fig.add_trace(edge_trace,   row=1, col=1)  # 0
    fig.add_trace(node_trace,   row=1, col=1)  # 1
    fig.add_trace(ribbon_trace, row=1, col=2)  # 2
    fig.add_trace(conn_trace,   row=1, col=2)  # 3
    fig.frames = frames

    fig.update_layout(
        **_base_layout(height=500),
        updatemenus=_play_controls(frame_duration),
        sliders=_build_slider(frames, prefix="passo = "),
        annotations=base_anns + list(fig.layout.annotations),
    )

    # Axes do grafo
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)

    # Axes da fita
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False,
                     range=[-0.8, WINDOW - 0.2], row=1, col=2)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,
                     range=[0, 1], row=1, col=2)

    for ann in fig.layout.annotations:
        if hasattr(ann, "font"):
            ann.font = dict(color="#666", size=10, family=_FONT_MONO)

    return fig


def _subplot_title_anns():
    """Devolve as anotações de título dos subplots (para recriar nos frames)."""
    return []   # os títulos ficam estáticos via layout inicial — sem necessidade de recriar