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


def _get_self_loop_path(x, y, dx, dy, radius=0.12):
    cx = x + dx * radius * 1.4
    cy = y + dy * radius * 1.4
    t = np.linspace(0, 2 * np.pi, 40)
    xs = cx + radius * np.cos(t)
    ys = cy + radius * np.sin(t)
    return list(xs) + [None], list(ys) + [None]


def _self_loop_base_trace(P: np.ndarray, pos: dict) -> go.Scatter:
    ex, ey = [], []
    n = P.shape[0]
    for i in range(n):
        if P[i, i] > 1e-9:
            x0, y0 = pos[i]
            norm = np.hypot(x0, y0)
            dx, dy = (0.0, 1.0) if norm < 1e-5 else (x0 / norm, y0 / norm)
            xs, ys = _get_self_loop_path(x0, y0, dx, dy)
            ex.extend(xs)
            ey.extend(ys)
    return go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(color="rgba(127,127,255,0.45)", width=1.5),
        hoverinfo="none", showlegend=False,
    )


def _arrow_annotations(P: np.ndarray, pos: dict, xref="x", yref="y") -> list[dict]:
    n = P.shape[0]
    anns = []
    for i in range(n):
        for j in range(n):
            if P[i, j] <= 1e-9:
                continue
            x0, y0 = pos[i]
            x1, y1 = pos[j]
            if i == j:
                norm = np.hypot(x0, y0)
                dx, dy = (0.0, 1.0) if norm < 1e-5 else (x0 / norm, y0 / norm)
                tx = x0 + dx * 0.36
                ty = y0 + dy * 0.36
                anns.append(dict(
                    x=tx, y=ty, text=f"{P[i, i]:.2f}",
                    xref=xref, yref=yref, showarrow=False,
                    font=dict(color="rgba(192,192,255,0.85)", size=9, family="monospace"),
                    bgcolor="rgba(15,15,19,0.65)", borderpad=2,
                ))
            else:
                alpha = 0.30 + 0.60 * P[i, j]
                width = 0.80 + 2.50 * P[i, j]
                anns.append(dict(
                    x=x1, y=y1, ax=x0, ay=y0,
                    xref=xref, yref=yref, axref=xref, ayref=yref,
                    showarrow=True, arrowhead=3, arrowsize=1.0,
                    arrowwidth=width,
                    arrowcolor=f"rgba(127,127,255,{alpha:.2f})",
                    startstandoff=24, standoff=24,
                ))
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                anns.append(dict(
                    x=mx, y=my, text=f"{P[i, j]:.2f}",
                    xref=xref, yref=yref, showarrow=False,
                    font=dict(color="rgba(192,192,255,0.85)", size=8, family="monospace"),
                    bgcolor="rgba(15,15,19,0.65)", borderpad=2,
                ))
    return anns


def _invisible_edge_trace(P: np.ndarray, pos: dict) -> go.Scatter:
    n = P.shape[0]
    ex, ey = [], []
    for i in range(n):
        for j in range(n):
            if P[i, j] > 1e-9:
                x0, y0 = pos[i]
                if i == j:
                    norm = np.hypot(x0, y0)
                    dx, dy = (0.0, 1.0) if norm < 1e-5 else (x0 / norm, y0 / norm)
                    ex += [x0, x0 + dx * 0.4, None]
                    ey += [y0, y0 + dy * 0.4, None]
                else:
                    x1, y1 = pos[j]
                    ex += [x0, x1, None]
                    ey += [y0, y1, None]
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
        hoverlabel=dict(bgcolor=_CARD, font_color="#e8e8f0", font_family=_FONT_MONO),
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
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            dict(label="⏮  Reset", method="animate",
                 args=[["0"], dict(frame=dict(duration=0, redraw=True),
                                   mode="immediate", transition=dict(duration=0))]),
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
                                  mode="immediate", transition=dict(duration=0))],
            label=str(i), method="animate",
        ) for i, f in enumerate(frames)],
        bgcolor="#16161e", activebgcolor=_ACCENT,
        bordercolor="#3a3a52", tickcolor="#3a3a52",
        font=dict(color="#555", size=7),
        len=0.96, x=0.02,
    )]


def _style_axes(fig: go.Figure):
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_xaxes(tickfont=dict(color="#888", family="monospace"),
                     showgrid=False, row=1, col=2)
    fig.update_yaxes(range=[0, 1.22], gridcolor="#2a2a38", zeroline=False,
                     tickfont=dict(color="#888", family="monospace"), row=1, col=2)
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
    n   = P.shape[0]
    pos = _layout_positions(P)
    nx_arr = [pos[i][0] for i in range(n)]
    ny_arr = [pos[i][1] for i in range(n)]
    d0 = dist_results[0]

    edge_trace = _invisible_edge_trace(P, pos)
    loop_trace = _self_loop_base_trace(P, pos)

    node_trace = go.Scatter(
        x=nx_arr, y=ny_arr, mode="markers+text",
        marker=dict(size=_dist_to_sizes(d0), color=_dist_to_colors(d0),
                    line=dict(color=_NODE_HI, width=2.2)),
        text=state_names, textposition="middle center",
        textfont=dict(color="white", size=11, family="monospace"),
        hovertext=[f"<b>{state_names[i]}</b><br>π({i}) = {d0[i]:.4f}" for i in range(n)],
        hoverinfo="text", showlegend=False,
    )
    bar_trace = go.Bar(
        x=state_names, y=list(d0),
        marker=dict(color=_bar_colors(n, int(np.argmax(d0))),
                    line=dict(color=_NODE_HI, width=1.2)),
        text=[f"{v:.3f}" for v in d0], textposition="outside",
        textfont=dict(color="#c0c0ff", size=9, family="monospace"),
        showlegend=False,
    )

    frames = [
        go.Frame(
            name=str(step), traces=[2, 3],
            data=[
                go.Scatter(
                    marker=dict(size=_dist_to_sizes(dist), color=_dist_to_colors(dist),
                                line=dict(color=_NODE_HI, width=2.2)),
                    hovertext=[f"<b>{state_names[i]}</b><br>π({i}) = {dist[i]:.4f}" for i in range(n)],
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

    fig = make_subplots(rows=1, cols=2, column_widths=[0.57, 0.43],
                        subplot_titles=["Grafo associado à cadeia de Markov", "Distribuição π(t)"])
    fig.add_trace(edge_trace, row=1, col=1)  # 0
    fig.add_trace(loop_trace, row=1, col=1)  # 1
    fig.add_trace(node_trace, row=1, col=1)  # 2
    fig.add_trace(bar_trace,  row=1, col=2)  # 3
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
# 2. Animação passo-a-passo com frequência acumulada
#    Layout: grafo (esq, rowspan 2) | fita de estados (dir, topo)
#                                   | freq. acumulada  (dir, base)
# ═══════════════════════════════════════════════════════════════════

def create_step_animation(
    P: np.ndarray,
    state_names: list[str],
    trajectory: list[int],
    frame_duration: int = 800,
) -> go.Figure:

    n    = P.shape[0]
    traj = trajectory
    T    = len(traj)
    pos  = _layout_positions(P)
    nx_arr = [pos[i][0] for i in range(n)]
    ny_arr = [pos[i][1] for i in range(n)]

    WINDOW = 12

    # ── helpers locais ──────────────────────────────────────────────

    def _node_colors(step: int) -> list[str]:
        cur  = traj[step]
        prev = traj[step - 1] if step > 0 else -1
        return ["#b0b0ff" if i == cur else "#5050aa" if i == prev else _NODE_DIM
                for i in range(n)]

    def _node_sizes(step: int) -> list[float]:
        cur  = traj[step]
        prev = traj[step - 1] if step > 0 else -1
        return [58.0 if i == cur else 38.0 if i == prev else 30.0 for i in range(n)]

    def _node_borders(step: int) -> list[str]:
        return ["#ffffff" if i == traj[step] else "#7f7fff" for i in range(n)]

    def _ribbon(step: int):
        start  = max(0, step - WINDOW + 1)
        window = traj[start: step + 1]
        L      = len(window)
        xs     = list(range(L))
        ys     = [0.5] * L
        texts  = [state_names[s] for s in window]
        colors, sizes, borders = [], [], []
        for k in range(L):
            t = k / max(L - 1, 1)
            r = int(0x2e + (0x9f - 0x2e) * t)
            g = int(0x2e + (0x9f - 0x2e) * t)
            b = int(0x52 + (0xff - 0x52) * t)
            colors.append(f"rgb({r},{g},{b})")
            sizes.append(14 + 22 * t)
            borders.append("#ffffff" if k == L - 1 else "#7f7fff")
        return xs, ys, texts, colors, sizes, borders

    def _freq(step: int) -> list[float]:
        c = np.zeros(n)
        for s in traj[: step + 1]:
            c[s] += 1
        return (c / (step + 1)).tolist()

    def _hi_loop(step: int):
        if step == 0:
            return [], []
        src, dst = traj[step - 1], traj[step]
        if src == dst:
            x0, y0 = pos[src]
            norm = np.hypot(x0, y0)
            dx, dy = (0.0, 1.0) if norm < 1e-5 else (x0 / norm, y0 / norm)
            return _get_self_loop_path(x0, y0, dx, dy, radius=0.12)
        return [], []

    def _transition_ann(step: int) -> list[dict]:
        empty = dict(x=0, y=0, text="", showarrow=False, opacity=0, xref="x", yref="y")
        if step == 0:
            return [empty]
        src, dst = traj[step - 1], traj[step]
        if src == dst:
            return [empty]
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        return [dict(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=1.4,
            arrowwidth=3.5, arrowcolor="rgba(255,220,80,0.9)",
            startstandoff=24, standoff=24, opacity=1,
        )]

    # ── traces iniciais ─────────────────────────────────────────────

    f0 = _freq(0)
    rx0, ry0, rt0, rc0, rs0, rb0 = _ribbon(0)

    edge_trace     = _invisible_edge_trace(P, pos)           # 0
    loop_base      = _self_loop_base_trace(P, pos)           # 1
    loop_hi        = go.Scatter(                             # 2
        x=[], y=[], mode="lines",
        line=dict(color="rgba(255,220,80,0.95)", width=3.5),
        hoverinfo="none", showlegend=False,
    )
    node_trace     = go.Scatter(                             # 3
        x=nx_arr, y=ny_arr, mode="markers+text",
        marker=dict(size=_node_sizes(0), color=_node_colors(0),
                    line=dict(color=_node_borders(0), width=2.5)),
        text=state_names, textposition="middle center",
        textfont=dict(color="white", size=11, family="monospace"),
        hovertext=[f"<b>{state_names[i]}</b>" for i in range(n)],
        hoverinfo="text", showlegend=False,
    )
    ribbon_trace   = go.Scatter(                             # 4
        x=rx0, y=ry0, mode="markers+text",
        marker=dict(size=rs0, color=rc0, line=dict(color=rb0, width=2), symbol="circle"),
        text=rt0, textposition="top center",
        textfont=dict(color="#e8e8f0", size=10, family="monospace"),
        hoverinfo="skip", showlegend=False,
    )
    conn_trace     = go.Scatter(                             # 5
        x=[], y=[], mode="lines",
        line=dict(color="rgba(127,127,255,0.4)", width=1.5, dash="dot"),
        hoverinfo="none", showlegend=False,
    )
    freq_bar_trace = go.Bar(                                 # 6
        x=state_names, y=f0,
        marker=dict(color=_bar_colors(n, traj[0]), line=dict(color=_NODE_HI, width=1.2)),
        text=[f"{v:.3f}" for v in f0], textposition="outside",
        textfont=dict(color="#c0c0ff", size=9, family="monospace"),
        showlegend=False,
    )

    # ── figura: grafo ocupa as duas linhas à esquerda ───────────────

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.52, 0.48],
        row_heights=[0.50, 0.50],
        specs=[[{"rowspan": 2}, {}],
               [None,           {}]],
        subplot_titles=[
            "Grafo — transição atual",
            "Sequência de estados",
            "Freq. acumulada de visitas",
        ],
        vertical_spacing=0.14,
    )

    fig.add_trace(edge_trace,     row=1, col=1)
    fig.add_trace(loop_base,      row=1, col=1)
    fig.add_trace(loop_hi,        row=1, col=1)
    fig.add_trace(node_trace,     row=1, col=1)
    fig.add_trace(ribbon_trace,   row=1, col=2)
    fig.add_trace(conn_trace,     row=1, col=2)
    fig.add_trace(freq_bar_trace, row=2, col=2)

    # ── frames ──────────────────────────────────────────────────────

    base_anns = _arrow_annotations(P, pos)
    frames    = []

    for step in range(T):
        rx, ry, rt, rc, rs, rb = _ribbon(step)
        freq = _freq(step)
        hx, hy = _hi_loop(step)

        conn_x, conn_y = [], []
        for k in range(len(rx) - 1):
            conn_x += [rx[k], rx[k + 1], None]
            conn_y += [ry[k], ry[k + 1], None]

        frames.append(go.Frame(
            name=str(step),
            traces=[2, 3, 4, 5, 6],
            data=[
                go.Scatter(x=hx, y=hy),
                go.Scatter(
                    marker=dict(size=_node_sizes(step), color=_node_colors(step),
                                line=dict(color=_node_borders(step), width=2.5)),
                    hovertext=[f"<b>{state_names[i]}</b>" for i in range(n)],
                ),
                go.Scatter(
                    x=rx, y=ry,
                    marker=dict(size=rs, color=rc, line=dict(color=rb, width=2)),
                    text=rt,
                ),
                go.Scatter(x=conn_x, y=conn_y),
                go.Bar(
                    y=freq,
                    marker=dict(color=_bar_colors(n, traj[step]),
                                line=dict(color=_NODE_HI, width=1.2)),
                    text=[f"{v:.3f}" for v in freq],
                ),
            ],
            layout=go.Layout(annotations=base_anns + _transition_ann(step)),
        ))

    fig.frames = frames

    # ── layout geral ────────────────────────────────────────────────

    fig.update_layout(
        **_base_layout(height=600),
        updatemenus=_play_controls(frame_duration),
        sliders=_build_slider(frames, prefix="passo = "),
        annotations=base_anns + list(fig.layout.annotations),
    )

    # grafo
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    # fita
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False,
                     range=[-0.8, WINDOW - 0.2], row=1, col=2)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,
                     range=[0, 1], row=1, col=2)
    # barras de frequência
    fig.update_xaxes(tickfont=dict(color="#888", family="monospace"),
                     showgrid=False, row=2, col=2)
    fig.update_yaxes(range=[0, 1.25], gridcolor="#2a2a38", zeroline=False,
                     tickfont=dict(color="#888", family="monospace"), row=2, col=2)

    for ann in fig.layout.annotations:
        if hasattr(ann, "font"):
            ann.font = dict(color="#666", size=10, family=_FONT_MONO)

    return fig