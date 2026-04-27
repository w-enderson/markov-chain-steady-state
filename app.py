import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from markov_generators import generate_transition_matrix, generate_probability_vector
from others import *

# ═════════════════════════════════════════════════════════════════════════════
# Configuração da página
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Cadeia de Markov",
    page_icon="🔗",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f0f13;
    color: #e8e8f0;
}
h1,h2,h3 { font-family: 'Space Mono', monospace; }

section[data-testid="stSidebar"] {
    background: #16161e;
    border-right: 1px solid #2a2a38;
}

div.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    padding: 0.6rem 1.2rem;
    letter-spacing: 0.04em;
    transition: opacity .2s;
    width: 100%;
}
div.stButton > button:hover { opacity: 0.85; }

div[data-testid="stNumberInput"] input {
    background: #1c1c28;
    border: 1px solid #3a3a52;
    color: #e8e8f0;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #5555aa;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
}

.metric-card {
    background: #1c1c28;
    border: 1px solid #2e2e42;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    height: 90px;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    color: #7f7fff;
    font-weight: 700;
    line-height: 1.4;
}
.metric-label {
    font-size: 0.7rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.scc-chip {
    display: inline-block;
    background: #1c1c28;
    border: 1px solid #3a3a52;
    border-radius: 20px;
    padding: 0.15rem 0.7rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #a0a0cc;
    margin: 0.2rem;
}
.scc-chip.absorb {
    border-color: #7f7fff;
    color: #c0c0ff;
}

.traj-step {
    display: inline-block;
    background: #1c1c28;
    border: 1px solid #3a3a52;
    border-radius: 6px;
    padding: 0.1rem 0.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #a0a0cc;
    margin: 0.15rem;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Funções de visualização (responsabilidade exclusiva do front)
# ═════════════════════════════════════════════════════════════════════════════

def _mpl_style(fig, ax):
    fig.patch.set_facecolor("#0f0f13")
    ax.set_facecolor("#0f0f13")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2e2e42")
    ax.tick_params(colors="#888")

def update_matrix():
    edited_data = st.session_state.mat_editor["edited_rows"]
    
    for row_idx, cols in edited_data.items():
        for col_name, new_val in cols.items():
            col_idx = state_names.index(col_name)
            st.session_state.matrix[int(row_idx), col_idx] = float(new_val)

def draw_markov_graph(
    P: np.ndarray,
    state_names: list[str],
    highlight_node: int | None = None,
    highlight_dist: np.ndarray | None = None,
):
    """Renderiza o grafo dirigido ponderado da cadeia."""
    n = P.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    edge_labels = {}
    for i in range(n):
        for j in range(n):
            if P[i, j] > 1e-9:
                G.add_edge(i, j, weight=P[i, j])
                edge_labels[(i, j)] = f"{P[i,j]:.2f}"

    fig, ax = plt.subplots(figsize=(7, 5.5))
    _mpl_style(fig, ax)

    pos = (nx.circular_layout(G) if n <= 6
           else nx.spring_layout(G, seed=42, k=1.5))

    # Cores dos nós: intensidade proporcional à dist (se fornecida)
    if highlight_dist is not None:
        base = np.array([0x2e, 0x2e, 0x52]) / 255
        hi   = np.array([0x7f, 0x7f, 0xff]) / 255
        node_colors = [
            tuple(base + (hi - base) * highlight_dist[i]) for i in range(n)
        ]
    else:
        node_colors = [
            "#7f7fff" if i == highlight_node else "#2e2e52" for i in range(n)
        ]

    weights = [G[u][v]["weight"] for u, v in G.edges()]
    edge_widths = [0.8 + 3.5 * w for w in weights]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=900, linewidths=1.8,
                           edgecolors="#7f7fff")
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={i: state_names[i] for i in range(n)},
                            font_color="#e8e8f0", font_family="monospace",
                            font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color="#7f7fff", width=edge_widths, alpha=0.75,
                           arrows=True, arrowsize=20,
                           connectionstyle="arc3,rad=0.15",
                           min_source_margin=25, min_target_margin=25)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels,
                                 font_color="#c0c0ff", font_size=8,
                                 font_family="monospace",
                                 bbox=dict(boxstyle="round,pad=0.2",
                                           fc="#0f0f13", ec="none", alpha=0.7))
    ax.axis("off")
    plt.tight_layout()
    return fig


def draw_distribution_bar(values: np.ndarray, labels: list[str], title: str = ""):
    """Gráfico de barras para distribuições de probabilidade."""
    fig, ax = plt.subplots(figsize=(6, 2.6))
    _mpl_style(fig, ax)
    highlight = int(np.argmax(values))
    colors = ["#4f46e5" if i == highlight else "#2e2e52" for i in range(len(values))]
    bars = ax.bar(labels, values, color=colors, edgecolor="#7f7fff", linewidth=1.1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008, f"{val:.3f}",
                ha="center", va="bottom", color="#c0c0ff",
                fontsize=8, fontfamily="monospace")
    if title:
        ax.set_title(title, color="#888", fontsize=9, fontfamily="monospace")
    ax.set_ylabel("prob", color="#888", fontsize=8)
    ax.set_ylim(0, min(1.15, values.max() * 1.3 + 0.05))
    plt.tight_layout()
    return fig


def draw_trajectory_timeline(trajectory: list[int], state_names: list[str], max_show: int = 60):
    """Gráfico de linha da trajetória simulada."""
    traj = trajectory[:max_show]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    _mpl_style(fig, ax)
    ax.step(range(len(traj)), traj, color="#7f7fff", linewidth=1.5, where="post")
    ax.scatter(range(len(traj)), traj, color="#4f46e5", s=18, zorder=5)
    ax.set_yticks(range(len(state_names)))
    ax.set_yticklabels(state_names, fontfamily="monospace", color="#a0a0cc", fontsize=9)
    ax.set_xlabel("Passo", color="#888", fontsize=8)
    ax.set_title(f"Trajetória (primeiros {len(traj)} passos)",
                 color="#888", fontsize=9, fontfamily="monospace")
    plt.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Session state
# ═════════════════════════════════════════════════════════════════════════════
def _init_state(n: int):
    P = generate_transition_matrix(n, alpha=1.0)
    pi0 = generate_probability_vector(n, alpha=1.0)
    st.session_state.update(
        n_states=n,
        matrix=P,
        init_dist=pi0,
        state_names=[f"S{i}" for i in range(n)],
        sim_dist_results=None,
        sim_trajectory=None,
    )

if "n_states" not in st.session_state:
    _init_state(3)


# ═════════════════════════════════════════════════════════════════════════════
# Cabeçalho
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("## Cadeia de Markov")
st.markdown('<p class="section-label">simulador de processos estocásticos</p>',
            unsafe_allow_html=True)
st.divider()

left, right = st.columns([1, 1.8], gap="large")


# ═════════════════════════════════════════════════════════════════════════════
# PAINEL ESQUERDO — Configuração
# ═════════════════════════════════════════════════════════════════════════════
with left:
    st.markdown('<p class="section-label">configuração</p>', unsafe_allow_html=True)

    # ── Número de estados ──────────────────────────────────────────────────
    n_input = st.number_input(
        "Número de estados", min_value=2, max_value=20,
        value=st.session_state.n_states, step=1,
    )
    if n_input != st.session_state.n_states:
        _init_state(int(n_input))
        st.rerun()

    n = st.session_state.n_states
    state_names = st.session_state.state_names

    # ── Nomes dos estados ──────────────────────────────────────────────────
    st.markdown('<p class="section-label">nomes dos estados</p>', unsafe_allow_html=True)
    cols = st.columns(n)
    for i, col in enumerate(cols):
        with col:
            state_names[i] = st.text_input(
                "", value=state_names[i], key=f"sname_{i}",
                label_visibility="collapsed", max_chars=4,
            )

    # ── Geração aleatória ──────────────────────────────────────────────────
    with st.expander("Gerar aleatoriamente"):
        alpha = st.slider(
            "Alpha (concentração Dirichlet)", 0.1, 10.0, 1.0, 0.1,
            help="α<1 → esparsa  |  α=1 → uniforme  |  α>1 → concentrada",
        )
        if st.button("Gerar nova matriz + vetor"):
            st.session_state.matrix = generate_transition_matrix(n, alpha)
            st.session_state.init_dist = generate_probability_vector(n, alpha)
            st.session_state.sim_dist_results = None
            st.session_state.sim_trajectory = None
            st.rerun()

    # ── Matriz de transição ────────────────────────────────────────────────

    st.markdown('<p class="section-label">matriz de transição n × n</p>', unsafe_allow_html=True)

    mat_df = pd.DataFrame(
        st.session_state.matrix, index=state_names, columns=state_names,
    )

    edited_mat = st.data_editor(
        mat_df, 
        key="mat_editor", 
        on_change=update_matrix,
        use_container_width=True, 
        num_rows="fixed"
    )

    P = st.session_state.matrix

    if st.button("Normalizar matrix"):
        P = normalize_rows(np.clip(edited_mat.values.astype(float), 0, None))
        st.session_state.matrix = P
        st.rerun()


    # ── Distribuição inicial ───────────────────────────────────────────────
    st.markdown('<p class="section-label">distribuição inicial</p>',
                unsafe_allow_html=True)
    dist_df = pd.DataFrame(
        st.session_state.init_dist.reshape(1, -1), columns=state_names,
    )
    edited_dist = st.data_editor(dist_df, key="dist_editor",
                                 use_container_width=True, num_rows="fixed")
    
    pi0 = edited_dist.values.flatten().astype(float)
    if st.button("Normalizar vetor"):
        pi0 = normalize_vector(edited_dist.values.flatten().astype(float))
        st.session_state.init_dist = pi0

    # ── Parâmetros da simulação ────────────────────────────────────────────
    st.markdown('<p class="section-label">simulação</p>', unsafe_allow_html=True)
    n_steps   = st.slider("Passos (distribuição)", 1, 100, 20)
    traj_size = st.slider("Tamanho da trajetória", 100, 20_000, 5_000, 100)

    if st.button("▶  Simular"):
        # Evolução da distribuição: π_{t+1} = π_t @ P
        dist_results = [pi0.copy()]
        d = pi0.copy()
        for _ in range(n_steps):
            d = d @ P
            dist_results.append(d.copy())
        st.session_state.sim_dist_results = dist_results

        # Trajetória estocástica via others.simulate_trajectory
        st.session_state.sim_trajectory = simulate_trajectory(P, pi0, traj_size)


# ═════════════════════════════════════════════════════════════════════════════
# PAINEL DIREITO — Visualizações
# ═════════════════════════════════════════════════════════════════════════════
with right:
    tab_graph, tab_sim, tab_traj, tab_stats = st.tabs(
        ["Grafo", "Distribuição", "Trajetória", "Diagnóstico"]
    )

    # ── Aba: Grafo dirigido ───────────────────────────────────────────────
    with tab_graph:
        st.markdown('<p class="section-label">grafo dirigido</p>',
                    unsafe_allow_html=True)
        fig = draw_markov_graph(P, state_names)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("Ver matriz normalizada"):
            P = normalize_rows(np.clip(edited_mat.values.astype(float), 0, None))
            st.dataframe(
                pd.DataFrame(P, index=state_names, columns=state_names)
                .style.format("{:.4f}")
                .background_gradient(cmap="Blues"),
                use_container_width=True,
            )

    # ── Aba: Evolução da distribuição ─────────────────────────────────────
    with tab_sim:
        if st.session_state.sim_dist_results is None:
            st.info("Clique em **▶ Simular** para ver a evolução da distribuição.")
        else:
            results = st.session_state.sim_dist_results
            st.markdown('<p class="section-label">evolução de π ao longo do tempo</p>',
                        unsafe_allow_html=True)

            chart_df = pd.DataFrame(results, columns=state_names)
            chart_df.index.name = "Passo"
            st.line_chart(chart_df, use_container_width=True, height=260)

            # Grafo colorido pelo passo selecionado
            st.markdown('<p class="section-label">distribuição no passo selecionado</p>',
                        unsafe_allow_html=True)
            step = st.slider("Passo", 0, len(results) - 1, 0, key="dist_step")
            dist_step = results[step]

            fig2 = draw_markov_graph(P, state_names, highlight_dist=dist_step)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

            fig3 = draw_distribution_bar(dist_step, state_names,
                                         title=f"π no passo {step}")
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)

            with st.expander("Tabela completa"):
                st.dataframe(
                    chart_df.style.format("{:.5f}")
                    .background_gradient(cmap="Blues", axis=0),
                    use_container_width=True,
                )

    # ── Aba: Trajetória ───────────────────────────────────────────────────
    with tab_traj:
        if st.session_state.sim_trajectory is None:
            st.info("Clique em **▶ Simular** para gerar uma trajetória.")
        else:
            traj = st.session_state.sim_trajectory
            n_states = st.session_state.n_states

            st.markdown('<p class="section-label">trajetória estocástica</p>',
                        unsafe_allow_html=True)

            fig_t = draw_trajectory_timeline(traj, state_names)
            st.pyplot(fig_t, use_container_width=True)
            plt.close(fig_t)

            # Distribuição empírica
            st.markdown('<p class="section-label">distribuição empírica</p>',
                        unsafe_allow_html=True)
            counts = np.bincount(traj, minlength=n_states)
            emp_dist = counts / len(traj)
            fig_e = draw_distribution_bar(emp_dist, state_names,
                                          title=f"Empírica ({len(traj)} passos)")
            st.pyplot(fig_e, use_container_width=True)
            plt.close(fig_e)

            # Comparação empírica vs teórica
            pi_stat = get_stationary_distribution(P)
            if pi_stat is not None:
                st.markdown('<p class="section-label">empírica vs estacionária teórica</p>',
                            unsafe_allow_html=True)
                cmp_df = pd.DataFrame(
                    {"Empírica": emp_dist, "Estacionária": pi_stat},
                    index=state_names,
                )
                st.bar_chart(cmp_df, use_container_width=True, height=220, stack=False)

    # ── Aba: Diagnóstico ──────────────────────────────────────────────────
    with tab_stats:
        st.markdown('<p class="section-label">propriedades da cadeia</p>',
                    unsafe_allow_html=True)

        # Usa diagnose_with_graphs de others.py
        is_irred, is_aperiodic, scc_list = diagnose_with_graphs(P)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-val">{"✓" if is_irred else "✗"}</div>'
                f'<div class="metric-label">Irredutível</div></div>',
                unsafe_allow_html=True)
        with c2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-val">{"✓" if is_aperiodic else "✗"}</div>'
                f'<div class="metric-label">Aperiódica</div></div>',
                unsafe_allow_html=True)
        with c3:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-val">{len(scc_list)}</div>'
                f'<div class="metric-label">Comp. fortemente conexas</div></div>',
                unsafe_allow_html=True)

        st.markdown("&nbsp;", unsafe_allow_html=True)

        # SCCs
        st.markdown('<p class="section-label">classes comunicantes</p>',
                    unsafe_allow_html=True)
        for idx, scc in enumerate(sorted(scc_list, key=len, reverse=True)):
            names = " · ".join(state_names[i] for i in sorted(scc))
            size_lbl = f"{len(scc)} estado{'s' if len(scc) > 1 else ''}"
            cls = "scc-chip absorb" if len(scc) > 1 else "scc-chip"
            st.markdown(
                f'<span class="{cls}">C{idx+1}: {names} &nbsp;<small>({size_lbl})</small></span>',
                unsafe_allow_html=True,
            )

        # Distribuição estacionária
        st.markdown("&nbsp;", unsafe_allow_html=True)
        pi_stat = get_stationary_distribution(P)

        if pi_stat is not None:
            st.markdown('<p class="section-label">distribuição estacionária π (QR)</p>',
                        unsafe_allow_html=True)
            pi_df = pd.DataFrame(pi_stat.reshape(1, -1), columns=state_names)
            st.dataframe(
                pi_df.style.format("{:.6f}").background_gradient(cmap="Purples"),
                use_container_width=True,
            )
            fig_pi = draw_distribution_bar(pi_stat, state_names, title="π estacionária")
            st.pyplot(fig_pi, use_container_width=True)
            plt.close(fig_pi)

            # Verificação: π @ P ≈ π
            residual = np.max(np.abs(pi_stat @ P - pi_stat))
            st.caption(f"Verificação  ‖π P − π‖∞ = {residual:.2e}")
        else:
            st.warning("Não foi possível calcular a distribuição estacionária.")