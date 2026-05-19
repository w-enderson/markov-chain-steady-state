import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from markov_generators  import generate_transition_matrix, generate_probability_vector
from others             import simulate_trajectory, diagnose_with_graphs, normalize_rows, normalize_vector, get_stationary_distribution
from markov_animations  import create_distribution_animation, create_step_animation


# ══════════════════════════════════════════════════════════════════════════════
# Página
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Cadeia de Markov", page_icon="🔗", layout="wide")

st.markdown('''
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

.lbl {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    color: #5555aa;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.25rem;
}

.metric-card {
    background: #1c1c28;
    border: 1px solid #2e2e42;
    border-radius: 12px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.55rem;
    color: #7f7fff;
    font-weight: 700;
    line-height: 1.35;
}
.metric-label {
    font-size: 0.68rem;
    color: #777;
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
    font-size: 0.76rem;
    color: #a0a0cc;
    margin: 0.2rem;
}
.scc-chip.hi { border-color: #7f7fff; color: #c0c0ff; }

.hint {
    font-size: 0.78rem;
    color: #555;
    font-style: italic;
    margin-top: 0.3rem;
}
</style>
''', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Visualizações estáticas (matplotlib)
# ══════════════════════════════════════════════════════════════════════════════

def _mpl_dark(fig, ax):
    fig.patch.set_facecolor("#0f0f13")
    ax.set_facecolor("#0f0f13")
    for s in ax.spines.values():
        s.set_edgecolor("#2e2e42")
    ax.tick_params(colors="#888")

def update_matrix():
    edited_data = st.session_state.mat_ed["edited_rows"]
    
    for row_idx, cols in edited_data.items():
        for col_name, new_val in cols.items():
            col_idx = state_names.index(col_name)
            st.session_state.matrix[int(row_idx), col_idx] = float(new_val)

def draw_static_graph(P: np.ndarray, state_names: list[str]):
    n = P.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    edge_labels = {}
    for i in range(n):
        for j in range(n):
            if P[i, j] > 1e-9:
                G.add_edge(i, j, weight=P[i, j])
                edge_labels[(i, j)] = f"{P[i,j]:.2f}"

    fig, ax = plt.subplots(figsize=(6.5, 5))
    _mpl_dark(fig, ax)
    pos = nx.circular_layout(G) if n <= 7 else nx.spring_layout(G, seed=42, k=1.6)
    weights = [G[u][v]["weight"] for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#2e2e52",
                           node_size=900, linewidths=2, edgecolors="#7f7fff")
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={i: state_names[i] for i in range(n)},
                            font_color="#e8e8f0", font_family="monospace",
                            font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#7f7fff",
                           width=[0.8 + 3 * w for w in weights], alpha=0.7,
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


def draw_bar(values: np.ndarray, labels: list[str],
             highlight: int | None = None, title: str = ""):
    hi = highlight if highlight is not None else int(np.argmax(values))
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    _mpl_dark(fig, ax)
    colors = ["#4f46e5" if i == hi else "#2e2e52" for i in range(len(values))]
    bars = ax.bar(labels, values, color=colors, edgecolor="#7f7fff", linewidth=1.1)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom",
                color="#c0c0ff", fontsize=8, fontfamily="monospace")
    if title:
        ax.set_title(title, color="#888", fontsize=9, fontfamily="monospace")
    ax.set_ylabel("prob", color="#888", fontsize=8)
    ax.set_ylim(0, min(1.2, values.max() * 1.35 + 0.05))
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════
def _init(n: int):
    P   = generate_transition_matrix(n, alpha=1.0)
    pi0 = generate_probability_vector(n, alpha=1.0)
    st.session_state.update(
        n_states       = n,
        matrix         = P,
        init_dist      = pi0,
        state_names    = [f"S{i}" for i in range(n)],
        sim_dist       = None,
        sim_traj       = None,
        anim_dist_fig  = None,
        anim_traj_fig  = None,
    )

if "n_states" not in st.session_state:
    _init(3)


# ══════════════════════════════════════════════════════════════════════════════
# Cabeçalho
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Simulador de Cadeia de Markov")
st.divider()

left, right = st.columns([1, 1.85], gap="large")


# ══════════════════════════════════════════════════════════════════════════════
# PAINEL ESQUERDO
# ══════════════════════════════════════════════════════════════════════════════
with left:
    st.markdown('<p class="lbl">Número de estados</p>', unsafe_allow_html=True)

    n_input = st.number_input("",min_value=2, max_value=30,
                              value=st.session_state.n_states, step=1)
    if int(n_input) != st.session_state.n_states:
        _init(int(n_input))
        st.rerun()

    n           = st.session_state.n_states
    state_names = st.session_state.state_names

    # Geração aleatória
    st.markdown('<p class="lbl">Gerar aleatoriamente</p>', unsafe_allow_html=True)

    alpha = st.slider("Alpha (Dirichlet)", 0.1, 10.0, 1.0, 0.1,
                        help="α<1 esparsa · α=1 uniforme · α>1 concentrada")
    if st.button("Nova matriz + vetor"):
        st.session_state.matrix    = generate_transition_matrix(n, alpha)
        st.session_state.init_dist = generate_probability_vector(n, alpha)
        st.session_state.sim_dist  = None
        st.session_state.sim_traj  = None
        st.session_state.anim_dist_fig = None
        st.session_state.anim_traj_fig = None
        st.rerun()

    # Matriz de transição
    st.markdown('<p class="lbl">matriz de transição n × n</p>', unsafe_allow_html=True)
    mat_df = pd.DataFrame(st.session_state.matrix,
                          index=state_names, columns=state_names)
    edited = st.data_editor(mat_df, key="mat_ed",
                            use_container_width=True,
                            num_rows="fixed", 
                            on_change=update_matrix)

    P = st.session_state.matrix
    if st.button("Normalizar matrix"):
        P = normalize_rows(np.clip(edited.values.astype(float), 0, None))
        st.session_state.matrix = P
        st.rerun()

    # Distribuição inicial
    st.markdown('<p class="lbl">distribuição inicial</p>', unsafe_allow_html=True)
    dist_df  = pd.DataFrame(st.session_state.init_dist.reshape(1, -1),
                            columns=state_names)
    edited_d = st.data_editor(dist_df, key="dist_ed",
                               use_container_width=True, num_rows="fixed")


    pi0 = edited_d.values.flatten().astype(float)
    if st.button("Normalizar vetor"):
        pi0 = normalize_vector(edited_d.values.flatten().astype(float))
        st.session_state.init_dist = pi0

    # Parâmetros
    st.markdown('<p class="lbl">parâmetros</p>', unsafe_allow_html=True)
    n_steps    = st.slider("Passos de distribuição",  5,   150,  30)
    traj_size  = st.slider("Tamanho da trajetória", 100, 20000, 5000, 100)
    anim_steps = st.slider("Passos a animar",         20,  300,  100)

    spd_dist = 650
    spd_traj = 800

    if st.button("▶  Simular"):
        # Evolução de π(t) — distribuição
        dist_acc = [pi0.copy()]
        d = pi0.copy()
        for _ in range(n_steps):
            d = d @ P
            dist_acc.append(d.copy())
        st.session_state.sim_dist = dist_acc

        # Trajetória estocástica
        traj = simulate_trajectory(P, pi0, traj_size)
        st.session_state.sim_traj = traj

        # Animação da evolução de π(t)
        st.session_state.anim_dist_fig = create_distribution_animation(
            P, state_names, dist_acc, frame_duration=spd_dist,
        )

        # Animação de trajetória — usa create_step_animation
        st.session_state.anim_traj_fig = create_step_animation(
            P, state_names, traj[:anim_steps],
            frame_duration=spd_traj,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAINEL DIREITO
# ══════════════════════════════════════════════════════════════════════════════
with right:
    tab_graph, tab_anim_dist, tab_anim_traj = st.tabs([
        "Grafo",
        "Evolução de π(t)",
        "Trajetória",
    ])

    # ── Aba 1: grafo estático ─────────────────────────────────────────────
    with tab_graph:
        st.markdown('<p class="lbl">grafo dirigido</p>', unsafe_allow_html=True)
        fig_g = draw_static_graph(P, state_names)
        st.pyplot(fig_g, use_container_width=True)
        plt.close(fig_g)
        
        st.markdown('<p class="lbl">propriedades da cadeia</p>',
                    unsafe_allow_html=True)

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

        st.markdown('<p class="lbl">classes comunicantes</p>',
                    unsafe_allow_html=True)
        for idx, scc in enumerate(sorted(scc_list, key=len, reverse=True)):
            names = " · ".join(state_names[i] for i in sorted(scc))
            cls   = "scc-chip hi" if len(scc) > 1 else "scc-chip"
            st.markdown(
                f'<span class="{cls}">C{idx+1}: {names} &nbsp;'
                f'<small>({len(scc)} estado{"s" if len(scc)>1 else ""})'
                f'</small></span>',
                unsafe_allow_html=True,
            )

        st.markdown("&nbsp;", unsafe_allow_html=True)
        pi_stat = get_stationary_distribution(P)

        if pi_stat is not None:
            st.markdown(
                '<p class="lbl">distribuição estacionária </p>',
                unsafe_allow_html=True)
            pi_df = pd.DataFrame(pi_stat.reshape(1, -1), columns=state_names)

            fig_pi = draw_bar(pi_stat, state_names, title="π estacionária")
            st.pyplot(fig_pi, use_container_width=True)
            plt.close(fig_pi)
            residual = np.max(np.abs(pi_stat @ P - pi_stat))
            st.caption(f"Verificação  ‖π P - π‖  ͚   = {residual:.2e}")
        else:
            st.warning("Não foi possível calcular a distribuição estacionária.")

    # ── Aba 2: animação da distribuição ───────────────────────────────────
    with tab_anim_dist:
        if st.session_state.anim_dist_fig is None:
            st.info("Clique em **▶ Simular** para gerar a animação.")
            st.markdown(
                '<p class="hint">Mostra como π(t) evolui passo a passo — '
                'nós do grafo crescem/colorem conforme a probabilidade aumenta. '
                'Use ▶ Play ou arraste o slider.</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<p class="lbl">evolução de π(t)</p>',
                        unsafe_allow_html=True)
            st.plotly_chart(st.session_state.anim_dist_fig,
                            use_container_width=True, key="fig_dist")

            st.markdown('<p class="lbl">evolução de π com o tempo</p>',
                        unsafe_allow_html=True)
            df_evo = pd.DataFrame(st.session_state.sim_dist, columns=state_names)
            df_evo.index.name = "Passo"
            st.line_chart(df_evo, use_container_width=True, height=190)

    # ── Aba 3: animação da trajetória (passo a passo) ─────────────────────
    with tab_anim_traj:
        if st.session_state.anim_traj_fig is None:
            st.info("Clique em **▶ Simular** para gerar a animação.")
            st.markdown(
                '<p class="hint">Percorre a cadeia estado a estado — '
                'a seta iluminada destaca a transição atual no grafo, '
                'enquanto a fita lateral mostra o histórico recente de estados visitados.</p>',
                unsafe_allow_html=True,
            )
        else:
            traj = st.session_state.sim_traj
            st.markdown('<p class="lbl">transição estado a estado</p>',
                        unsafe_allow_html=True)
            st.plotly_chart(st.session_state.anim_traj_fig,
                            use_container_width=True, key="fig_traj")

            # Empírica vs estacionária
            st.markdown('<p class="lbl">empírica vs teórica</p>',
                        unsafe_allow_html=True)
            counts   = np.bincount(traj, minlength=n)
            emp_dist = counts / len(traj)
            pi_stat  = get_stationary_distribution(P)

            if pi_stat is not None:
                cmp_df = pd.DataFrame(
                    {"Empírica": emp_dist, "Estacionária (QR)": pi_stat},
                    index=state_names,
                )
                st.bar_chart(cmp_df, use_container_width=True, height=200, stack=False)
            else:
                fig_e = draw_bar(emp_dist, state_names,
                                 title=f"Empírica ({len(traj)} passos)")
                st.pyplot(fig_e, use_container_width=True)
                plt.close(fig_e)