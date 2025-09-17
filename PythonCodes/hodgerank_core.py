import pandas as pd
import numpy as np
import networkx as nx
from scipy.sparse.linalg import lsqr
import math
import scipy.sparse as ss
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from adjustText import adjust_text
import io
import random

# ================================================================
# ======================= BACKEND CORE ===========================
# ================================================================

def preprocess_data(raw_data: pd.DataFrame):
    header = raw_data.iloc[0, 1:].astype(str).tolist()   # Items
    ids = raw_data.iloc[1:, 0].astype(str).tolist()      # IDs of the voters

    data = raw_data.transpose()
    data = data.drop(columns=0)   # remove header
    data = data.drop(index=0)     # remove IDs
    data = data.astype(np.float64)
    data.columns = ids
    data.index = header
    return data

def apply_cut(data: pd.DataFrame, cut_percentage: float):
    binary_dataframe = (data.copy() != 0).astype(np.int8)
    n_items, n_voters = binary_dataframe.shape
    n_items_to_remove = math.floor(n_items * cut_percentage/100)

    if n_items == 0 or cut_percentage <= 0 or n_items_to_remove == 0:
        feedback = {
            "num_items_removed": 0,
            "num_items_original": n_items,
            "num_items_remaing": n_items,
            "dropped": [],
            "note": "No items removed."}
        return data, feedback

    info = []
    for item in binary_dataframe.index:
        count = binary_dataframe.loc[item].sum()
        frequency = 100*(count/n_voters)
        info.append(frequency)

    frequency_dataframe = pd.DataFrame(info, columns=["Frequency (%)"], index= binary_dataframe.index).sort_values(by="Frequency (%)", ascending=True)

    to_drop = frequency_dataframe.iloc[:n_items_to_remove]
    remaining = data.copy().drop(index=to_drop.index)

    feedback = {
        "num_items_removed": to_drop.shape[0],
        "num_items_original": n_items,
        "num_items_remaing": remaining.shape[0],
        "dropped": to_drop,
        "note": "Removed from the START of the ascending-sorted frequency list (least frequent items)."}
    return remaining, feedback

def compute_hodgerank(info:pd.DataFrame):
    binary_data_df = (info != 0).astype(np.int8)
    data_df = info.copy()
    itens = data_df.index
    n_voters = len(data_df.columns)
    n_itens = len(itens)
        
    def weighted_adjacency_matrix(binaryM):
        matrix = binaryM.values @ binaryM.values.T
        matrix = matrix.astype(np.int32)
        np.fill_diagonal(matrix, 0)
        return pd.DataFrame(matrix, index= binaryM.index, columns= binaryM.index)

    def edges_flows_matrix(datamatrix: pd.DataFrame, weightmatrix: pd.DataFrame):
        num_itens = len(datamatrix.index)
        matrix = np.zeros((num_itens, num_itens), dtype=np.float64)
        
        for idxi, item1 in enumerate(datamatrix.index):
            for idxj, item2 in enumerate(datamatrix.index):
                
                if idxi >= idxj: continue
                
                elif weightmatrix.loc[item1, item2] != 0:
                    flow_sum = 0
                    for votante in datamatrix.columns:
                        if datamatrix.loc[item1, votante] * datamatrix.loc[item2, votante] != 0:
                            flow_sum += datamatrix.loc[item2, votante] - datamatrix.loc[item1, votante]
                    matrix[idxi, idxj] = flow_sum / weightmatrix.loc[item1, item2]
                    matrix[idxj, idxi] = (-1) * matrix[idxi, idxj]

        return pd.DataFrame(matrix, index= datamatrix.index, columns= datamatrix.index)

    weight_adjacency_matrix_df = weighted_adjacency_matrix(binary_data_df)
    adjacency_matrix_df= (weight_adjacency_matrix_df > 0).astype(np.int8)
    flow_matrix_df = edges_flows_matrix(data_df, weight_adjacency_matrix_df)
    
    nodes = itens
    edges = []

    for idxi, item_1 in enumerate(itens):
        for idxj, item_2 in enumerate(itens):
            if idxi<idxj and adjacency_matrix_df.loc[item_1, item_2] == 1: edges.append((item_1, item_2))
     
    number_of_edges = len(edges)

    def build_incidence_matrix(nodelist, edgelist):
        zeros = np.zeros((len(nodelist), len(edgelist)),dtype=np.int8)
        incidence_matrix_df = pd.DataFrame(zeros, index= nodelist, columns= [str(e) for e in edgelist])

        for e in edgelist:
            node1, node2 = e
            e = str(e)
            incidence_matrix_df.loc[node1, e] = -1
            incidence_matrix_df.loc[node2, e] = 1
       
        return incidence_matrix_df
    
    incidence_matrix_df = build_incidence_matrix(nodes, edges)
    gradient_matrix_df = incidence_matrix_df.transpose()

    flow_vector_df = pd.DataFrame([flow_matrix_df.loc[item1, item2] for item1, item2 in edges], index= [str(e) for e in edges], columns= ["Flows"])    
    weight_edges_vector_df = pd.DataFrame([weight_adjacency_matrix_df.loc[item1, item2] for item1, item2 in edges], index=[str(e) for e in edges], columns= ["Comparisons"])
    
    edgesweight = ss.diags(weight_edges_vector_df.values.flatten(), shape=(number_of_edges, number_of_edges), format="csr", dtype= np.int32)
    inv_edgesweight = ss.diags(1.0/weight_edges_vector_df.values.flatten(), shape= (number_of_edges, number_of_edges), format="csr", dtype= np.float64)

    def find_triangles(adjacency: pd.DataFrame):
        num_itens = len(adjacency.index)
        triangles = []

        for idxi, item1 in enumerate(adjacency.index):
            for idxj, item2 in enumerate(adjacency.index):
                
                if idxi >= idxj: continue
                else: 
                    if adjacency.loc[item1, item2] == 1:          # Verify if exist a edge between item1 and item2
                        
                        for idxk, item3 in enumerate(adjacency.index):    
                            if idxk <= idxj: continue
                            else:
                                if adjacency.loc[item1, item3] == 1 and adjacency.loc[item2, item3] == 1:   # Verify a item3 that have edges to item1 and item2
                                    triangles.append((item1, item2, item3))                                     

        return triangles
    
    all_triangles = find_triangles(adjacency_matrix_df)
    num_triangles = len(all_triangles)

    def build_curl_matrix(triangleslist, edgeslist):
        zeros = np.zeros((len(triangleslist), len(edgeslist)), dtype=np.int8)
        curl_matrix_df = pd.DataFrame(zeros, index= [str(t) for t in triangleslist], columns= [str(e) for e in edgeslist])

        for t in triangleslist:
    
            edge1 = str((t[0], t[1]))
            edge2 = str((t[1], t[2]))
            edge3 = str((t[0], t[2]))
            
            t = str(t)

            curl_matrix_df.loc[t, edge1] = 1
            curl_matrix_df.loc[t, edge2] = 1
            curl_matrix_df.loc[t, edge3] = -1
       
        return curl_matrix_df

    curl_matrix_df = build_curl_matrix(all_triangles, edges)

    # Weight for the triangles with harmonic median
    # --------------------------------------------------------------------

    def calculate_weight_triangle(triangleslist, weightsedges: pd.DataFrame):
        list = []
        for t in triangleslist:

            edge1 = str((t[0], t[1]))
            edge2 = str((t[1], t[2]))
            edge3 = str((t[0], t[2]))

            weight_edge1 = weightsedges.loc[edge1, "Comparisons"]
            weight_edge2 = weightsedges.loc[edge2, "Comparisons"]
            weight_edge3 = weightsedges.loc[edge3, "Comparisons"]

            list.append(1/(1/weight_edge1 + 1/weight_edge2 + 1/weight_edge3))
          
        return pd.DataFrame(list, index=[str(t) for t in all_triangles], columns= ["HarmonicWeight"])

    weight_triangles_vector_df = calculate_weight_triangle(all_triangles, weight_edges_vector_df)

    trianglesweight = ss.diags(weight_triangles_vector_df.values.flatten(), shape=(num_triangles, num_triangles), format="csr", dtype= np.float64)

    curl_matrix = ss.csr_matrix(curl_matrix_df.values)
    curl_matrix_adjoint = inv_edgesweight @ curl_matrix.T @ trianglesweight

    # Encontrar Yc (parte do curl- inconsistências)
    solution = lsqr(A= curl_matrix_adjoint.T @ edgesweight @ curl_matrix_adjoint,
                    b= curl_matrix_adjoint.T @ edgesweight @ flow_vector_df.values.flatten(),
                   atol= 1e-15, btol= 1e-15)[0]
   
    Yc = curl_matrix_adjoint @ solution

    # Encontrar Yg (parte do gradiente - ranking)

    r = flow_vector_df.values.flatten() - Yc

    solution = lsqr(A = ss.csr_matrix(gradient_matrix_df.values).T @ edgesweight @ ss.csr_matrix(gradient_matrix_df.values),
                b = gradient_matrix_df.values.T @ edgesweight @ r,
                atol=1e-15, btol=1e-15)[0]
    
    Yg = gradient_matrix_df.values @ solution
    Yh = r - Yg                               # Harmonic part

    potentials_df = pd.DataFrame(solution, index=itens, columns= ["Potential"]).sort_values(by= "Potential", ascending=False)
    ranking = []
    for idxi, item in enumerate(potentials_df.index):
        ranking.append(f"{idxi+1}°")
    potentials_df["Rank"] = ranking

    informations = []
    for item in itens:
        num_votes = binary_data_df.loc[item].sum()
        total_evaluation = data_df.loc[item].sum()
        median = total_evaluation/num_votes
        potencial = potentials_df.loc[item, "Potential"]
        rank = potentials_df.loc[item, "Rank"]
        frequency = 100 * num_votes/n_voters
        
        informations.append([rank, potencial, frequency, num_votes, total_evaluation, median])
    informations = pd.DataFrame(informations, index= itens, 
                                columns= ["Rank", "Potential Score", "Evaluation Frequency (%)", "Number of Votes", "Total Score", "Average Score"])
    
    norm_flow = flow_vector_df.values.T @ edgesweight @ flow_vector_df.values
    norm_curl = Yc.T @ edgesweight @ Yc
    norm_grad = Yg.T @ edgesweight @ Yg
    norm_harmonic = Yh.T @ edgesweight @ Yh
   
    potential_matrix_df = pd.DataFrame(np.zeros((n_itens, n_itens), dtype= np.float64), index= itens, columns= itens)

    for idxi, item1 in enumerate(itens):
        for idxj, item2 in enumerate(itens):
            if idxi >= idxj: continue
            
            potential_matrix_df.loc[item1, item2] = informations.loc[item2, "Potential Score"] - informations.loc[item1, "Potential Score"]
            potential_matrix_df.loc[item2, item1] = informations.loc[item1, "Potential Score"] - informations.loc[item2, "Potential Score"]

    residuals = flow_matrix_df - potential_matrix_df
    curl_values = pd.DataFrame(curl_matrix_df.values @ Yc, index= [str(t) for t in all_triangles], columns= ["Value"])

    final_results = {
        "rankingtable": informations,
        "weightadjacencymatrix": weight_adjacency_matrix_df,
        "flowmatrix": flow_matrix_df,
        "gradientmatrix": gradient_matrix_df,
        "curlmatrix": curl_matrix_df,
        "flowvector": flow_vector_df,      # vector that was decomposed
        "edgeweightvector": weight_edges_vector_df,
        "trianglesweightvector": weight_triangles_vector_df,
        "edgeweightmatrix": edgesweight,
        "triangleweightmatrix": trianglesweight,
        "nodes": nodes,
        "edges": edges,
        "triangles": all_triangles,
        "gradientcomponent": Yg,
        "curlcomponent": Yc,
        "harmoniccomponent": Yh,
        "gradientnorm": norm_grad,
        "curlnorm": norm_curl,
        "harmonicnorm": norm_harmonic,
        "flownorm": norm_flow[0][0],
        "residuals": residuals,
        "curlvalues": curl_values,
        "potentialmatrix": potential_matrix_df,
        "numbercontexts": n_voters}
    
    return final_results

# ================================================================
# ======================= PLOTS CORE =============================
# ================================================================

def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def plot_decomposition_pie(grad_share, curl_share, harm_share):
    if grad_share == 0.0 and curl_share == 0.0 and harm_share == 0.0:
        return None
    if harm_share <= 1e-6:
        values = [grad_share, curl_share]
        labels = ["Consistency", "Local inconsistencies"]
        colors = ["#6BA292", "#E17C05"]

    else:
        values = [grad_share, curl_share, harm_share]
        labels = ["Consistency", "Local inconsistencies", "Global inconsistencies"]
        colors = ["#6BA292", "#E17C05", "#5D3A9B"]

    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    wedges, _, autotexts = ax.pie(
        values,
        labels=None,
        colors=colors,
        startangle=90,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 0 else "",
        textprops=dict(color="white", fontsize=7, weight="bold")
    )
    ax.legend(wedges, labels, loc="lower right", bbox_to_anchor=(1.2, -0.05), fontsize=8, frameon=True)
    ax.axis("equal")
    return fig

def plot_scatter_potentials_freq(items, potentials, freqs, size, letters, fontsize):
    
    fig, ax = plt.subplots(figsize=size)
    ax.scatter(potentials, freqs, color="purple", alpha=0.7)

    texts = []
    for i, label in enumerate(items):
        short_label = label[:letters] + "..." if len(label) > letters else label
        texts.append(ax.text(potentials[i], freqs[i], short_label, fontsize=fontsize))

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color="gray", lw=1),force_text=(10,10), force_static=(10,10))
    ax.set_xlabel("Potentials", fontsize=11)
    ax.set_ylabel("Frequency (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)
    return fig

def plot_graph(items, edges, weightedges, residuals,
               layout1="kamada", layout2="kamada",
               highlight="default",
               potentials=None, freqs=None,
               labels1 = (0,0.05), labels2 = (0,0.05)):

    def abbreviate(label: str) -> str:
        return label if len(label) <= 5 else label[:5] + "..."

    abbrev_map = {node: abbreviate(node) for node in items}

    def get_positions(layout_type, wgraph):
        if layout_type == "default" and potentials is not None and freqs is not None:
            return {item: (potentials[i], freqs[i]) for i, item in enumerate(items)}
        else:
            G_tmp = nx.Graph()
            G_tmp.add_nodes_from(items)

            if wgraph:
                G_tmp.add_edges_from(edges)
            
            else:
               G_tmp.add_edges_from(edges, weight=None)
    
            return nx.kamada_kawai_layout(G_tmp)

    # -------------------------
    # Weighted edges graph
    # -------------------------
    def plot_weighted_graph(pos):
        edge_weights = [weightedges.loc[str(e), "Comparisons"] for e in edges]
        max_w = max(edge_weights) if edge_weights else 1

        widths = [0.25 + 2.0 * (w / max_w) for w in edge_weights]
        alphas = [0.15 + 0.8 * (w / max_w) for w in edge_weights]

        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw_networkx_nodes(nx.Graph(edges), pos,
                               node_size=100, node_color="white",
                               edgecolors="black", linewidths=0.8, ax=ax)

        for node, (x, y) in pos.items():
            ax.text(x + labels1[0], y + labels1[1], abbrev_map[node], fontsize=8, ha="center")

        for (u, v), lw, a in zip(edges, widths, alphas):
            direction = random.choice([1,2])
         
            arrow = FancyArrowPatch(pos[u], pos[v], connectionstyle=f"arc3,rad={random.choice([-0.25,0.25])}",
                                    arrowstyle="-", color="gray", linewidth=lw, alpha=a)
            ax.add_patch(arrow)

        ax.axis("off")
        return fig

    # -------------------------
    # Residuals graph
    # -------------------------
    def plot_residual_graph(pos):
        def draw_residual_arrow(ax, u, v, res, color, lw, alpha):
            if res >= 0:
                start, end = pos[u], pos[v]
            else:
                start, end = pos[v], pos[u]
            arrow = FancyArrowPatch(start, end, connectionstyle="arc3,rad=0.2",
                                    arrowstyle="-|>", mutation_scale=11,
                                    color=color, linewidth=lw, alpha=alpha, zorder=3)
            ax.add_patch(arrow)

        edgeresidual = [residuals.loc[item1, item2] for item1, item2 in edges]
        max_w = max(abs(v) for v in edgeresidual) or 1
        mean_abs = np.mean([abs(v) for v in edgeresidual]) if edgeresidual else 0

        widths = [0.05 + 3 * (abs(v) / max_w) for v in edgeresidual]
        alphas = [0.005 + 0.8 * (abs(v) / max_w) for v in edgeresidual]

        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw_networkx_nodes(nx.Graph(edges), pos,
                               node_size=100, node_color="white",
                               edgecolors="black", linewidths=0.8, ax=ax)

        for node, (x, y) in pos.items():
            ax.text(x + labels2[0], y + labels2[1], abbrev_map[node], fontsize=8, ha="center")

        for (u, v), lw, a, res in zip(edges, widths, alphas, edgeresidual):

            if highlight == "positive" and res <= 0:
                continue
            if highlight == "negative" and res >= 0:
                continue

            if res > 0:
                color = "green"
            else:
                color = "red"

            draw_residual_arrow(ax, u, v, res, color, lw, a)

        ax.axis("off")
        return fig

    fig1 = plot_weighted_graph(get_positions(layout1, wgraph=True))
    fig2 = plot_residual_graph(get_positions(layout2,wgraph=False))
    return fig1, fig2

def plot_matrix(matrix_df, title, cbar_label, part="lower"):

    """
    Plots a heatmap of a matrix.

    Parameters:
    - matrix_df: pd.DataFrame, the matrix to plot
    - title: str, plot title
    - cbar_label: str, colorbar label
    - part: str, one of 'lower', 'upper', 'full' to choose which part of the matrix to display
    """

    vals = matrix_df.values.copy()

    if part == "lower":
        vals = np.tril(vals, k=-1)  # keep lower triangle only
    elif part == "upper":
        vals = np.triu(vals, k=1)  # keep upper triangle only

    np.fill_diagonal(vals, 0)

    vmax = np.max(np.abs(vals)) if np.max(np.abs(vals)) > 0 else 1

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("white")

    im = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="none", aspect="equal")

    ax.set_xticks(range(len(matrix_df)))
    ax.set_yticks(range(len(matrix_df)))
    ax.set_xticklabels(matrix_df.index, rotation=65, ha="right", fontsize=7)
    ax.set_yticklabels(matrix_df.index, fontsize=7)
    
    if part != "full":
        ax.plot([0, len(matrix_df)-1], [0, len(matrix_df)-1], color="gray", linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=13, weight="bold", pad=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(cbar_label, fontsize=9)

    ax.spines[:].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    fig.tight_layout()
    return fig