import streamlit as st
import pandas as pd
import numpy as np
import hodgerank_core as hc
import os
import io

def download_csv(data: pd.DataFrame, filename: str = "RankingTable.csv"):
    csv = data.round(4).to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {filename}",
        data=csv,
        file_name=filename,
        mime="text/csv")

def download_excel(data: pd.DataFrame, filename: str = "RankingTable.xlsx"):
    buffer = io.BytesIO()
    data.round(4).to_excel(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label=f"Download {filename}",
        data=buffer,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def download_png(fig, filename: str = "figure.png"):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    st.download_button(
        label=f"Download {filename}",
        data=buffer,
        file_name=filename,
        mime="image/png")

def download_pdf(fig, filename: str = "figure.pdf"):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    st.download_button(
        label=f"Download {filename}",
        data=buffer,
        file_name=filename,
        mime="application/pdf")

st.set_page_config(page_title="HodgeRank Analyzer", layout="wide")

# -----------------------------
# Initialize step state
# -----------------------------
if "step" not in st.session_state:
    st.session_state["step"] = 1
    
step = st.session_state["step"]

# =========================================================================================================================
# STEP 1 – Informations
# =========================================================================================================================
if step == 1:

    st.markdown("""
        # 1. HodgeRank Analyzer – Introduction

        The **HodgeRank method** is a mathematical framework located at the intersection of  
        **linear algebra**, **graph theory**, and **optimization techniques**. It belongs to  
        a broader field called *combinatorial Hodge theory*, which investigates how complex  
        systems of comparisons can be decomposed into coherent and interpretable structures.

        ---

        ### The challenge: inconsistencies in evaluation data

        In many contexts we collect **evaluations** or **scores**—for instance, ratings,  
        performance measures, or preference judgments—across a set of items. These evaluations  
        may come from very different sources: people giving feedback, experiments producing  
        measurements, or events assigning points.  

        A common difficulty emerges: **conflicting information**.  
        - One source may strongly favor item A over B.  
        - Another may prefer B over C.  
        - Yet another may suggest C over A.  

        When such loops occur, they form what is known as a **Condorcet cycle** in decision  
        theory: a paradox where preferences circle back on themselves, making it impossible  
        to obtain a consistent overall ranking by simple averaging or majority rules.  
        Traditional aggregation methods ignore these contradictions, producing results that  
        may look neat but fail to reveal the underlying disagreement.

        ---

        ### The HodgeRank approach

        HodgeRank addresses this challenge by representing evaluation data as a **graph**:  
        - **Nodes (vertices)** represent the items being evaluated.  
        - **Edges (links)** represent pairwise differences of evaluation between items.  

        Through methods from **linear algebra** and **optimization**, this graph is  
        decomposed into complementary components:

        1. **Global ranking (gradient part):** the most consistent overall ordering of items.  
        2. **Local inconsistencies (curl part):** small contradictory cycles, such as  
        *A > B > C > A*.  
        3. **Global inconsistencies (harmonic part):** larger structural conflicts that  
        cannot be explained by local cycles alone.  

        This decomposition not only produces a ranking but also quantifies **how much of the  
        data is consistent** and **where contradictions are concentrated**.

        ---

        ### Why this matters

        Unlike raw averages or simple totals, HodgeRank provides:  
        - A **global ranking** that respects the mathematical structure of the data.  
        - A **measurement of consistency vs. inconsistency**, distinguishing between  
        local and global conflicts.  
        - **Interpretability**, by explaining not only *which items are ranked highest*  
        but also *how reliable that ranking is*, and *where the disagreements lie*.  

        For this reason, HodgeRank has become increasingly relevant in fields as diverse as  
        **political science**, **psychology**, **linguistics**, **marketing research**, and  
        **sports analytics**. It transforms subjective, fragmented, and sometimes  
        contradictory evaluations into a **coherent structure** that can be rigorously  
        analyzed.

        ---

        ### Examples of HodgeRank in practice

        To better illustrate the idea, let us look at a few scenarios where **HodgeRank**  
        can provide valuable insights:

        1. **Sports competitions**  
        Imagine a round-robin tournament where teams play against each other.  
        - Team A beats Team B,  
        - Team B beats Team C,  
        - but Team C beats Team A.  
        A simple "win count" fails to capture the circular contradiction.  
        With HodgeRank, the results are decomposed into:  
        - a **global ranking** reflecting the general strength of teams, and  
        - a **curl component** revealing the inconsistent cycle among A, B, and C.

        2. **Peer review in academia**  
        Suppose multiple referees evaluate research papers. One reviewer prefers  
        paper X over Y, while another ranks Y over Z, and a third favors Z over X.  
        HodgeRank highlights where these **contradictions** occur and quantifies  
        their impact, allowing editors to separate consistent judgments from  
        structural disagreement in the community.

        3. **Customer product ratings**  
        In marketing research, customers may express preferences across competing  
        products. Traditional averages might misleadingly crown a "winner."  
        HodgeRank, however, uncovers whether the consensus is **robust** or if  
        certain customer groups create contradictory preference loops.

        ---

        ### Takeaway

        These examples show that HodgeRank is not just about producing a ranking:  
        it is about **understanding the structure of agreement and disagreement**  
        within data. By making contradictions explicit, it equips analysts to  
        interpret results with greater **transparency and rigor**.

        """)
    
    if st.button("Next ➡️ Upload file"):
        st.session_state["step"] = 2
        st.rerun()

# =========================================================================================================================
# STEP 2 – Upload + Preview + Summary
# =========================================================================================================================
elif step == 2:

    st.markdown("""
        # 2. Upload + Preview + Summary

        ### Structure of the dataset

        To apply the HodgeRank method, the dataset must follow a specific structure:  

        - **First row (header):** contains the **items being evaluated**.  
        These can be words, products, individuals, or any entities of interest.  
        - **First column (index):** contains the **identifiers of the contexts or evaluators**.  
        These may represent participants, environments, events, or any source that provides evaluations.  
        - **Remaining cells (matrix body):** contain **numeric values** that represent the evaluation of each item within each context.  
        - **Negative values** indicate unfavorable evaluations or low scores.  
        - **0** indicates a neutral evaluation or absence of preference.  
        - **Positive values** indicate favorable evaluations or high scores.  
        - Values can be any number (integer or float), not necessarily restricted to a specific scale.  

        This tabular structure ensures that evaluations are consistently represented as a matrix, which is the natural input for the HodgeRank decomposition.

        ---

        ### Example of Dataset Structure

        | Context / Evaluator | Item 1 | Item 2 | Item 3 | Item 4 |
        |--------------------|--------|--------|--------|--------|
        | Evaluator A        | 2.5    | -1     | 0      | 3      |
        | Evaluator B        | -2     | 0      | 1.5    | -0.5   |
        | Evaluator C        | 0      | 2      | -3     | 0      |

        ---

        ### Validity checks

        When a dataset is uploaded, the application automatically validates its structure to guarantee mathematical consistency. The following checks are performed:

        1. **Minimum size:** the table must contain at least two rows and two columns.  
        2. **Item labels (first row):** no empty cells are allowed; duplicate item labels trigger a warning.  
        3. **Context identifiers (first column):** no empty entries are allowed.  
        4. **Numeric values:** every cell in the body must be numeric. Non-numeric entries are invalid.  

        If any of these conditions fail, the dataset is rejected with an error message. If minor issues are detected (such as duplicated item names), the dataset is accepted but a warning is displayed.

        ---

        ### Automatic cleaning

        In addition, the system applies a cleaning step:  
        - Any **context (row)** with all zero values is removed.  
        - Any **item (column)** with all zero values is removed.  

        This prevents the analysis from being distorted by empty or irrelevant entries.

        ---

        **In summary:**  
        Your dataset should resemble a **matrix of evaluations**, with items as columns, contexts as rows, and numeric values (negative, zero, or positive) as the entries. Once the file passes validation, the application will provide a preview and summary of the data, ready for the HodgeRank procedure.

        """)

    col1, _, _ = st.columns([3.5, 1, 1])

    with col1:
        
        uploaded_file = st.file_uploader("Upload a .csv or .xlsx file", type=["csv", "xlsx"])

        def validate_dataset_structure(df: pd.DataFrame):
            """
            Validates whether the uploaded dataset follows the required structure
            for the HodgeRank analysis.

            Expected format:
            - First row (excluding the first cell): names of the items being evaluated.
            - First column (excluding the first cell): identifiers of the evaluation contexts.
            - Remaining cells: numeric, non-negative evaluations.
            """

            errors, warnings = [], []
            rows, cols = df.shape

            # 1. Basic size requirement
            if rows < 2 or cols < 2:
                errors.append("The dataset must contain at least two rows and two columns.")
                return False, errors, warnings

            # 2. Item labels (first row, excluding first cell)
            header_items = df.iloc[0, 1:]
            if header_items.isna().any() or (header_items.astype(str).str.strip() == "").any():
                errors.append("The first row (items) contains empty or missing labels.")

            duplicated = header_items[header_items.duplicated(keep=False)]
            if len(duplicated) > 0:
                errors.append(f"Duplicated item labels detected: {sorted(set(map(str, duplicated)))}.")

            # 3. Row identifiers (first column, excluding first cell)
            row_ids = df.iloc[1:, 0]
            if row_ids.isna().any() or (row_ids.astype(str).str.strip() == "").any():
                errors.append("The first column (row identifiers) contains empty or missing labels.")

            # 4. Matrix body (all other cells)
            body = df.iloc[1:, 1:]
            body_coerced = pd.to_numeric(body.stack(), errors="coerce")

            if body_coerced.isna().any():
                errors.append("Some cells in the evaluation matrix are not numeric.")

            # Final verdict
            return len(errors) == 0, errors, warnings

        if uploaded_file is not None:
            st.session_state["saved_file"] = uploaded_file
            if uploaded_file.name.endswith(".csv"):
                raw_data = pd.read_csv(uploaded_file, header=None)
            else:
                raw_data = pd.read_excel(uploaded_file, header=None)
            st.session_state["loaded_data"] = raw_data


    if "loaded_data" in st.session_state:

        raw_data = st.session_state["loaded_data"]
        ok, errs, warns = validate_dataset_structure(raw_data)

        if not ok:
            st.error("❌ Invalid structure:")
            for e in errs: st.write(f"- {e}")
            st.stop()
        
        if warns:
            for w in warns: st.warning(f"⚠️ {w}")

        st.success("✅ File uploaded successfully.")

        data = hc.preprocess_data(raw_data)
        binary_dataframe = (data.copy() != 0).astype(np.int8)

        zero_voters = [v for v in binary_dataframe.columns if binary_dataframe.loc[:, v].sum()==0]
        zero_items = [i for i in binary_dataframe.index if binary_dataframe.loc[i].sum() == 0]

        if zero_voters:
            data = data.drop(columns=zero_voters)
            st.warning(f"Removed {len(zero_voters)} voter(s): {zero_voters}")

        if zero_items:
            data = data.drop(index=zero_items)
            st.warning(f"Removed {len(zero_items)} item(s): {zero_items}")

        st.subheader("Loaded Data Preview")
        st.dataframe(data)
        st.subheader("Dataset Summary")
        st.markdown(f"- **Number of items:** {data.shape[0]}  \n- **Number of voters:** {data.shape[1]}")
        st.session_state["actual_data"] = data

        left, right = st.columns([0.25, 1])
        if left.button("⬅️ Back"):
            st.session_state["step"] = 1; st.rerun()
        if right.button("Next ➡️ Configurations"):
            st.session_state["step"] = 3; st.rerun()

    else:

        left, right = st.columns([0.25, 1])
        if left.button("⬅️ Back"):
            st.session_state["step"] = 1; st.rerun()

# =========================================================================================================================
# STEP 3 – Configurations
# =========================================================================================================================
elif step == 3:
    st.title("3. Configurations")
    left , right = st.columns([1, 1])
    with left: 
        st.markdown("""
        ### Parameter: Remove Least-Frequently Evaluated Items

        Before running the HodgeRank analysis, you can adjust certain parameters that influence the results.  

        - **Remove least frequently evaluated items (%):**  
        In some datasets, certain items may receive very few evaluations from the available contexts (e.g., participants, events, environments, or other sources of assessment).  
        Keeping these rarely evaluated items can introduce noise or distort the global ranking.  
        This option allows you to remove the X% of items with the fewest evaluations, focusing the analysis on items that have been evaluated more consistently across contexts.  
        This helps ensure that the resulting ranking is more robust and meaningful.
        """)

        st.warning("""
        ⚠️ **Important note about evaluation frequency:**  
        Some items might have received very few scores. When this occurs, they can appear artificially high or low in the ranking simply due to chance.  

        For example, if an item was evaluated only once and received an extreme score (high or low), there is insufficient information for the algorithm to place it reliably in the global ranking.  

        To reduce this effect, it is recommended to apply a frequency cut of at least 10%, removing the least-evaluated items before computing the HodgeRank potentials.  
        This ensures that the ranking reflects consistent evaluations rather than outliers or rare observations.
        """)

        
        data = st.session_state.get("actual_data")
        cut_percentage = st.slider(r"Remove the X% least frequent items:", 0.0, 80.0, st.session_state.get("saved_percentage", 0.0), 0.5)

        data_filtered, feedback = hc.apply_cut(data, cut_percentage)

        nav1, nav2 = st.columns([0.25, 1])
        if nav1.button("⬅️ Back"):
            st.session_state["step"] = 2
            st.session_state["saved_percentage"] = cut_percentage
            st.rerun()

        if nav2.button("Next ➡️ Hodge Rank Procedure"):
            st.session_state["step"] = 4
            st.session_state["saved_percentage"] = cut_percentage
            st.session_state["data_filtered"] = data_filtered
            st.rerun()
    
    with right:
        st.header(f"Feedback on Removed Items - {cut_percentage}% removed")

        if feedback["num_items_removed"] == 0:
            st.subheader("✅ No items were removed")
            st.markdown(f"The dataset contains {feedback['num_items_original']} items. All items were sufficiently evaluated, so the data remains unchanged.")
        else:
            st.subheader(f"⚠️ {feedback['num_items_removed']} items removed")
            st.markdown(
                f"The dataset originally had {feedback['num_items_original']} items. "
                f"After applying the frequency cut, {feedback['num_items_remaing']} items remain. "
                "The removed items and their frequencies are shown in the table below. "
            )
            st.markdown("### Removed Items")
            st.table(feedback["dropped"])

# =========================================================================================================================
# STEP 4 – Hodge Rank Procedure
# =========================================================================================================================

elif step == 4:
    st.title("4. HodgeRank Procedure")
    st.markdown("""
        The HodgeRank procedure is now running.  
        This step computes the global ranking (potentials) of items based on the evaluations provided.

         Please note: computation time may vary depending the size of the dataset.
    """)
    st.info("Running HodgeRank...")


    data = st.session_state.get("data_filtered")
    st.session_state["results"] = hc.compute_hodgerank(data)

    st.success("Hodge Rank finished!")
    col2c, col1c = st.columns(2)

    if col1c.button("➡️ Go to Results"):
        st.session_state["step"] = 5; st.rerun()
    if col2c.button("⬅️ Back"):
        st.session_state["step"] = 3; st.rerun()

# =========================================================================================================================
# STEP 5 – Ranking Table
# =========================================================================================================================

if step == 5:
    st.title("HodgeRank Results")
    st.title("1. Ranking Table")

    col1, col2 = st.columns([1.5,1.5])

    with col2:
        results = st.session_state["results"]
        ranking_table = results["rankingtable"]

        st.markdown("### Ranking Table obtained")
        st.table(ranking_table.sort_values(by= "Potential Score", ascending = False))

    with col1: 
        st.markdown("""
            ### How to Interpret the Ranking Table

            The ranking table provides a comprehensive summary of the items based on the HodgeRank analysis. Each column has a specific meaning:

            **Rank**  
            The overall position of the item in the ranking. Items with **higher potential** are ranked better.  

            **Potential Score**  
            A numeric score that summarizes the global standing of each item, derived from all pairwise comparisons.  
            - **Higher potentials → better rank**.  
            - Large gaps between potentials indicate clear separation between items.  
            - Values near each other suggest close competition or ties.  
            - Negative potentials reflect unfavorable overall evaluations relative to other items.  

            **Evaluation Frequency (%)**  
            Shows how often an item was evaluated across all contexts (participants, events, or other sources).  
            - **Higher frequency** → more reliable potential.  
            - **Low frequency** → potential may be less stable; interpret with caution.  

            **Number of Votes**  
            The absolute number of evaluations received by the item.  
            - More votes generally increase confidence in the item's potential.  

            **Total Score**  
            Sum of all scores received by the item across contexts.  
            - Provides a raw measure of total support or evaluation.  

            **Average Score**  
            Average Score given to the item.  
            - Less sensitive to extreme scores than the total or mean, giving a robust central tendency.  

            **In summary:**  
            The table not only tells you **who ranks best (Potential)** but also **how robust that ranking is (Frequency and Quantity of Votes)**.  
            By combining these indicators, you can understand both the ranking and its reliability.
            """)

        left, right = st.columns(2)
        with left:
            download_csv(ranking_table, "RankingTable.csv")
        with right:
            download_excel(ranking_table, "RankingTable.xlsx")

        col1c, col2c = st.columns(2)
        if col1c.button("⬅️ Back"): st.session_state["step"] = 3; st.rerun()
        if col2c.button("➡️ Proceed"): st.session_state["step"] = 6; st.rerun()

# =========================================================================================================================
# STEP 6 – Consistency & Inconsistency
# =========================================================================================================================

if step == 6:
    st.title("HodgeRank Results")
    st.title("2. Consistency & Inconsistency Analysis")
    st.markdown("""
        When we try to build a single ranking from pairwise comparisons, some contradictions are inevitable.  
        For example, you may have **A > B**, **B > C**, but also **C > A**.  

        HodgeRank measures how much of the data can be explained by a consistent ranking (the **gradient part**) versus how much remains as contradictions:  

        - **Consistency (Gradient):** the portion of the comparisons that fit into one global ranking. Higher values mean the ranking is a good summary of the data.  
        - **Curl (Local Inconsistencies):** contradictions that happen in small cycles of 3 items (e.g., A > B > C > A).  
        - **Harmonic (Global Inconsistencies):** contradictions that occur at a larger scale in the comparison network. These are usually small unless the data is very fragmented.  

        In short:  
        - **High consistency** (close to 100%) → the ranking is stable and explains most of the preferences.  
        - **High inconsistency** → the dataset has many conflicts; be cautious when interpreting the top positions.
        """)
    
    st.subheader("Decomposition summary")
    
    results = st.session_state["results"]

    flow_norm = results["flownorm"]
    grad_norm = results["gradientnorm"]
    curl_norm = results["curlnorm"]
    harm_norm = results["harmonicnorm"]

    grad_share = (grad_norm/flow_norm) if flow_norm > 1e-7 else 0.0
    curl_share = (curl_norm/flow_norm) if flow_norm > 1e-7 else 0.0
    harm_share = (harm_norm/flow_norm) if flow_norm > 1e-7 else 0.0

    left, right = st.columns([1.5, 1.5])

    with left: 
        col1, col2, col3 = st.columns(3)

        col1.metric("Consistency (Gradient)", f"{100*grad_share:.6f}%")
        col2.metric("Local inconsistencies (Curl)", f"{100*curl_share:.6f}%")
        col3.metric("Global inconsistencies (Harmonic)", f"{100*harm_share:.6f}%")

        if "fig_pie" not in st.session_state:
            st.session_state["fig_pie"] = hc.plot_decomposition_pie(grad_share, curl_share, harm_share)
            if st.session_state["fig_pie"] is not None:
                st.session_state["img_pie"] = hc.fig_to_png(st.session_state["fig_pie"])
        
        if st.session_state["fig_pie"] is not None:
            img = st.session_state["img_pie"]
            fig = st.session_state["fig_pie"]
            st.image(img)

        else: 
            st.warning("The pie chart was not generated because the values are null.")
        s1, s2 = st.columns(2)
        with s1:
            if st.session_state["fig_pie"] is not None:
                exp1, exp2 = st.columns(2)
                with exp1:
                    download_pdf(fig, "DecompositionChart.pdf")
                with exp2:
                    download_png(fig, "DecompositionChart.png")

        with s2:
            col1c, col2c = st.columns([0.5,1])
            if col1c.button("⬅️ Back"): st.session_state["step"] = 5; st.rerun()
            if col2c.button("➡️ Proceed"): st.session_state["step"] = 7; st.rerun()

# =========================================================================================================================
# STEP 7 – Scatter Potentials x Frequency
# =========================================================================================================================

if step == 7:
    st.title("HodgeRank Results")
    st.title("3. Scatter Plot: Potentials x Frequency")
    st.markdown("""
        ### How to interpret this scatter plot

        This plot shows the relationship between **Potentials** (x-axis) and **Frequencies** (y-axis) of the items analyzed with the HodgeRank method.

        - **Potentials (x-axis):** represent the global ranking position of each item. Items farther to the right are more favored, those to the left less favored.  
        - **Frequencies (y-axis):** indicate how often each item was evaluated by participants (in %). Higher values mean more supporting evidence and therefore more reliable positions.

        **Reading the plot:**  
        - **Upper-right:** strong leaders (high potential and robust frequency).  
        - **Lower-right:** potential leaders, but with low frequency → unstable.  
        - **Upper-left:** less favored but reliable (many evaluations).  
        - **Lower-left:** weak and rarely evaluated.  
        """)
    
    col1, col2 = st.columns([1.5, 1])
    results = st.session_state["results"]
    ranking_table = results["rankingtable"]

    potentials = []
    frequencies= []
    items = ranking_table.index

    for item in items:
        potentials.append(ranking_table.loc[item, "Potential Score"])
        frequencies.append(ranking_table.loc[item, "Evaluation Frequency (%)"])

    rebuild = st.session_state.get("rebuild", False)

    with col1:
        if "scatter" not in st.session_state or rebuild:
            st.session_state["scatter"] = hc.plot_scatter_potentials_freq(items, potentials, frequencies, st.session_state.get("size", (6,4)), st.session_state.get("letters", 5), st.session_state.get("fontsize", 7))
            st.session_state["imgscatter"] = hc.fig_to_png(st.session_state["scatter"])

        st.session_state["rebuild"] = False
        fig = st.session_state["scatter"]
        img = st.session_state["imgscatter"]
        st.image(img)

        s1, s2 = st.columns(2)
        with s1:
            exp1, exp2 = st.columns([0.5,1])
            with exp1:
                download_pdf(fig, "ScatterPlot.pdf")
            with exp2:
                download_png(fig, "ScatterPlot.png")
        with s2:
            col2c, col1c = st.columns([0.6, 1])
            if col2c.button("⬅️ Back"): st.session_state["step"] = 6; st.rerun()
            if col1c.button("➡️ Proceed"): st.session_state["step"] = 8; st.rerun()
    
    with col2:
        st.markdown("### Plot customization")
        st.markdown(
            "Here you can configure some aspects of the plot, such as:\n"
            "- **Figure size (m × n)**\n"
            "- **Maximum number of characters to display in labels** (to avoid very long labels)\n"
            "- **Font size of labels**")
        
        st.markdown("### Plot customization")

        with st.expander("⚙️ Customize scatter plot", expanded=False):
            # Figure size
            col1, col2 = st.columns(2)
            with col1:
                width = st.number_input("Figure width", min_value=3, max_value=20, value=st.session_state.get("width", 7), step=1)
            with col2:
                height = st.number_input("Figure height", min_value=3, max_value=20, value=st.session_state.get("height", 4), step=1)
            
            col1a, col2a = st.columns(2)
                # Label max length
            with col1a:
                max_letters = st.number_input( "Max characters in labels", min_value=3, max_value=20, value= st.session_state.get("letters", 5), step=1, help="Longer names will be truncated with '...'")
            with col2a:
                # Font size
                fontsize = st.number_input("Font size of labels", min_value=5, max_value=20, value= st.session_state.get("fontsize", 7), step=1)
            
            if st.button("🔄 Rebuild scatter plot"):
                st.session_state["width"] = width
                st.session_state["height"] = height
                st.session_state["size"] = (width, height)
                st.session_state["letters"] = max_letters
                st.session_state["fontsize"] = fontsize
                st.session_state["rebuild"] = True
                st.rerun()

# =========================================================================================================================
# STEP 8 – Graph Visualizations
# =========================================================================================================================
if step == 8:
    st.title("HodgeRank Results")
    st.title("4. Graph Visualizations")

    st.markdown("""
        ### How to Interpret the Graphs

        The graphs provide a visual summary of comparisons and the consistency of the ranking.

        **Weighted Edges Graph**  
        Each edge represents a pairwise comparison. Thicker and darker edges indicate stronger evidence (more evaluations), while thinner and lighter edges indicate weaker support. This graph helps you see the overall structure of the dataset and which comparisons are most reliable.

        **Residuals Graph (R*)**  
        This graph highlights disagreements between observed comparisons and the global ranking. Edge color shows the direction of the disagreement: **red** for negative residuals (observed value lower than expected) and **green** for positive residuals (observed value higher than expected). Edge thickness reflects the magnitude of the residual, so thicker edges indicate stronger disagreements.  
        Use this graph to identify controversial relationships or items with uncertain positions in the ranking.
        
        ---
        """)
    
    results= st.session_state.get("results")
    items = results["nodes"]
    edges = results["edges"]
    residuals = results["residuals"]
    edgesweight = results["edgeweightvector"]
    ranking_table = results["rankingtable"]
    potentials = ranking_table.loc[:, "Potential Score"]
    frequencies = ranking_table.loc[:, "Evaluation Frequency (%)"]

    if "graphs" not in st.session_state or st.session_state["rebuild"]:
        
        fig1, fig2 = hc.plot_graph(items, edges, edgesweight, residuals,
                                   layout1 = st.session_state.get("Layout1", "kamada"), layout2 = st.session_state.get("Layout2", "kamada"),
                                   highlight = st.session_state.get("Highlight", "default"),
                                   potentials=st.session_state.get("scaled_potential", potentials), freqs= st.session_state.get("scaled_frequencies", frequencies),
                                   labels1= (st.session_state.get("dx1", 0.0), st.session_state.get("dy1", 0.05)),
                                   labels2= (st.session_state.get("dx2", 0.0), st.session_state.get("dy2", 0.05)))
        
        img1 = hc.fig_to_png(fig1)
        img2 = hc.fig_to_png(fig2)
        st.session_state["graphs"] = fig1, fig2
        st.session_state["graph_imgs"] = img1, img2

        st.session_state["rebuild"] = False
    
    img1, img2 = st.session_state["graph_imgs"]
    fig1, fig2 = st.session_state["graphs"]

    with st.expander("⚙️ Customize plot", expanded=False):

        st.markdown("""
            ### Graph Layout Options

            You can customize how the nodes are positioned in the graphs:

            - **Kamada-Kawai Layout**  
            A force-directed algorithm that spreads nodes in 2D space to minimize edge crossing.  
            Useful for visualizing the *overall structure* of the comparison network.

            - **Potentials × Frequencies Layout**  
            Places each node according to its **Potential** (x-axis) and **Frequency (%)** (y-axis),  
            exactly as in the scatter plot.  
            This makes it easier to compare the graph with the global ranking and reliability of items.
            ### Adjusting Labels Position

            By default, labels are placed slightly above each node.  
            However, in some cases they may overlap with the node itself or with edges,  
            making them hard to read.

            You can manually **shift the labels** along the x and y axes:

            - **Label shift (x-axis):** moves labels left or right.  
            - **Label shift (y-axis):** moves labels up or down.  

            This adjustment improves readability without altering the graph structure.
                    
            ---

            ### Edge Highlight Options

            In the **Residuals Graph**, you can choose how residuals are emphasized:

            - **Default**: green for positive residuals, red for negative ones.  
            - **Highlight only negative residuals**: shows only the edges where items performed worse than expected.  
            - **Highlight only positive residuals**: shows only the edges where items performed better than expected.""")
        
        weight_graph, residual_graph = st.columns(2)
        st.session_state["reseted"] = False

        with weight_graph:
            with st.expander("**Weighted Edges Graph**", expanded=False):
                part_option1 = st.radio("Select the layout of the **Weighted Edges Graph**:",
                                options=["Kamada Kawai Layout", "Potentials x Frequencies"])
                
                dx1 = st.slider("Label shift (x-axis) **(Weighted Edges Graph)**", min_value=-5.5, max_value=5.5, step=0.01, value= 0.0)
                dy1 = st.slider("Label shift (y-axis) **(Weighted Edges Graph)**", min_value=-5.5, max_value=5.5, step=0.01, value= 0.05)
                
                st.session_state["dx1"] = dx1  
                st.session_state["dy1"] = dy1
    
        with residual_graph:
            with st.expander("**Residual Graph**", expanded=False):
                part_option2 = st.radio("Select the layout of the  **Residuals Graph**:",
                                    options=["Kamada Kawai Layout", "Potentials x Frequencies"])
            
                part_option3 = st.radio("Select how you want to highlight the edges",
                            options=["Default",
                            "Highlight only negative residuals",
                            "Highlight only positive residuals"])

                dx2 = st.slider("Label shift (x-axis) **(Residuals Graph)**", min_value=-5.5, max_value=5.5, step=0.01, value= 0.0)
                dy2 = st.slider("Label shift (y-axis) **(Residuals Graph)**", min_value=-5.5, max_value=5.5, step=0.01, value= 0.05)

                st.session_state["dx2"] = dx2
                st.session_state["dy2"] = dy2

        if part_option1 == "Potentials x Frequencies" or part_option2 == "Potentials x Frequencies":
            st.markdown("""
                --- 
                        
                ### Scaling Axes in the Graph Layout

                When you choose the **Potentials × Frequencies layout**,  
                each node is placed at coordinates:

                - **X-axis:** its Potential (ranking score)  
                - **Y-axis:** its Frequency (%) of evaluations  

                Sometimes these values are too close together, making nodes overlap.  
                To improve visibility, you can apply a **scaling factor** to each axis:

                - **X-axis scale (Potentials):** expands or compresses the ranking dimension.  
                - **Y-axis scale (Frequencies):** expands or compresses the reliability dimension.  

                ⚠️ Note: Scaling affects only the **visualization of node positions**.  
                It does **not** change the underlying ranking or frequencies.  
                """)
            
            scale1, scale2 = st.columns(2)
            with scale1:
                scale_factor1 = st.slider(
                    "Scale Potentials (multiply by a constant):",
                    min_value=0.1, max_value=100.0, step=0.1, value= st.session_state.get("reset", st.session_state.get("another_reset", 1.0)),
                    help="Multiply the potentials by a constant to stretch/compress the x-axis. "
                        "This can improve visualization if points are too close.")
            with scale2:
                scale_factor2 = st.slider(
                    "Scale frequencies (multiply by a constant):",
                    min_value=0.1, max_value=100.0, step=0.1, value= st.session_state.get("reset", st.session_state.get("another_reset", 1.0)),
                    help="Multiply the frequencies by a constant to stretch/compress the y-axis. "
                        "This can improve visualization if points are too close.")

            if "reset" not in st.session_state:
                st.session_state["reset"] = 2.0
                st.session_state["another_reset"] = 3.0 - st.session_state["reset"]

            if st.session_state["reseted"] == True:
                st.session_state["reseted"] = False
                st.rerun()

            if st.button("Reset scaling"):
                st.session_state["reset"] = st.session_state["another_reset"]
                st.session_state["another_reset"] = 3.0 - st.session_state["reset"]
                st.session_state["reseted"] = True
                st.rerun()

            st.session_state["scaled_potential"] = [p * scale_factor1 for p in potentials]
            st.session_state["scaled_frequencies"] = [f * scale_factor2 for f in frequencies]

        layout_map = {
            "Kamada Kawai Layout": "kamada",
            "Potentials x Frequencies": "default"}
        
        highlight_map = {
            "Default": "default",
            "Highlight only negative residuals": "negative",
            "Highlight only positive residuals": "positive",
            "Highlight most residuals (absolute value)": "abs"}
        
        st.session_state["Layout1"] = layout_map[part_option1]
        st.session_state["Layout2"] = layout_map[part_option2]
        st.session_state["Highlight"] = highlight_map[part_option3]
        
        if st.button("🔄 Rebuild graphs plots"):
            st.session_state["rebuild"] = True
            st.session_state["type_plotw"] = st.session_state["Layout1"]
            st.session_state["type_plotr"] = st.session_state["Layout2"]
            st.rerun()
    
    type_plot = {
        "kamada": "kamada_layout",
        "default": "scatter_layout"}
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("""
                --- 
                ### Weighted Edges Graph
                    """)
        st.image(img1)

        st.markdown("""
                --- 
                ### Residuals Graph
                    """)
        st.image(img2)

        s1, s2 = st.columns(2)
        with s1:

            exp1, exp2 = st.columns([0.5,1])

            name = "WeightedGraph_" + type_plot[st.session_state["type_plotw"]]
            name2 = "ResidualGraph_" + type_plot[st.session_state["type_plotr"]]
            with st.expander("Downloads in PDF"):
                download_pdf(fig1, name +".pdf")
                download_pdf(fig2, name2 + ".pdf")
            
            with st.expander("Downloads in PNG"):
                download_png(fig1, name + ".png")
                download_png(fig2, name2 + ".png")

        with s2:

            col2c, col1c = st.columns([0.6, 1])
            if col2c.button("⬅️ Back"): st.session_state["step"] = 7; st.rerun()
            if col1c.button("➡️ Proceed"): st.session_state["step"] = 9; st.rerun()

# =========================================================================================================================
# STEP 9 – Matrix Visualizations
# =========================================================================================================================

if step == 9:
    st.title("HodgeRank Results")
    st.title("5. Matrix Visualizations")
    left, right = st.columns(2)
    with left:

        results = st.session_state["results"]
        ranking_table = results["rankingtable"].sort_values(by= "Potential Score", ascending= False)

        ordering = ranking_table.index
        flow_matrix_df = results["flowmatrix"].loc[ordering, ordering]
        potential_matrix_df = results["potentialmatrix"].loc[ordering, ordering]
        residual_matrix = results["residuals"].loc[ordering, ordering]

        if "fig_matrices" not in st.session_state or st.session_state.get("rebuild", False):
            fig1 = hc.plot_matrix(flow_matrix_df, "Flow Matrix (Y)", "Flow Value", part= st.session_state.get("part", "lower"))
            fig2 = hc.plot_matrix(potential_matrix_df, "Potential Component (Yg)", "Potential Value",part= st.session_state.get("part", "lower"))
            fig3 = hc.plot_matrix(residual_matrix, "Residuals (R*)", "Residual Value", part= st.session_state.get("part", "lower"))
            st.session_state["fig_matrices"] = fig1, fig2, fig3

            img1 = hc.fig_to_png(fig1)
            img2 = hc.fig_to_png(fig2)
            img3 = hc.fig_to_png(fig3)

            st.session_state["img_matrices"] = img1, img2, img3
            st.session_state["rebuild"] = False

        fig1, fig2, fig3 = st.session_state["fig_matrices"]
        img1, img2, img3 = st.session_state["img_matrices"]
        st.image(img1); st.image(img2); st.image(img3)

    with right:

        st.markdown("""
        ### How to interpret the matrices

        - **Flow Matrix (Y)**  
          Represents the aggregated pairwise comparisons across all voters.  
          Positive values indicate preference for item *i* over *j*,  
          negative values the opposite.

        - **Potential Component (Yg)**  
          The consistent part of the flows, fully explained by the global ranking.  
          Ideally, this matrix should align with the Flow Matrix if the data is consistent.

        - **Residuals (R*)**  
          The disagreement between observed comparisons and the global ranking.  
          Large values (red/blue) highlight controversial or inconsistent item pairs.
        """)

        with st.expander("⚙️ Customize matrices plots", expanded=False):
            st.markdown("""You can choose which part of the matrix to display in the .  
                        By default, only the **lower triangle** is shown, which avoids redundancy since the matrix is symmetric.  
                        Alternatively, you can plot the **upper triangle** if you prefer to focus on the other half, or display the
                        **entire matrix** to see all values at once. This flexibility
                        allows you to highlight the information most relevant for your analysis while keeping the visualization clear and easy
                        to interpret.""")

            part_option = st.radio("Select which part of the matrix to display:",
                                    options=["Lower Triangle", "Upper Triangle", "Full Matrix"])

            part_map = {
                "Lower Triangle": "lower",
                "Upper Triangle": "upper",
                "Full Matrix": "full"}
            
            st.session_state["part"] = part_map[part_option]

            st.markdown("After choosing an option, click **Rebuild** to generate the updated plot.")

            if st.button("🔄 Rebuild matrices plot"):
                st.session_state["rebuild"] = True
                st.rerun()

        s1, s2 = st.columns(2)
        with s1:

            with st.expander("Export matrices to PDF"):
                download_pdf(fig1, "FlowMatrix.pdf")
                download_pdf(fig2, "PotentialMatrix.pdf")
                download_pdf(fig3, "ResidualsMatrix.pdf")

            with st.expander("Export matrices to PNG"):
                download_png(fig1, "FlowMatrix.png")
                download_png(fig2, "PotentialMatrix.png")
                download_png(fig3, "ResidualsMatrix.png")

        with s2:

            col2c, col1c = st.columns([0.6, 1])
            if col2c.button("⬅️ Back"): st.session_state["step"] = 8; st.rerun()
            if col1c.button("➡️ Proceed"): st.session_state["step"] = 10; st.rerun()

# =========================================================================================================================
# STEP 10 – Final Summary
# =========================================================================================================================

if step == 10:
    st.title("HodgeRank Results")
    st.title("5. Final Summary of the Analysis")

    if "results" in st.session_state:
        results = st.session_state["results"]
        ranking_table = results["rankingtable"]

        # ==========================================================
        # 1. Global Statistics
        # ==========================================================
        st.subheader("1. Global Statistics")
        st.markdown("""
        *This section summarizes the overall structure of the dataset.*

        - **Items evaluated:** the set of entities being compared or ranked.  
        - **Evaluation contexts:** the situations where items are assessed  
          (e.g., experiments, sessions, judges, or events).  
        - **Valid comparisons:** the number of pairwise relations extracted from the data.  
        - **Consistency:** the proportion of the dataset that can be explained  
          by a coherent global ranking. Higher consistency indicates more reliable orderings.  
        """)

        n_items = ranking_table.shape[0]
        n_contexts = results["numbercontexts"]
        n_comparisons = len(results["edges"])

        grad_share = 100 * results["gradientnorm"] / results["flownorm"]
        curl_share = 100 * results["curlnorm"] / results["flownorm"]
        harm_share = 100 * results["harmonicnorm"] / results["flownorm"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Items evaluated", n_items)
        col2.metric("Evaluation contexts", n_contexts)
        col3.metric("Valid comparisons", n_comparisons)
        col4.metric("Consistency", f"{grad_share:.1f}%")

        # ==========================================================
        # 2. Ranking Highlights
        # ==========================================================
        st.subheader("2. Ranking Highlights")
        st.markdown("""
        *The global ranking assigns a **potential score** to each item.  
        Higher scores indicate stronger overall performance across contexts.*

        - **Top items:** consistently favored across most evaluation contexts.  
        - **Bottom items:** consistently weaker.  
        - **Unstable items:** high potential but relatively low frequency of evaluation.  
          Their position may be less reliable, since fewer observations support it.  
        """)

        top3 = ranking_table.sort_values("Potential Score", ascending=False).head(3)
        bottom3 = ranking_table.sort_values("Potential Score", ascending=True).head(3)

        st.markdown("**Top 3 items:**")
        st.table(top3[["Rank", "Potential Score", "Evaluation Frequency (%)"]])

        st.markdown("**Bottom 3 items:**")
        st.table(bottom3[["Rank","Potential Score", "Evaluation Frequency (%)"]].sort_values(by = "Rank", ascending= True))

        unstable = ranking_table[
            (ranking_table["Potential Score"] > ranking_table["Potential Score"].median()) &
            (ranking_table["Evaluation Frequency (%)"] < 10)].sort_values(by= "Potential Score", ascending= False)
        if not unstable.empty:
            st.warning("""
            ⚠️ Some items appear as **unstable**.  
            This does not mean their ranking is incorrect, but that they have high potential (greater than average)  
            supported by relatively fewer evaluations compared to others.  
            Their exact placement should therefore be interpreted with caution.
            """)
            st.table(unstable[["Rank","Potential Score","Evaluation Frequency (%)"]])

        # ==========================================================
        # 3. Inconsistencies
        # ==========================================================
        st.subheader("3. Inconsistencies in the Data")
        st.markdown("""
        *Not all data is perfectly consistent. Contradictions may occur.*  

        Example: Item A is preferred over B, B over C, but C also over A.  

        These contradictions are captured as:  
        - **Local inconsistencies (3-cycles):** contradictions in small groups of three items.  
        - **Global inconsistencies:** larger cyclic structures across the dataset.  

        Local cycles are common and reflect natural variability.  
        Global inconsistencies are more critical, indicating deep conflicts in the evaluations.  
        """)

        st.markdown(f"""
        - **Local inconsistencies (3-cycles):** {curl_share:.2f}%  
        - **Global inconsistencies:** {harm_share:.4f}%  
        """)

        # ----------------------------------------------------------
        # Most controversial comparisons
        # ----------------------------------------------------------
        st.markdown("### Most Controversial Comparisons")
        st.markdown("""
        *These are the pairwise comparisons where the observed outcomes  
        most strongly disagree with the global ranking.*

        - If the global ranking suggests **Item A > Item B**,  
          but many contexts show the opposite, the residual becomes high.  
        - Such pairs highlight unstable or disputed relations in the dataset.  
        """)

        residuals = results["residuals"].abs()
        inconsistent_edges = []
        for u,v in results["edges"]:
            inconsistent_edges.append({
                "Comparison": f"{u} ↔ {v}",
                "Residual": residuals.loc[u,v]
            })
        edge_df = pd.DataFrame(inconsistent_edges).sort_values("Residual", ascending=False)
        st.table(edge_df.head(5))
        
        left1, right1 = st.columns(2)
        with left1:
            download_csv(edge_df, "InconsistentEdges.csv")

        with right1:
            download_excel(edge_df, "InconsistentEdges.xlsx")


        # ----------------------------------------------------------
        # Most cyclic triangles
        # ----------------------------------------------------------
        st.markdown("### Most Cyclic Triangles")
        st.markdown("""
        *These are groups of three items that form a **Condorcet cycle**:  
        A > B, B > C, and C > A.*  

        - If the value is near **0**, the triad is consistent.  
        - If the **absolute value** is large, it reflects strong local contradiction.  
        - The **sign** of the value only indicates cycle orientation,  
          while the absolute value measures its strength.  

        **Interpretation:**  
        A high value does not mean the global ranking is invalid.  
        It indicates that in some contexts, the expected order was reversed,  
        producing a circular pattern.  

        Such cycles reveal **where local contradictions concentrate**,  
        even if the global ranking remains valid.  
        """)

        if not results["curlvalues"].empty:
            tri_df = results["curlvalues"].copy()
            tri_df["AbsValue"] = tri_df["Value"].abs()
            st.table(tri_df.sort_values("AbsValue", ascending=False).head(5))

            left2, right2 = st.columns(2)
            with left2:
                download_csv(tri_df, "InconsistentTriangles.csv")
            with right2:
                download_excel(tri_df, "InconsistentTriangles.xlsx")
        else:
            st.success("✅ No significant 3-cycles detected.")

        # ==========================================================
        # 4. Automatic Interpretation
        # ==========================================================
        st.subheader("Resume")
        st.markdown("""
        *This section provides a narrative summary of the results:*

        - **High consistency (> 70%)** → the ranking is stable and reliable.  
        - **Moderate local cycles** → small contradictions exist, but the global ranking remains interpretable.  
        - **Significant global inconsistencies** → the dataset contains fundamental conflicts;  
          the ranking should be interpreted with caution.  
        """)

        if grad_share > 70 and harm_share < 1e-3:
            st.success("✅ High overall consistency. The global ranking is reliable, with only minor local cycles.")
        elif harm_share >= 1e-3:
            st.error("⚠️ Significant global inconsistencies detected. Interpret the ranking with caution.")
        else:
            st.warning("⚠️ Moderate local cycles detected. Some comparisons may weaken ranking stability.")


        left, center, right= st.columns([0.6, 0.6, 1])
        if left.button("⬅️ Back"):
             st.session_state["step"] = 9; st.rerun()
        if center.button("New Analysis"):
             st.session_state["step"] = 1
             st.session_state.clear()
             st.rerun()
        if right.button("Finish"):
            st.session_state.clear()
            st.rerun()