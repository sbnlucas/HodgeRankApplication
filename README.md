# HodgeRank Analyzer

**Authors:** Your Name(s)  
**Repository license:** Apache License 2.0  
**Released application:** available directly on [Streamlit Community Cloud](https://hodgerankapplication.streamlit.app/)

---

## Abstract

This repository implements an interactive tool that accepts a table of item evaluations (pairwise/row-wise comparisons or scores) and decomposes the induced **edge flow** on the comparison graph into three orthogonal components:

1. **Gradient (global ranking / potential)** — the part explained by a scalar potential (a global score for each item).  
2. **Curl (local inconsistencies)** — cyclic contradictions concentrated in small cycles (triangles / 3-cycles).  
3. **Harmonic (global inconsistencies)** — structural cycles not reducible to local triangles.

The decomposition is intended for exploratory analysis of ranking / classification problems where the data may include conflicting judgments: the method quantifies how much of the observations are consistent with a single global ranking and where contradictions lie. The core numeric implementation and the Streamlit UI are provided in this repository. :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}

---

## Quick summary (what this software actually does)

- Validates and preprocesses a tabular CSV input (first row: items; first column: evaluator/context identifiers; body: numeric scores), removing empty voters/items. The input validation and upload flow are implemented in `app.py`. :contentReference[oaicite:8]{index=8}  
- Optionally removes the X% least-frequently evaluated items via a slider before analysis. :contentReference[oaicite:9]{index=9}  
- Builds a weighted comparison graph (weight = number of common evaluations between items), computes a per-edge aggregated flow (average signed difference when both items were evaluated by the same context), detects triangles (3-cycles) and assembles incidence / curl matrices. The core numerical decomposition and solver logic are in `hodgerank_core.py`. :contentReference[oaicite:10]{index=10}  
- Solves two least-squares problems (via `scipy.sparse.linalg.lsqr`) to extract the curl component and the gradient (potential) component; the harmonic remainder is obtained as the residual. Norms of each component (used to compute decomposition shares) are returned. :contentReference[oaicite:11]{index=11}  
- Produces these outputs (available in the UI and downloadable):  
  - **Ranking table** with columns: `Rank`, `Potential Score`, `Evaluation Frequency (%)`, `Number of Votes`, `Total Score`, `Average Score`. :contentReference[oaicite:12]{index=12}  
  - **Flow matrix (Y)**, **Potential component matrix (Yg)** and **Residuals (R\*)** (visualized as heatmaps). :contentReference[oaicite:13]{index=13}  
  - **Decomposition summary** (grad / curl / harmonic norms and pie chart). :contentReference[oaicite:14]{index=14}  
  - **Scatter** of Potentials × Frequency and **Graph visualizations**: weighted edge graph and residuals graph (with options to highlight negative/positive residuals or use Potentials×Frequency layout). :contentReference[oaicite:15]{index=15} :contentReference[oaicite:16]{index=16}  
  - CSV / PNG / PDF downloads for tables and figures (provided by `app.py` download helpers). :contentReference[oaicite:17]{index=17}

> Note: the internal numeric components used to compute norms (Yg, Yc, Yh) are computed inside `compute_hodgerank`, but the raw component arrays are not included in the default returned dictionary as named export fields (they are used to build matrices, norms and diagnostics). If you want the raw component vectors/matrices exported, a small change to `hodgerank_core.py` is required. :contentReference[oaicite:18]{index=18}

---

## Input format (required)

The Streamlit app expects a CSV with the following arrangement (no header row detection — `read_csv(..., header=None)` is used by the UI):

- **First row (top-left cell can be empty or a label):** item names start at column 2 (i.e., columns 1..M are item labels).  
- **First column (from row 2 downward):** identifiers of evaluation contexts / voters (strings).  
- **Body (rows 2.., columns 2..):** numeric evaluations (float or int). Values may be negative, zero or positive; zero is treated as "no evaluation" for the counting of co-occurrences. The app validates that there are at least two rows and two columns and that the body is numeric. See the input validation in `app.py`. :contentReference[oaicite:19]{index=19}

**Minimal example (CSV)**

| | Item A | Item B | Item C |
|---: | :---: | :---: | :--- |
| voter1 |  2.0 | -1.0 | 0.0 |
| voter2 | -0.5 |  0.0 | 1.2 |
| voter3 |  1.0 | 1.0 | -0.5 |

After upload the app transposes and converts this into an items × voters matrix used by the HodgeRank backend.

## Usage 
This application is available directly on **Streamlit Cloud**:  
[https://hodgerankapplication.streamlit.app/](https://hodgerankapplication.streamlit.app/)

If you prefer to run it locally:

1. Install Python (3.8+ recommended)  
2. Install the dependencies listed in `requirements.txt`  
3. From the terminal, run:

```bash
streamlit run app.py
```
## Outputs

When you upload a properly formatted CSV of evaluations, the application returns:

- **Ranking table** with each item's:
  - Rank position
  - Potential score (from gradient component)
  - Evaluation frequency (%)
  - Number of votes
  - Total score
  - Average score

- **Decomposition summary**:
  - Norm shares of Gradient, Curl, and Harmonic components
  - Pie chart representation

- **Visualizations**:
  - Scatter plot of Potentials × Frequency
  - Weighted edges graph of the comparison network
  - Residuals graph highlighting inconsistencies
  - Heatmaps for Flow matrix, Potential component, and Residuals

- **Diagnostics**:
  - Most controversial comparisons (largest residuals)
  - Most cyclic triangles (curl values)

All results can be exported as CSV, PNG, or PDF for further analysis.

---

## License and Notice

This project is released under the **Apache License 2.0**.  
Redistributions must preserve copyright and license notices.  

In addition to the `LICENSE` file, this repository also includes a `NOTICE` file.  
The `NOTICE` provides attribution and additional legal information required by the Apache License.  

- License: [Apache License 2.0](./LICENSE)  
- Notice: [NOTICE](./NOTICE)
  
Although the license does not *legally* require an academic citation, this software was developed for research purposes, and proper attribution is essential to support its continued development.  
If you use this tool in academic work, presentations, or derived projects, please acknowledge it by citing the repository and the Streamlit application. Doing so helps others find the tool, validates the research effort, and strengthens reproducibility in the community.  

**Suggested BibTeX:**

```bibtex
@software{hodgerank_analyzer,
  author       = {Santos, L. and M. Calçada and Lunardi, J.T.},
  title        = {HodgeRank Analyzer},
  year         = {2025},
  publisher    = {GitHub / Streamlit Community Cloud},
  url          = {https://github.com/sbnlucas/HodgeRankApplication},
  note         = {Official application hosted at \url{https://hodgerankapplication.streamlit.app/}. Released under the Apache License 2.0.}
}


