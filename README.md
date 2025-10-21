# Using RNN to discover individual differences in human decision-making 
Extended Joshua Peterson’s risky choice research by addressing limitations in capturing individual differences and temporal behavioral effects, replacing the original neural network with recurrent architectures to capture temporal dependencies to advance decision-making theory. A research codebase that extends Peterson et al.’s risky-choice work by replacing static neural utility with **state-conditioned, recurrent utility**. We test whether **sequential experience** (prior selections, outcomes, forgone payoffs) **reshapes utility** over time and whether the hidden state (h_t) encodes latent preferences (risk sensitivity, switching tendency, etc.).

---

## Data

* `c13k_obj_feats_uid.csv` — **problem-level** features by `uniqueID`: outcomes/probabilities for A & B (`Ha, La, pHa, Hb, Lb, pHb`) plus derived stats (EV, variance, skewness).
* `c13k_subject_data_uid.csv` — **subject-level** sequences: `uniqueID, Problem, Feedback, subjectID, aLeft, gambleOrder, sel1–sel5, reward1–reward5, forgone1–forgone5`, summaries (`bRate, avgReward, avgForgone, proportionRight`).

> Each row = a subject’s up-to-5 repeats of the same problem.

---

## What we model

* **Baseline (Neural-EU):** free-form utility (u(x)) (MLP) with linear probability weighting; predicts (V(A), V(B)) and choice via softmax temperature (\eta).
* **RNN Model:** GRU/LSTM summarizing recent history into (h_t); (h_t) **conditions** (u(x\mid h_t)) (FiLM/hypernet) before computing (V(A),V(B)).

**Choice rule:** (P(A)=\dfrac{e^{\eta V(A)}}{e^{\eta V(A)}+e^{\eta V(B)}}).

---

## Repository structure

```
.
├── data_preprocessing.py   # load, merge, long-format seqs, masks, splits
├── neural_eu.py            # baseline utility network u(x)
├── first_layer_rnn.py      # GRU/LSTM core: h_t = RNN(x_dyn, h_{t-1})
├── model.py                # full models: baseline & RNN (shared API)
├── train.py                # training loop, metrics, checkpoints
├── validation.py           # loss-vs-train-size curves, plots
├── interprete.py           # h_t probes (PCA/UMAP), u(x) visualizations
├── analysis.py             # figure assembly + summary tables
├── main_model.py           # entrypoints: baseline / initial_rnn
└── README.md
```

---

## Quick start

1. **Prepare data**

```bash
python data_preprocessing.py \
  --obj_csv /path/to/c13k_obj_feats_uid.csv \
  --subj_csv /path/to/c13k_subject_data_uid.csv \
  --split_by subject   # or: problem
  --out_dir ./artifacts/datasets
```

2. **Baseline (Neural-EU)**

```bash
python main_model.py baseline \
  --data ./artifacts/datasets \
  --eta_mode learn_global \
  --epochs 50 --lr 1e-3 --batch_size 256
```

3. **RNN (state-conditioned utility)**

```bash
python main_model.py initial_rnn \
  --data ./artifacts/datasets \
  --rnn_type gru --hidden_dim 64 \
  --epochs 50 --lr 1e-3 --batch_size 256
```

> HPC: wrap the above in your `main.py` launcher or job scripts (`sbatch`), one target per run.

---

## Experiments

* **Train-size sweep:** for a fixed train/test split, train on {1,3,10,20,50,80,100}% of the train set; evaluate on **100% test**.
* **Two split modes:** `by=subject` (generalize to new people) and `by=problem` (new gambles).

---

## Evaluation

* **Primary metric:** Binary cross-entropy (BCE) on **trial-level** choices.
* **Secondary (optional):** problem-level MSE of predicted P(B) vs empirical proportion (to compare with prior aggregate analyses).
* Plots: **Loss vs % of train data** for baseline vs RNN (same axes, CIs over resamples).

---

## Interpretation

* Save (h_t) and visualize with **PCA/UMAP**, color by: last outcome (win/loss), switch/stay, feedback, trial index.
* Evaluate (u(x)) on a grid to compare **baseline** vs **state-conditioned** (u(x\mid h_t)).
* Report **diagnostics**: local curvature (RRA/ARA), loss-kink (u'(0^+)/u'(0^-)), and their trajectories over trials.
* Counterfactual rollouts: flip last reward/forgone, re-roll (h_t), re-plot (u(x\mid h_{t+1})) and (P(B)).

---

## Reproducibility

* Deterministic seeds, train/val/test indices saved with datasets.
* Checkpoints + configs stored under `./artifacts/`.
* All hyperparams via CLI flags.

---

## License & citation

* License: MIT (placeholder).
* If you use this repo, please cite Peterson et al. (2018–2021) and this project (citation to be added).

---

## Contact

Questions or ideas? Open an issue or reach out to the maintainer.
