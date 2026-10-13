# Predictive Maintenance of a Metro Air Compressor (MetroPT3)

Proof of Concept (PoC) for the course **SIN 323 – Artificial Intelligence (2026/2)**, Practical Project.

## 0. Ambient creation

```bash
conda create --name sin323 python=3.12.14
conda activate sin323

pip install -r requirements.txt
```

## 1. Purpose

A metro operator relies on an air compressor (the pneumatic unit that feeds the braking and door systems of the trains). When it fails, trains are taken out of service and maintenance is done reactively. The operator wants to know **hours before a failure that the compressor is likely to break**, so maintenance can be scheduled in advance.

This project asks whether machine learning on the compressor's sensor stream can **anticipate failures** and turn that prediction into a risk signal that a maintenance team can act on.

The project is judged on the brief's criteria: problem and relevance (20%), suitability of the AI (25%), quality of the PoC (20%), validation and results (20%), and poster/demo (15%). **Negative results are acceptable** as long as they are measured honestly and explained. A well-validated "this does not work yet, and here is why" is a valid outcome.

### Who has the problem and what they do today
- **Users:** the maintenance team of the metro operator.
- **Today:** fixed-interval maintenance and reactive repair after a failure, which means train downtime and unplanned costs.
- **Desired result:** an early, explainable warning with a known false-alarm rate, plus a view of the sensor trend that led to it.

## 2. Data

- **Source:** MetroPT3 dataset (UCI Machine Learning Repository), air compressor of a Metro do Porto train. See `data/Data Description_Metro.pdf`.
- **File:** `data/MetroPT3(AirCompressor).csv`, about 1.5 million rows, one sample about every 10 seconds, 1 Feb – 1 Sep 2020, 16 columns (analog pressures, oil temperature, motor current, and digital valve/switch signals).
- **Labels:** four failure windows reported by the company (`config.FALHAS`):
  1. 18 Apr 2020 (all day)
  2. 29–30 May 2020
  3. 5–7 Jun 2020
  4. 15 Jul 2020
  Maintenance dates are in `config.MANUTENCOES`. Only four events exist, which is the main limitation of the project.
- **Restrictions:** the data is anonymised machine data, so privacy risk is low. Use must follow the dataset's terms and be cited in the references.

## 3. Approaches to anticipate failure

The current code implements approach A. The other ideas are listed so the team can choose what to explore next.

### A. Supervised early warning (implemented)
Predict, for every hour, whether a failure starts in the next 24 h.
- Features: hourly aggregates (duty cycle, load transitions, pressure and oil means and standard deviations), 3 h lags and a 6 h oil trend.
- Models: logistic regression, gradient boosting; baseline is the duty-cycle rule.
- Validation: leave-one-failure-out, with a test window per failure and a 24 h purge around it.

### B. Horizon and label tuning
The horizon (`HORIZONTE_H`) is a guess. Test 6 h, 12 h, 48 h and 72 h. A short horizon is more precise but gives less time to act; a long horizon is easier to predict but less useful.

### C. Richer, cycle-level features
The compressor runs in cycles (load/unload). Per-cycle features may carry more signal than hourly averages:
- Duration of each load cycle and the idle time between cycles.
- Pressure decay rate in the reservoirs/towers while the compressor is off (a leak shows up as faster decay).
- Time for the tank pressure to rebuild after a cycle.
- Oil temperature rise rate per cycle.
- Number of cycles per hour (`transicoes` already gives a rough version).
- Valve timing: lag between `COMP` / `DV_eletric` and pressure changes.
- Motor current signature during loaded operation (its mean and spread, not only the on/off threshold).

### D. Anomaly detection (no labels needed)
Train only on normal operation (for example March 2020, which is quiet) and flag deviations.
- Isolation Forest, One-Class SVM, or an autoencoder that reconstructs normal multi-sensor windows; alarm when the reconstruction error rises.
- Advantage: works even if only a few failures are labelled. Risk: may flag normal but unusual operation.

### E. Trend and change-point detection
Detect a drift in the mean or variance of a sensor (CUSUM, Bayesian change-point, or a simple rolling z-score). A sustained drift before a failure is an explainable warning.

### F. Time-to-failure regression / survival analysis
Instead of a yes/no label, predict the hours remaining until the next failure, or use a survival model (Cox, Random Survival Forest). This matches how maintenance is actually planned.

### G. Sequence models
A 1D CNN or LSTM over the last N hours of raw or resampled signals. More expressive, but with four failures this will almost certainly overfit. Only consider it if simpler models show clear signal.

### H. Hybrid / rule-based baseline
Combine simple engineering rules (for example, pressure does not reach the setpoint within X minutes) with the ML score. Useful for explanation and as a strong baseline.

### I. Risk communication
- Traffic-light levels (`risk.py`: baixo / médio / alto) with thresholds chosen from the precision–recall curve rather than by hand.
- Time-line dashboard per train or per compressor (Streamlit): sensor trends, probability, and the reason for the alert (top features).
- Alert policy: require N consecutive high hours before raising an alarm, to reduce noise.

## 4. Possible results

The project is designed to report whichever outcome occurs. Possible results:

1. **Positive:** a model beats the duty-cycle baseline on PR-AUC and ROC-AUC across all four failures, with a useful lead time and an acceptable false-alarm rate. Report the numbers with the confidence interval from the four folds.
2. **Partially positive:** the model detects some failures (for example, the ones with a clear pressure or oil signature) but not others. Report which failure types are detectable and why the others are not.
3. **Negative, measured:** no approach beats the chance level or the simple rule out-of-sample. Report this, and explain what the features miss (for example, a failure whose signal is not in these 16 sensors, or too few events to learn from). The poster can still show a working, validated pipeline and the evidence for the negative result.
4. **Dataset-limited:** the four events are too few to conclude anything. Report the variance across folds and state that conclusions are provisional.

### Current status of results

**Approach A (supervised 24 h prediction): negative.** Out-of-sample, leave-one-failure-out, 892 hours (96 positive, chance PR-AUC 0.108):

| Method | PR-AUC | ROC-AUC |
|---|---|---|
| Duty-cycle rule (no learning) | 0.126 | 0.51 |
| Logistic regression | 0.081 | 0.37 |
| Gradient boosting | 0.072 | 0.29 |

No model beats the rule. The direction changed to anomaly detection with component alerts (approaches D and E, with the alert levels of I).

**Data findings.**
- The Kaggle copies are the same 1.5M-row UCI file (10 s steps). About 38 days are missing, and 2.8% of the rows are a frozen logger repeating old values (up to 62 h). The cleaned dataset (regular grid with quality flags) is built by `scripts/clean_data.py`.
- Valid data in the 24 h before each failure: 22% (18 Apr), 86% (29 May), 100% (5 Jun), 71% (15 Jul).
- All 4 labelled failures fall inside continuous-load episodes. Three unlabelled episodes (12 Apr, 13 May, 19-20 May) look like the labelled ones, so the labels are probably incomplete. Confirming this needs maintenance records.

**Approach: component anomaly alerts (current).**
- Six components (oil, pressure, motor, dryer, valves, flow) with severity 2-5 (our own synthesis from the MetroPT paper, not an official scale).
- Score: robust z against the same hour of day over the previous 14 days. Three levels (information, warning, critical) with thresholds calibrated on normal data before 2020-04-10. System information needs 2 or more components for 2 h.
- Evaluation follows Fig. 9 of Veloso et al. (arXiv 2207.05466): a predicted interval overlapping an observed failure is a true positive; no overlap is a false positive or false negative. Test period after 2020-04-10, thresholds trained without any failure.

| Level | Failures detected | Probable episodes detected | Started 2 h or more early | False intervals per week |
|---|---|---|---|---|
| Information | 4/4 | 3/3 | 2 of 7 events | 0.46 |
| Warning | 4/4 | 3/3 | 0 | 0.07 |
| Critical | 3/4 | 3/3 | 0 | 0.07 |

**Outcome: partially positive.** The system detects the anomalous state during leaks with few false alarms at warning level. Early warning before the failure starts is shown for only 2 of 7 events, so it is not demonstrated with 4 labelled events (3 more are only probable). Warning starts 3-8 h after the failure begins. Treat all numbers as provisional.

## 5. Plan

**Phase 1 – predict the four known failures (current)**
- Done: data preparation, EDA, labelling, first supervised models, validation, figures, simulation.
- Next: approaches B and C (horizon tuning, cycle-level features), then D (anomaly detection) as an independent check.
- Decide by the end of phase 1 whether the signal exists. If it does not, document the negative result and move on.

**Phase 2 – extend to other failures**
- Use the MetroPT3 report and other failures in the dataset, if the company reports list them.
- Test whether the best method generalises to failure types it was not tuned on.

**Phase 3 – product**
- Streamlit demo: scenario replay, risk timeline, alert explanation.
- Poster following the brief's structure (introduction, objective, methodology, architecture, results, discussion, conclusion, references).

## 6. Limitations and responsible use

- Four failure events: any metric has very wide uncertainty.
- The data is from one compressor on one train; results may not transfer.
- Thresholds (`LIMIAR_CARGA`, `RISCO_*`) are set from one histogram and a provisional rule; they must be validated.
- The system gives a recommendation, not a shutdown command. A human technician decides on maintenance.
- False alarms have a cost (unnecessary inspection) and missed failures have a larger cost (train out of service); the threshold should reflect that trade-off, which the team must discuss.

## 7. Repository layout

```
config.py          paths, failure windows, thresholds, validation settings
src/data.py        load CSV, Parquet cache, scenarios, gap detection
src/features.py    hourly features with coverage filter
src/labels.py      failure-within-horizon labels
src/model.py       dataset, models, leave-one-failure-out validation, final training
src/risk.py        probability to risk level
src/simulador.py   scenario timeline and lead-time summary
src/report.py      figures
scripts/           prepare_data, eda_plots, train, evaluate, run_simulation
reports/figuras/   generated figures
reports/resultados/ out-of-sample predictions and metrics
models/            saved model
```

## 8. How to run

Create the environment first (section 0). Download `MetroPT3(AirCompressor).csv` from the [UCI repository](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset) (or Kaggle, same file) and place it in `data/`. The CSV and the generated Parquet files are not in the repository.

Run from the repository root. On a machine without a display, set `MPLBACKEND=Agg` first (`eda_plots.py` calls `plt.show()`).

### Order of the scripts

Main pipeline (component alerts):

```bash
python scripts/prepare_data.py          # CSV -> Parquet, gap report
python scripts/clean_data.py            # cleaned 10 s grid, frozen-logger and gap flags
python scripts/episodes.py              # continuous-load episodes
python scripts/run_alerts.py            # component scores and the three alert levels
python scripts/plot_alerts.py           # alert timeline figure
python scripts/unlabelled_episodes.py   # compare unlabelled episodes with the labelled failures
python scripts/evaluate_intervals.py    # interval evaluation (overlap, false positives, 2 h lead)
```

Optional (exploration and the first, negative approach):

```bash
python scripts/eda_plots.py             # EDA figures 01-03 (after prepare_data)
python scripts/train.py                 # supervised 24 h prediction, leave-one-failure-out
python scripts/evaluate.py              # PR curve, per-failure probability, confusion matrix (after train)
python scripts/run_simulation.py        # risk timelines of the first approach (after train)
python scripts/leak_proxy.py            # leak proxies per episode
python scripts/leak_proxy_demanda.py    # same, controlled for demand
python scripts/anomaly_eval.py          # cycle-level isolation forest
```

`leak_proxy*.py` and `anomaly_eval.py` read `reports/resultados/ciclos.parquet`, which no script in this repository currently generates (see `src/ciclos.py`); they only work if that file exists.

## 9. References

- MetroPT3 dataset: Davari, N., Veloso, B., Costa, G., Pereira, P., Ribeiro, R., Gama, J. (UCI Machine Learning Repository / PHM 2021 data challenge).
- scikit-learn, pandas, matplotlib (libraries used).
