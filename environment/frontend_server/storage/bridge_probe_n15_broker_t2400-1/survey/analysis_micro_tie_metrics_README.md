# analysis_micro_tie_metrics.csv

Structured micro-tie perception metrics produced by `analyze_survey.py`.

## Aggregation rule

Each respondent answers two directed micro-tie questions per unordered
pair {j, k}: one with `target_j = j, target = k` and one with
`target_j = k, target = j`. For confusion-matrix computation, these are
aggregated to a single unordered "said-tied" judgment using the **OR
rule**:

    perceived_pair = 1  iff  at least one directed observation is "1"
    perceived_pair = 0  iff  all valid directed observations are "0"

## Fail-safe policy

Rows with `is_fail_safe == "1"` or non-binary `value` are dropped from
the confusion counts. A pair is excluded from `n_pairs_evaluated` only
if ALL of its directed observations are dropped. The count of dropped
directed rows per respondent is reported in the `n_fail_safe` column.

## Ground truth

`tie_cumulative` from the matching `ground_truth/ground_truth_edges_{step}.csv`
is the reference label for each unordered pair. A pair with
`tie_cumulative == 1` is the actual-positive class.

## Columns

| Column               | Meaning                                                              |
|----------------------|----------------------------------------------------------------------|
| wave_id              | Survey wave identifier (e.g. "pre", "post")                          |
| step                 | Simulation step at which the wave was captured                        |
| respondent           | Agent name, or "__overall__" for the per-wave micro-average          |
| tp                   | True positives: said-tied AND tie_cumulative==1                       |
| fp                   | False positives: said-tied AND tie_cumulative==0                      |
| fn                   | False negatives: said-not-tied AND tie_cumulative==1                  |
| tn                   | True negatives: said-not-tied AND tie_cumulative==0                   |
| precision            | tp / (tp + fp); empty cell when denominator is 0                      |
| recall               | tp / (tp + fn); empty cell when denominator is 0                      |
| fpr                  | fp / (fp + tn); empty cell when denominator is 0                      |
| fnr                  | fn / (fn + tp); empty cell when denominator is 0                      |
| pair_accuracy        | (tp + tn) / (tp + fp + fn + tn)                                       |
| n_fail_safe          | Number of directed micro-tie rows with is_fail_safe==1 (dropped)      |
| n_pairs_evaluated    | Number of unordered pairs with at least one valid observation         |

The `__overall__` row per wave is a **micro-average**: TP/FP/FN/TN are
summed across respondents and the ratios are recomputed from the sums.

## Invariants

    tp + fp + fn + tn == n_pairs_evaluated    # for every row
