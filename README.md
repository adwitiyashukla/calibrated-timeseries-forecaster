# calibrated-timeseries-forecaster

Probabilistic forecasting on the full M4 competition dataset. 100,000 series across six
frequencies, two global models trained across all of them, conformal prediction intervals with
measured coverage, and a small serving API.

Everything is scored with the official M4 protocol, so the numbers sit next to published results
instead of only next to each other.

## Results

The held out test set is the one the competition used: 1,277,717 future points that no part of the
pipeline sees during training or calibration.

| Model | sMAPE | MASE | OWA |
| --- | --- | --- | --- |
| Naive | 14.208 | 2.044 | 1.0581 |
| Seasonal naive | 14.657 | 2.057 | 1.0781 |
| Naive2 | 13.565 | 1.912 | 1.0000 |
| Theta | 12.312 | 1.697 | 0.8975 |
| LightGBM | 12.015 | 1.601 | 0.8616 |
| N-HiTS | 11.776 | 1.583 | 0.8481 |
| Blend | 11.816 | 1.571 | 0.8462 |

The four baselines are in there as a check on the measurement. The M4 organisers published
sMAPE 13.564 and MASE 1.912 for Naive2, and 12.309 and 1.696 for Theta. This repo measures
13.565, 1.912, 12.312 and 1.697, so the metric code, the seasonality test and the
deseasonalisation match the reference implementation. For scale, the competition winner scored
OWA 0.821 and second place 0.838.

OWA per frequency:

| Model | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly |
| --- | --- | --- | --- | --- | --- | --- |
| Theta | 0.8717 | 0.9178 | 0.9069 | 0.9693 | 0.9989 | 1.0055 |
| LightGBM | 0.8288 | 0.8701 | 0.8854 | 0.7015 | 1.0090 | 0.4221 |
| N-HiTS | 0.8024 | 0.8657 | 0.8828 | 0.7404 | 0.9735 | 0.5319 |
| Blend | 0.8039 | 0.8618 | 0.8738 | 0.6967 | 1.0090 | 0.4297 |

Series counts are 23,000 Yearly, 24,000 Quarterly, 48,000 Monthly, 359 Weekly, 4,227 Daily and
414 Hourly. Hourly is where the global models help most, because 168 lags of history carry both
the daily and the weekly cycle. Daily is where they help least, and only N-HiTS beats Naive2
there. The blend weight on Daily was fitted to 1.0, meaning pure LightGBM, and that turned out to
be the wrong pick for the final origin.

Differences in OWA against the model each one is compared with, from 2,000 bootstrap resamples
over the 100,000 series:

| Comparison | OWA difference | 95 percent interval |
| --- | --- | --- |
| Blend minus LightGBM | -0.01541 | -0.01597 to -0.01485 |
| Blend minus N-HiTS | -0.00193 | -0.00401 to +0.00012 |
| Blend minus Theta | -0.05131 | -0.05409 to -0.04851 |
| LightGBM minus Theta | -0.03590 | -0.03871 to -0.03298 |
| N-HiTS minus Theta | -0.04938 | -0.05247 to -0.04615 |

The blend beats LightGBM by a margin that survives resampling. Against N-HiTS on its own the
interval crosses zero, so the blend is ahead but not by enough to call it.

## Prediction intervals

Intervals come from conformal calibration rather than from a distributional assumption. The last
four forecast origins of every series are held out of training and used only to collect absolute
errors, scaled by the window level so scores from different series can be pooled.

Two variants are calibrated. Split conformal uses one pooled quantile per horizon step. The
adaptive variant shrinks each series towards that pooled quantile using its own four scores, so a
series with a stable history gets a tighter band.

Coverage on the test set, in percent:

| Frequency | Split 80 | Adaptive 80 | Split 95 | Adaptive 95 |
| --- | --- | --- | --- | --- |
| Yearly | 82.0 | 85.4 | 96.2 | 95.3 |
| Quarterly | 81.7 | 85.5 | 95.6 | 94.9 |
| Monthly | 76.1 | 79.5 | 94.0 | 91.2 |
| Weekly | 77.4 | 80.5 | 92.5 | 89.9 |
| Daily | 81.3 | 79.9 | 95.1 | 91.9 |
| Hourly | 80.4 | 84.9 | 95.2 | 94.4 |

Mean interval width, divided by the MASE denominator so the frequencies are comparable:

| Frequency | Split 80 | Adaptive 80 | Split 95 | Adaptive 95 |
| --- | --- | --- | --- | --- |
| Yearly | 15.50 | 13.16 | 42.81 | 26.82 |
| Quarterly | 6.25 | 5.17 | 16.07 | 10.08 |
| Monthly | 6.12 | 4.43 | 15.57 | 9.15 |
| Weekly | 17.97 | 12.16 | 41.44 | 23.90 |
| Daily | 10.68 | 10.07 | 20.11 | 14.79 |
| Hourly | 17.20 | 9.97 | 55.65 | 29.19 |

Split conformal lands within about two points of nominal on four of the six frequencies. Monthly
and Weekly come in low, which is what happens when the errors at the final origin are larger than
the ones seen during calibration. The adaptive version is much tighter, roughly half the width at
the 95 level on Hourly, and it pays for that with coverage a few points below nominal.

## Rolling origin backtest

The four calibration origins per series also work as a backtest, since no model trained on them.
That is 400,000 windows instead of the single origin the competition scores.

| Model | sMAPE | MASE | OWA |
| --- | --- | --- | --- |
| Naive2 | 13.627 | 2.032 | 1.0000 |
| Theta | 13.086 | 1.818 | 0.9275 |
| LightGBM | 11.273 | 1.569 | 0.7997 |
| N-HiTS | 11.445 | 1.587 | 0.8105 |
| Blend | 11.182 | 1.554 | 0.7927 |

The ordering matches the test set, so the ranking is not an artefact of one lucky origin.

## How it works

Data. The six wide M4 CSVs are pulled from GitHub and converted into long parquet panels, one per
frequency, with the series index and step as columns. Nothing is held in memory across
frequencies.

Windows. Each series contributes windows of `lookback` inputs and `horizon` targets, sampled
evenly across its history and capped per series. The last four origins go to calibration, five
percent of series are held out entirely for early stopping and for fitting the blend weight, and
everything else trains. A test asserts that no training or calibration window can reach past the
end of the training history.

Features. One feature spec builds the matrix for training and for serving, so the two cannot drift
apart. Each window is divided by its own mean absolute level, which makes series of wildly
different scale comparable. On top of the scaled lags there are window statistics: standard
deviation, slope, first and seasonal autocorrelation, and the position in the seasonal cycle.

Models. LightGBM is trained direct multi-horizon, one model per step, so 107 models across the six
frequencies, sharing a single binned dataset per frequency. N-HiTS is a plain PyTorch
implementation with max pooling at three rates and linear interpolation back up to the horizon,
trained globally on CPU. The blend weight is a single number per frequency, chosen on held out
series by minimising OWA rather than mean absolute error, because the two disagree and OWA is what
gets reported.

## Serving

`forecaster serve` runs a FastAPI app. Post a history and a frequency, get back the point forecast
and both interval levels. If the series id is one the model was calibrated on, the response uses
the adaptive interval and says so.

Single request latency on two CPU cores:

| Frequency | p50 ms | p90 ms | p99 ms |
| --- | --- | --- | --- |
| Yearly | 1.48 | 5.50 | 32.58 |
| Quarterly | 1.67 | 5.68 | 25.37 |
| Monthly | 6.76 | 8.13 | 32.96 |
| Weekly | 6.48 | 8.83 | 34.72 |
| Daily | 6.30 | 8.78 | 38.80 |
| Hourly | 24.08 | 40.15 | 48.53 |

Latency tracks the horizon because direct multi-horizon means one LightGBM call per step, and
Hourly has 48 of them. The neural half runs through ONNX Runtime and costs 0.17 ms, and export is
checked against the PyTorch output to 1e-6 before it is written.

## Running it

Python 3.11 or newer.

```
pip install -e ".[dev]"
```

```
forecaster all
```

That runs every stage in order: download and panel build, baselines, training, blend weight,
conformal calibration, backtest, ONNX export, evaluation and the latency benchmark. The download
is about 260 MB. On two cores the whole thing takes a little under two hours, and training is
almost all of it.

Stages also run on their own, for example `forecaster evaluate`, and each one writes into
`artifacts/`. Every setting lives in `config.toml` and is validated on load, so a bad value fails
immediately with the key that caused it. `window_cap_scale` shrinks the number of training windows
if you want a faster pass.

```
pytest -q
```
