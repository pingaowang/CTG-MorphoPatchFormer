# CTG-MorphoPatchFormer

Research code for fetal heart rate acceleration and deceleration event detection.
Software credit: Pingao Wang; Department of Obstetrics and Gynecology, The Seventh
Affiliated Hospital, Sun Yat-sen University.

This package contains model architectures, inference, event evaluation,
statistical analysis and aggregate results. It does **not** include training
code, dataset preparation, clinical data, annotations, record identifiers,
split lists, individual predictions or trained weights. Additional materials
require separate author approval.

## Models

The proposed model and seven neural comparators are listed below, alongside
the deterministic WMFB comparator. The neural comparators are local
reimplementations, not the original authors' software. The factory uses the profiles below; individual class
defaults and optional experimental branches are not benchmark profiles.

| Method | Source in `src/ctg_morphopatchformer` | Selected architecture |
|---|---|---|
| CTG-MorphoPatchFormer | `model.py`, `layers.py` | Local width 48, depth 3; patches 64/32; Transformer width 128, 4 heads, 2 layers; gated fusion |
| CTGNet | `comparators/unet1d.py` | Width 32, kernel 11, dropout 0.1 |
| EMAU-Net | `comparators/emaunet1d.py` | Four kernels 21/31/61/81, width 16, ECA/nonlocal attention, OR-style logits |
| ETCNN | `comparators/etcnn1d.py` | Four kernels 5/9/15/21, width 8, C-Res/CCT, CCT 5 layers and 4 heads, OR-style logits |
| LCResU-Net | `comparators/lcresunet1d.py` | Six resolutions; C-Net/L-Net residual blocks; single `LCResUNet1D`, width 8, kernel 5, dropout 0; not the four-branch ELCResU ensemble |
| MSEU-Net | `comparators/mseunet1d.py` | Width 32, kernels 7/15/21, base kernel 17, `concat_only`, dropout 0.1 |
| MTU-Net3+ | `comparators/mtunet3plus1d.py` | Existing five-level U-Net3+ adapter; historical `faithful_v3` source/configuration binding remains unverified |
| WaveFHR-VNet | `comparators/wavefhr_vnet1d.py`, `comparators/voting.py` | Five bases db4/db6/sym4/sym5/bior3.5; width 64; hard vote |
| WMFB | `third_party/wmfb/wmfb.py` | Separate weighted-median algorithm, GPL-3.0 |

ETCNN uses the accepted four-branch experiments, not the older PatchTST proxy.
Its CCT and OR-style fusion are local adaptations, not certified formula-level
copies of the original method. MTU code is provided for architecture inspection;
it is **not certified as the producer of the archived MTU results**. Constructors
do not load weights or reproduce published scores by themselves.

## Use

Python 3.10+ with the dependencies in `pyproject.toml`:

```bash
python -m pip install .
```

```python
from ctg_morphopatchformer.comparators import build_model, inspection_parameters

model = build_model("ETCNN")  # untrained, evaluation mode
parameters = inspection_parameters("LCResU")
```

Neural inputs are prepared `[batch, channel, time]` tensors. Class order is
background/acceleration/deceleration. Most models return three-class logits;
the WaveFHR voting wrapper returns labels. For whole records, use
`inference.predict_record_probabilities`; for WaveFHR, use
`inference.predict_wavefhr_record`, which reconstructs each basis before voting.

`pipelines.py` contains Raw, Common and Full. Common merges background-only
same-class gaps of at most 15 samples and removes decelerations shorter than
20 samples. Full applies the proposed method's probability rules directly;
it does not run after Common. Window/stride defaults are 1200/300 **samples**.
The caller supplies the prepared signal; inference does not preprocess it.

## Metrics and statistics

`evaluation/event_extraction.py`, `event_matching.py` and `event_metrics.py`
define events and metrics. `cluster_inference.py` contains the experiment's
pooled OOF, record-bootstrap, paired-permutation and Holm calculations;
`evaluation/api.py` provides validated array interfaces.

- Events are contiguous same-class intervals. Matching is greedy one-to-one,
  same-class temporal IoU >= 0.30, with onset-based deterministic tie breaking.
- Reported measures include class-specific and Macro Event-F1, event-count
  ratio, pointwise F1, matched-event IoU, and boundary/duration MAE.
- Pool all held-out records per seed before scoring; report the mean and SD
  across complete seeds. [`summarize_runs`](src/ctg_morphopatchformer/evaluation/api.py)
  accepts one expert label sequence and one prediction per record per seed;
  its docstring gives the required input fields.
- Bootstrap resamples whole records jointly across seeds (10,000 replicates
  by default). `paired_comparison` returns a Macro Event-F1 difference, paired
  CI and unadjusted two-sided permutation P value. Apply `holm_adjust` to the
  complete prespecified comparison family.
- No matched events gives NaN IoU/MAE. An event class absent from both truth
  and predictions is excluded from event macro F1. The retained Pred/GT
  zero-denominator sentinel is 0 and is not an interpretable ratio.

To score one record, supply an NPZ file with `y_true` and `y_pred` arrays.
Run `ctg-evaluate --help` for the required arguments. The command does not
generate cohort CIs or train a network.

Aggregate results and their update scope are described in [results](results/README.md).
Tests and internal release-audit utilities are not distributed in this package.

## Attribution and terms

Original methods: [CTGNet](https://doi.org/10.1097/FM9.0000000000000147),
[EMAU-Net](https://doi.org/10.1016/j.ins.2023.119281),
[ETCNN](https://doi.org/10.1016/j.bspc.2024.106629),
[LCResU-Net](https://doi.org/10.1016/j.eswa.2023.119610),
[MSEU-Net](https://doi.org/10.1109/IJCNN64981.2025.11227893),
[MTU-Net3+](https://doi.org/10.1007/s13534-024-00388-x),
[WaveFHR-VNet](https://doi.org/10.1109/JBHI.2025.3647057),
[WMFB](https://doi.org/10.1016/j.compbiomed.2019.103468).

First-party code uses the [noncommercial research license](LICENSE).
WMFB retains separate GPL-3.0 terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
See [license scope](LICENSE_STATUS.md) for the distinction between the first-party code and WMFB.
