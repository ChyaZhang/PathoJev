# PathoJev

<p align="left">
  <a href="https://huggingface.co/Mikezcy/PathoJev"><img src="https://img.shields.io/badge/🤗%20Model-PathoJev--9B%20%7C%2027B-yellow" alt="Model"></a>
</p>

## Introduction📝

PathoJev answers pathology questions directly as probabilities over the answer options, for whole-slide images
(WSIs) and regions of interest (ROIs). It builds on the System One model JEV and a Qwen3.5 hybrid backbone
(Gated DeltaNet + softmax attention). A WSI enters as a 32 x 32 grid of pooled CONCH v1.5 patch features. The slide is
encoded once, and every question is evaluated as an isolated row on the shared slide context, so many questions about
the same slide cost little more than one. ROI images go through the backbone's own vision tower.

Each answer is read from a typed decision head in one forward pass, without generating or parsing text. Supported
question types are multiple choice (up to 16 options) and true/false statements.

## TODOs📌

- [x] PathoJev-9B and PathoJev-27B weights
- [x] Inference code and demo
- [ ] Paper
- [ ] Training and evaluation code

## Installation🛠️

```bash
git clone https://github.com/ChyaZhang/PathoJev.git
cd PathoJev
conda create -n pathojev python=3.11 -y
conda activate pathojev
pip install -r requirements.txt
```

The weights are gated on Hugging Face. Request access on the [model page](https://huggingface.co/Mikezcy/PathoJev),
then log in with `hf auth login`. PathoJev-9B needs about 20 GB of GPU memory and PathoJev-27B about 60 GB (bf16 backbone). The backbones
([Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) for 9B, [autotrust/JEV-27B-VL](https://huggingface.co/autotrust/JEV-27B-VL)
for 27B) are downloaded on first use.

## Demo🏃

The demo answers the eight SlideBench-TCGA questions of one bladder carcinoma slide (TCGA-XF-AAMJ) with a single slide
encoding.

1. Get the slide's CONCH v1.5 patch features. They are available in the gated dataset
   [sofieneb/conch_v15_features](https://huggingface.co/datasets/sofieneb/conch_v15_features) (request access first):

```bash
hf download sofieneb/conch_v15_features --repo-type dataset --local-dir features \
    --include "TCGA/TCGA-BLCA/TCGA-XF-AAMJ-01Z-00-DX1.726E3D84-8370-45AC-9C98-ACA30CAB2359.h5"
```

2. Run PathoJev:

```bash
python demo.py --features features/TCGA/TCGA-BLCA/TCGA-XF-AAMJ-01Z-00-DX1.726E3D84-8370-45AC-9C98-ACA30CAB2359.h5
```

Expected output (PathoJev-9B):

```
[Cytomorphological Characteristics] Based on the cellular morphology observed in the bladder tissue scan,which nuclear feature is most prominent?
  -> Prominent nucleoli  p = 0.64 ✓
[Differential Diagnosis] Based on the features observed in the provided whole slide image,which of the following could be considered a differential diagnosis for the lesions in the bladder?
  -> Transitional cell carcinoma  p = 0.97 ✓
[Disease Classification] Based on the provided histological image of the bladder tumor,what is the most likely diagnosis?
  -> Urothelial carcinoma,high-grade  p = 1.00 ✓
[Grading] What is the histopathological grade of the carcinoma shown in the bladder tissue slide?
  -> High-grade  p = 0.98 ✓
[Histopathological Changes] What additional histopathological changes are observed in the provided bladder sections apart from the main carcinoma?
  -> Cystitis cystica and glandularis  p = 0.99 ✓
[Staging] According to the provided whole slide image of the bladder tumor,which TNM stage corresponds to the pathological assessment?
  -> pT3bN0M0  p = 0.98 ✓
[Tissue Architecture and Arrangement] What type of tissue involvement can be observed in the provided bladder tissue slide?
  -> Muscularis propria invasion  p = 1.00 ✓
[Tumor Characteristics] Examine the whole slide image: What is the extent of invasion of the tumor in the bladder?
  -> Invading the entire bladder wall and perivesical tissue  p = 0.99 ✓
```

Add `--size 27b` for PathoJev-27B. Probabilities can differ in the second decimal across GPUs and kernels.

## Usage

```python
from PIL import Image
from pathojev import PathoJev, make_grid

model = PathoJev.from_pretrained("9b")  # or "27b"

# whole slide: CONCH v1.5 features (TRIDENT, 20x, 512 x 512 px patches) pooled onto a 32 x 32 grid
grid = make_grid("slide.h5")
questions = [
    {"question": "What is the histologic grade of the tumor?", "options": ["Low-grade", "High-grade"]},
    {"question": "Lymphovascular invasion is present.", "kind": "noul"},  # true / false statement
]
print(model.answer_slide(grid, questions))
# [{"answer": "High-grade", "probs": {"Low-grade": ..., "High-grade": ...}}, {"answer": "true", "probs": {...}}]

# region of interest: one question per image
print(model.answer_roi([Image.open("roi.png").convert("RGB")],
                       [{"question": "Which tissue is shown?", "options": ["Colon", "Lung", "Breast"]}]))
```

Features of new slides can be extracted with [TRIDENT](https://github.com/mahmoodlab/TRIDENT)
(`--patch_encoder conch_v15 --mag 20 --patch_size 512`). The probabilities are the decision head's softmax at
temperature 1, without post-hoc calibration.

## Results📊

Accuracy (%) on three whole-slide and five ROI benchmarks.

| Model | WSI-VQA | WSI-Bench | SlideBench-TCGA | PathMMU-tiny | PathVQA | Quilt-VQA | OmniMedVQA | MedXpertQA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PathoJev-9B | 65.4 | 90.4 | 75.9 | 71.9 | 94.6 | 70.8 | 75.3 | 25.6 |
| PathoJev-27B | 66.4 | 91.1 | 76.9 | 73.8 | 95.1 | 67.3 | 69.9 | 50.0 |

WSI-Bench and SlideBench-TCGA are the closed-ended sets, PathVQA the yes/no subset, Quilt-VQA the closed-ended subset,
and OmniMedVQA and MedXpertQA their pathology subsets. On one RTX PRO 6000, PathoJev-9B takes 0.039 s per WSI-Bench
question and 0.090 s per PathMMU-tiny question (GPU forward time, batch size 1).

## Acknowledgements✨

PathoJev builds on [JEV](https://huggingface.co/autotrust/JEV-9B), [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-9B),
[CONCH](https://huggingface.co/MahmoodLab/conchv1_5) and [TRIDENT](https://github.com/mahmoodlab/TRIDENT). Training uses
SlideInstruct, WSI-Bench, WSI-VQA, TCGA pathology reports, PathVQA and PMC-VQA; we thank their authors for releasing
these resources.

## License

Code and weights are released under [CC BY-NC 4.0](LICENSE) for non-commercial research use. CONCH v1.5 and the
pre-extracted TCGA features have their own licenses. PathoJev is not a medical device and must not be used for
clinical decision-making.
