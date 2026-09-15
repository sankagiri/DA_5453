# DA5453 — Assignment 2: Learning from preferences

Start with [the overview](OVERVIEW.md), then complete [the notebook](DA5453_coding_assignment_handout.ipynb). The assignment has five parts and **20 questions worth one mark each**. Coding and interpretation tasks are labelled separately. Optional activities are unmarked.

## Repository contents

You do not need to open every file. Keep the supplied folder structure unchanged so that the notebook can find its supporting code and data.

- `DA5453_coding_assignment_handout.ipynb` is the notebook you complete and submit.
- `OVERVIEW.md` maps the five parts and Q1–Q20; `ATTRIBUTION.md` records the dataset and model sources.
- `requirements.txt` lists the Python packages, and `reward_support.py`, `diagnostic_support.py` and `dpo_support.py` contain supplied helper code. Keep these files beside the notebook; you do not need to edit them.
- `data/pilot_*.csv` contains the main preference-training, validation and optional orientation pairs used in Parts 1–3 and 5.
- `data/shortcut_*.csv` contains the small controlled datasets used in Part 4.
- `data/dpo/reference_135m/` contains metrics and generated answers from the supplied Part 5 run, without model weights. Use it only if you take the documented supplied-results route.
- `data/dpo/staff_*` contains additional runs used only by the optional Part 5 comparison. The other JSON, manifest and licence files record data provenance, configurations and licensing; no action is required on them.

## Set up

You need Python 3.11 (3.10–3.12 should also work), a terminal, and about 500 MB of disk for the downloaded models. On GitHub, select **Code → Download ZIP**, extract the download, and open a terminal inside the extracted folder. Then create an environment and install the pinned libraries:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

A conda environment works equally well; the only requirement is that `pip install -r requirements.txt` runs inside it. Open `DA5453_coding_assignment_handout.ipynb` from the Jupyter file browser and run the Setup cell; it checks that the three Python support files and `data/` are beside the notebook. You do not need to edit the Python files.

The first run needs internet access to download the text encoder (about 90 MB) and, if you train in Part 5, the language model (about 270 MB). Later runs reuse the downloaded files. Parts 1–4 run on CPU in a few minutes. Part 5 training is faster on a GPU (CUDA or Apple Silicon) but also completes on CPU; the `provided` route needs no model download at all. No particular speed or model score is required for marks.

## Working through the questions

Complete the marked coding TODOs and replace the written-answer placeholders. Keep the supplied settings for the required experiments so that the comparisons remain meaningful. Cells that call the supplied Python files carry a comment saying what they compute and display. Short answers supported by the actual tables, plots or response text are sufficient; approximate sentence counts are guidance.

The recorded preferences are observations, not guarantees of answer quality. An answer may be incomplete or questionable even when it is labelled chosen. Inspect the text when interpreting model errors. The optional activities can be skipped by leaving their switches `False`.

## Part 5: DPO

Part 5 fine-tunes SmolLM2-135M-Instruct with your Q17 and Q18 functions: 64 pairs, two epochs, beta 0.1. The first run downloads the model (about 270 MB). Training takes a few minutes on a GPU (CUDA or Apple Silicon) and noticeably longer on a CPU; the progress bars also cover reference scoring, validation and generation, so an epoch is more than its optimisation updates. Results (metrics and generated answers, not weights) are saved under `results/`; a completed run is never trained again unless you delete its folder.

If Part 5 genuinely cannot run on your machine, analyse the supplied run in `data/dpo/reference_135m` instead and say so in Q19. Q17 and Q18 are required either way, and the marks are the same.

## Submit

**Deadline:** Sunday, 4 October 2026, 11:59 PM Indian Standard Time (IST).

- Add your name and roll number.
- Complete Q1–Q20, including the prediction before Q13's experiment.
- Retain the outputs and plots that support your answers. Check Q1, Q17 and Q18 sanity checks.
- If Q19 uses the supplied run rather than your own, say so.
- Rename the completed notebook `ROLLNUMBER_DA5453_Assignment2.ipynb`, replacing `ROLLNUMBER` with your own roll number.
- Submit that `.ipynb` file through the [assignment submission form](https://docs.google.com/forms/d/e/1FAIpQLSfJTfSfDTtguBdAeX3STW_lUxZZF4WkimQL2oKiYvB-M0VEVw/viewform). Do not submit model weights or the surrounding folder.
- Keep the emailed response receipt. To replace your notebook before the deadline, use **Edit your response** from that receipt and upload the revised file; do not create a second submission.

Dataset and model sources are listed in [ATTRIBUTION.md](ATTRIBUTION.md). The supplied data include 1,536 training pairs, 256 validation pairs and eight separate pairs for the optional judgment activity. Use the supplied splits; do not fetch additional rows from the source dataset.
