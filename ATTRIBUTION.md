# Data and model sources

The real preference pairs are a course subset of [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf), helpful-base, revision `09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa`. The source's [MIT licence](https://github.com/anthropics/hh-rlhf/blob/master/LICENSE) is reproduced in `data/HH_RLHF_LICENSE.txt`. Selection rules, source checksums and split sizes are recorded in `data/provenance.json`. The original source-test metadata are preserved there for provenance; the test rows are not included in this release.

The controlled shortcut data were created for the course. Their construction and checksums appear in `data/shortcut_provenance.json`. The completed DPO tables and generated answers were produced for the assignment; their configurations and data identifiers are included under `data/dpo/`.

The frozen encoder is [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The required DPO model is [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. These weights are downloaded from their source repositories, not included in the assignment archive. The optional supplied comparisons use [SmolLM2-360M-sft](https://huggingface.co/wassname/SmolLM2-360M-sft), revision `f6fba1d52ca818ff2eff00af64c3c7f5c49cb795`.

The DPO objective is from [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), Rafailov et al. (2023).
