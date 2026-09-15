# Assignment 2 at a glance

This homework gives you hands-on experience with the concepts we studied in Module 2 on LLM alignment. It consists of five parts.

- **Part 1** trains a reward model from human preference comparisons.
- **Part 2** diagnoses what that model has learned.
- **Part 3** tests its robustness by changing prompts and responses, and by selecting responses using its scores.
- **Part 4** investigates a controlled shortcut by changing the training data and the model's features.
- **Part 5** implements DPO and fine-tunes a generative language model.

The assignment combines coding with observation and interpretation: examining plots, inspecting responses, and explaining what can reasonably be inferred from the experiments.

Each question below is worth **one mark**, for a total of **20 marks**. “Both” means a coding TODO and a written interpretation. The notebook supplies the detailed instructions and the code for repeated experiments.

## Part 1: Learn a reward model from preferences

Recall the reward model discussed in class: an encoder represents a prompt–response pair, and a linear head assigns it a scalar reward. Here, the encoder is fixed and only the head is trained. This part asks you to implement the loss and training update, examine how the model is selected, and interpret the effect of using more preference data.

| Question | Task | Type |
|---|---|---|
| Q1 | Implement the Bradley–Terry loss and sanity checks. | Coding |
| Q2 | Implement the update in a standard PyTorch training loop. | Coding |
| Q3 | Explain the supplied training loop: what is evaluated at each step, and on what basis a model is saved. | Interpretation |
| Q4 | Interpret training and validation curves for loss and accuracy. | Interpretation |
| Q5 | Compare models trained on different amounts of preference data. | Interpretation |

Optional: inspect eight separate preference pairs and compare your judgments with the recorded preferences and the model's choices.

## Part 2: Diagnose the reward model

Length bias is a common problem: a model may favour longer answers even when the additional text is not useful. Part 2 digs deeper into the validation results using a length-based rule, subgroup accuracies and a correlation plot. It then removes the prompt and changes the reward head. Two pairs on which the model and the label disagree are printed for reading. These comparisons help identify behaviour that overall accuracy may conceal.

| Question | Task | Type |
|---|---|---|
| Q6 | Calculate accuracy when the preferred response is longer or shorter, and how often the model chooses the longer response. | Coding |
| Q7 | Interpret the length diagnostics: subgroup accuracies, longer-choice rate and scatter plot. | Interpretation |
| Q8 | Train using response-only inputs and interpret what the comparison does and does not show. | Both |
| Q9 | Implement a small nonlinear reward head and compare its overall and preferred-shorter performance. | Both |

Optional: propose a rule based on another text feature, report its validation accuracy, and inspect cases where it agrees with the recorded preference but the reward model does not.

## Part 3: Stress-test the fitted reward model

This part investigates robustness to artefacts in the text. We keep the fitted model fixed and change prompts or responses from the validation set. In effect, this creates a new, somewhat contrived evaluation set. Some changes preserve the main answer, while others damage it. We examine whether the scores respond appropriately, then inspect what happens when we select edited responses by their scores.

| Question | Task | Type |
|---|---|---|
| Q10 | Calculate paired score changes after four response edits and judge whether the model responds to content. | Both |
| Q11 | Interpret what happens when the response stays fixed but the prompt changes. | Interpretation |
| Q12 | Let the reward model choose among edited candidates; judge the choices and their consequence for training. | Interpretation |

Optional: examine whether editing a rejected response can reverse an initially correct preference prediction.

## Part 4: Investigate and repair a controlled shortcut

In real data, response content, length and style vary together. We therefore use a small artificial dataset to control the association between style and preference. We first test whether the model relies on a stylistic wrapper. We then change the training data and retrain the model, and examine whether its features also need to change. This investigates how the data and representation affect what a model learns. Unlike Part 3, the training experiment itself changes. All comparisons use the same small set of prompts and answers; they do not test new-topic generalisation.

| Question | Task | Type |
|---|---|---|
| Q13 | Predict and interpret performance when style is aligned with, independent of, or opposed to preference. | Prediction and interpretation |
| Q14 | Retrain after removing the style–preference association and interpret the result. | Interpretation |
| Q15 | Implement an interaction between prompt and response features and explain its purpose. | Both |
| Q16 | Compare changing the data alone, the features alone and both together. | Interpretation |

## Part 5: Fine-tune a language model with DPO

The earlier parts train and investigate reward models. DPO instead updates a generative language model directly from preference pairs, using a fixed reference policy. This part asks you to implement response log-probabilities and the DPO loss, then run a small training experiment yourself and interpret its measurements and generated answers. If it cannot run on your machine, a supplied run can be analysed instead, for the same marks.

| Question | Task | Type |
|---|---|---|
| Q17 | Compute summed log-probabilities for response tokens, excluding the prompt and padding. | Coding |
| Q18 | Implement the DPO loss using policy and reference log-probabilities. | Coding |
| Q19 | Interpret the training and validation measurements of your DPO run and explain what the margin fraction counts. | Interpretation |
| Q20 | Compare an answer before and after fine-tuning, using evidence from its text. | Interpretation |

Optional: examine supplied experiments with different amounts of data and different beta values.
