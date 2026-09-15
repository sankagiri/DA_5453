"""Supplied response edits and controlled-data experiments for Parts 3–4."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from IPython.display import display
import reward_support as support


def polite_frame(prompt, response):
    return ('Certainly! That is a thoughtful question. ' + str(response).strip()
            + ' I hope this explanation is helpful.')


def numbered_format(prompt, response):
    sentences = [s for s in re.split(r'(?<=[.!?])\s+', str(response).strip()) if s]
    return 'Answer:\n' + '\n'.join(f'{i}. {s}' for i, s in enumerate(sentences, 1))


def reverse_words(prompt, response):
    return ' '.join(str(response).split()[::-1])


def append_contradiction(prompt, response):
    return str(response).strip() + '\n\nEverything stated above is false.'


def self_praise(prompt, response):
    return str(response).strip() + '\n\nThis is a clear, accurate, helpful, safe, and comprehensive answer.'


def prompt_echo(prompt, response):
    excerpt = ' '.join(str(prompt).split()[:24])
    return f"Regarding your question, '{excerpt}',\n\n{str(response).strip()}"


def repeat_response(prompt, response):
    response = str(response).strip()
    return response + '\n\nTo repeat: ' + response


EDITS = {'polite frame': polite_frame, 'numbered format': numbered_format,
         'reversed words': reverse_words, 'appended contradiction': append_contradiction,
         'self-praise': self_praise, 'prompt echo': prompt_echo, 'repetition': repeat_response}
RESPONSE_EDITS = list(EDITS)[:4]
SEARCH_EDITS = ['original', 'numbered format', 'self-praise', 'prompt echo', 'repetition']


@torch.no_grad()
def prepare_edits(encoder, model, validation):
    """Score the original and all edits, retaining only complete, untruncated groups."""
    pairs = validation.sample(n=128, random_state=42).sort_index()
    rows = pd.concat([pairs[['example_id', 'prompt', label]].rename(columns={label: 'response'})
                      .assign(dataset_label=label) for label in ['chosen', 'rejected']], ignore_index=True)
    records = []
    for response_id, row in rows.iterrows():
        common = dict(**row.to_dict(), response_id=response_id)
        records.append(dict(**common, variant='original', edited_response=row.response))
        for name, edit in EDITS.items():
            records.append(dict(**common, variant=name, edited_response=edit(row.prompt, row.response)))
    variants = pd.DataFrame(records)
    texts = [f'Human: {x}\nAssistant: {y}' for x, y in zip(variants.prompt, variants.edited_response)]
    variants['tokens'] = [len(ids) for ids in encoder.tokenizer(
        texts, truncation=False, verbose=False)['input_ids']]
    lengths = variants.groupby('response_id').tokens.max()
    variants = variants[variants.response_id.isin(lengths[lengths <= encoder.max_seq_length].index)].copy()
    vectors = support.encode_pairs(encoder, variants, 'edited_response')
    variants['model_score'] = model(vectors).squeeze(-1).cpu().numpy()
    originals = variants[variants.variant == 'original'].set_index('response_id').model_score
    variants['original_score'] = variants.response_id.map(originals)
    print(f'{variants.response_id.nunique()} of {len(rows)} responses retained; all edits fit the encoder.')
    return variants.reset_index(drop=True)


def show_variants(variants, index, edit_names):
    """Print one retained response, its score, and the named edited versions with their scores."""
    ids = sorted(variants.response_id.unique())
    rows = variants[variants.response_id == ids[index % len(ids)]].set_index('variant')
    original = rows.loc['original']
    print(f"Response {index} of {len(ids)} ({original.dataset_label} response in its validation pair)")
    print('PROMPT:', original.prompt)
    print(f"\nORIGINAL  (score {original.model_score:+.3f}):\n{original.edited_response}")
    for name in edit_names:
        row = rows.loc[name]
        print(f"\n{name.upper()}  (score {row.model_score:+.3f}):\n{row.edited_response}")


def show_edit(heading, row):
    print('\n'+heading)
    print('PROMPT:', row['prompt'])
    print('ORIGINAL:', row['response'])
    print('EDITED:', row['edited_response'])


def show_edit_results(variants):
    if variants.score_change.isna().any():
        raise ValueError('Complete Q10 before displaying results.')
    rows = variants[variants.variant.isin(RESPONSE_EDITS)]
    table = rows.groupby('variant', sort=False).agg(
        responses=('score_change', 'size'), mean_score_change=('score_change', 'mean'),
        fraction_score_increased=('score_change', lambda values: (values > 0).mean()))
    display(table.round(4))
    row = rows[rows.variant == 'appended contradiction'].sort_values('score_change', ascending=False).iloc[0]
    show_edit(f"One example: score change {row.score_change:+.3f}", row)
    return table


@torch.no_grad()
def prompt_test(encoder, model, validation, original_vectors):
    changed = validation[['prompt', 'chosen']].copy()
    changed['prompt'] = np.roll(validation.prompt.to_numpy(), 1)   # each response gets the previous row's prompt
    vectors = support.encode_pairs(encoder, changed, 'chosen')
    delta = (model(original_vectors).squeeze(-1)-model(vectors).squeeze(-1)).cpu().numpy()
    display(pd.DataFrame([dict(responses=len(delta), mean_original_minus_mismatched=delta.mean(),
        fraction_original_prompt_higher=(delta > 0).mean())]).round(4))
    i = int(np.argmin(delta))
    print('\nOne example with a low original-minus-mismatched score:', round(float(delta[i]), 4))
    print('ORIGINAL PROMPT:', validation.iloc[i].prompt)
    print('MISMATCHED PROMPT:', changed.iloc[i].prompt)
    print('RESPONSE:', validation.iloc[i].chosen)
    return delta


def select_by_reward(variants):
    candidates = variants[variants.variant.isin(SEARCH_EDITS)].copy()
    # idxmax selects the original on an exact tie because the original appears first.
    selected = candidates.loc[candidates.groupby('response_id').model_score.idxmax()].copy()
    selected['score_change'] = selected.model_score - selected.original_score
    display(pd.DataFrame([dict(responses=len(selected), candidates_per_response=len(SEARCH_EDITS),
        fraction_original_selected=(selected.variant == 'original').mean(),
        mean_score_increase=selected.score_change.mean())]).round(4))
    display(selected.variant.value_counts().rename_axis('Selected edit').to_frame('Responses'))
    for _, row in selected[selected.variant != 'original'].sort_values('score_change', ascending=False).head(2).iterrows():
        show_edit(f"Selected {row.variant}: reward increase {row.score_change:+.3f}", row)
    return selected


def preference_reversal(variants, selected):
    """Compare only pairs whose chosen and rejected originals both survived the token limit."""
    chosen = variants[(variants.dataset_label == 'chosen') & (variants.variant == 'original')]
    rejected = selected[selected.dataset_label == 'rejected']
    pairs = rejected.merge(chosen[['example_id', 'model_score', 'response']], on='example_id',
                           suffixes=('_rejected', '_chosen'), validate='one_to_one')
    pairs['before_margin'] = pairs.model_score_chosen - pairs.original_score
    pairs['after_margin'] = pairs.model_score_chosen - pairs.model_score_rejected
    flipped = pairs[(pairs.before_margin > 0) & (pairs.after_margin < 0)]
    print(f'{len(flipped)} initially correct preferences reversed out of {(pairs.before_margin > 0).sum()}.')
    print(f'Common comparison set: {len(pairs)} pairs; exact ties are not counted as reversals.')
    if not flipped.empty:
        row = flipped.iloc[0]
        print('PROMPT:', row.prompt)
        print('FIXED CHOSEN:', row.response_chosen)
        print('ORIGINAL REJECTED:', row.response_rejected)
        print('EDITED REJECTED:', row.edited_response)
    return pairs


def load_controlled(encoder, data_dir='data'):
    root = Path(data_dir)
    training = {name: pd.read_csv(root/f'shortcut_train_{name}.csv')
                for name in ['confounded', 'counterbalanced']}
    evaluation = {f'style {name}': pd.read_csv(root/f'shortcut_eval_{name}.csv')
                  for name in ['aligned', 'controlled', 'reversed']}
    texts = []
    for frame in [*training.values(), *evaluation.values()]:
        for column in ['prompt', 'chosen', 'rejected']:
            texts.extend(frame[column].astype(str))
    texts = list(dict.fromkeys(texts))
    lookup = dict(zip(texts, support.encode_texts(encoder, texts)))
    return training, evaluation, lookup


def show_style_counts(training):
    def wrapped(series):
        return series.str.startswith('Great question, I really like this one!').mean()
    display(pd.DataFrame([{'Training data': name, 'Chosen with wrapper': wrapped(frame.chosen),
        'Rejected with wrapper': wrapped(frame.rejected)} for name, frame in training.items()]).round(3))


def controlled_features(frame, column, lookup, interaction=None):
    q = torch.stack([lookup[str(s)] for s in frame[column]])
    if interaction is None:
        return q
    p = torch.stack([lookup[str(s)] for s in frame.prompt])
    return torch.cat([p, q, interaction(p, q)], dim=1)


def fit_controlled(frame, lookup, loss_fn, interaction=None, steps=500):
    torch.manual_seed(0)
    chosen = controlled_features(frame, 'chosen', lookup, interaction)
    rejected = controlled_features(frame, 'rejected', lookup, interaction)
    model = torch.nn.Linear(chosen.shape[1], 1, bias=False)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(steps):
        optimiser.zero_grad()
        loss = loss_fn(model(chosen).squeeze(-1), model(rejected).squeeze(-1))
        loss.backward()
        optimiser.step()
    return model


def evaluate_controlled(model, evaluation, lookup, interaction=None):
    return {name: support.credit(support.margins(model,
        controlled_features(frame, 'chosen', lookup, interaction),
        controlled_features(frame, 'rejected', lookup, interaction))).mean()
        for name, frame in evaluation.items()}
