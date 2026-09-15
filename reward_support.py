"""Supplied data, training and display functions for Assignment 2.

Students need not edit this file. Training uses the loss supplied by the notebook.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display


def encode_texts(encoder, texts):
    lengths = [len(ids) for ids in encoder.tokenizer(texts, truncation=False)["input_ids"]]
    if max(lengths) > encoder.max_seq_length:
        raise ValueError("A supplied example exceeds the encoder input limit.")
    return encoder.encode(texts, batch_size=32, normalize_embeddings=True,
                          convert_to_tensor=True, show_progress_bar=False)


def encode_pairs(encoder, frame, column):
    texts = [f"Human: {x}\nAssistant: {y}" for x, y in zip(frame.prompt, frame[column])]
    return encode_texts(encoder, texts)


def show_pairs(frame, limit=3):
    for index, row in frame.head(limit).iterrows():
        print(f"\nExample {row.get('example_id', index)}")
        print("PROMPT:", row['prompt'])
        print("CHOSEN RESPONSE:", row['chosen'])
        print("REJECTED RESPONSE:", row['rejected'])


@torch.no_grad()
def margins(model, chosen, rejected):
    model.eval()
    return (model(chosen).squeeze(-1) - model(rejected).squeeze(-1)).cpu().numpy()


def credit(margin):
    return (margin > 0).astype(float) + 0.5 * (margin == 0).astype(float)


@torch.no_grad()
def evaluate(model, chosen, rejected, loss_fn):
    model.eval()
    a, b = model(chosen).squeeze(-1), model(rejected).squeeze(-1)
    return {'loss': loss_fn(a, b).item(), 'accuracy': credit((a-b).cpu().numpy()).mean()}


def train_model(factory, chosen, rejected, val_chosen, val_rejected, loss_fn,
                steps=200, learning_rate=0.01, seed=42):
    """Repeat the Part 1 procedure for a different representation or reward head."""
    torch.manual_seed(seed)
    model = factory()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history, best_loss, best_weights, best_step = [], float('inf'), None, None
    for step in range(steps + 1):
        tr = evaluate(model, chosen, rejected, loss_fn)
        va = evaluate(model, val_chosen, val_rejected, loss_fn)
        history.append(dict(step=step, train_loss=tr['loss'], train_accuracy=tr['accuracy'],
                            validation_loss=va['loss'], validation_accuracy=va['accuracy']))
        if va['loss'] < best_loss:
            best_loss, best_step = va['loss'], step
            best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if step == steps:
            break
        model.train()
        optimiser.zero_grad()
        loss_fn(model(chosen).squeeze(-1), model(rejected).squeeze(-1)).backward()
        optimiser.step()
    final_model = factory()
    final_model.load_state_dict(model.state_dict())
    model.load_state_dict(best_weights)
    return dict(best_model=model, final_model=final_model,
                best_step=best_step, history=pd.DataFrame(history))


def plot_training(run):
    history = run['history']
    display(history.iloc[sorted({0, run['best_step'], len(history)-1})].round(4))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for split, label in [('train', 'Training'), ('validation', 'Validation')]:
        for ax, metric in zip(axes, ['loss', 'accuracy']):
            ax.plot(history.step, history[f'{split}_{metric}'], label=label)
    for ax, label in zip(axes, ['Mean Bradley–Terry loss', 'Preference accuracy']):
        ax.set(xlabel='Training step', ylabel=label)
        ax.legend()
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    plt.show()


def compare_sizes(runs, chosen, rejected, loss_fn):
    rows = []
    for size, run in sorted(runs.items()):
        metrics = evaluate(run['best_model'], chosen, rejected, loss_fn)
        rows.append(dict(training_pairs=size, saved_step=run['best_step'],
                         validation_loss=metrics['loss'], validation_accuracy=metrics['accuracy']))
    table = pd.DataFrame(rows)
    display(table.round(4))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, name in zip(axes, ['validation_loss', 'validation_accuracy']):
        ax.plot(table.training_pairs, table[name], marker='o')
        ax.set(xlabel='Training pairs', ylabel=name.replace('_', ' ').capitalize())
    fig.tight_layout()
    plt.show()
    return table


def word_length_gap(frame):
    """Chosen length minus rejected length, counted in words."""
    return (frame.chosen.str.split().str.len() - frame.rejected.str.split().str.len()).to_numpy()


def show_baselines(gap, margin):
    display(pd.DataFrame({'Method': ['Random choice (expected)', 'Choose longer', 'Reward model'],
                          'Validation accuracy': [0.5, credit(gap).mean(), credit(margin).mean()]}).round(4))


def _validate_length_audit(longer_accuracy, shorter_accuracy, longer_rate):
    for value in [longer_accuracy, shorter_accuracy, longer_rate]:
        if value is None or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Complete the three Q6 calculations with values between 0 and 1.')


def check_length_audit(longer_accuracy, shorter_accuracy, longer_rate):
    _validate_length_audit(longer_accuracy, shorter_accuracy, longer_rate)
    print(f'Accuracy when the preferred response is longer: {longer_accuracy:.4f}')
    print(f'Accuracy when the preferred response is shorter: {shorter_accuracy:.4f}')
    print(f'Model chooses the longer response (unequal lengths): {longer_rate:.4f}')


def show_length_audit(gap, margin, longer_accuracy, shorter_accuracy, longer_rate):
    _validate_length_audit(longer_accuracy, shorter_accuracy, longer_rate)
    equal = gap == 0
    display(pd.DataFrame({'Recorded preferred response': ['Longer', 'Shorter', 'Equal length'],
        'Pairs': [(gap > 0).sum(), (gap < 0).sum(), equal.sum()],
        'Model accuracy': [longer_accuracy, shorter_accuracy,
                           credit(margin[equal]).mean() if equal.any() else np.nan]}).round(4))
    print(f'Model chooses longer (unequal lengths): {longer_rate:.1%}')
    print(f'Correlation of length difference and reward difference: {np.corrcoef(gap, margin)[0,1]:.3f}')
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.scatter(gap, margin, alpha=0.55, s=22)
    ax.axhline(0, color='grey', lw=1)
    ax.axvline(0, color='grey', lw=1)
    ax.set(xlabel='Chosen minus rejected length (words)', ylabel='Chosen minus rejected reward')
    fig.tight_layout()
    plt.show()


def diagnostics(model, chosen, rejected, gap, loss_fn):
    margin = margins(model, chosen, rejected)
    metrics = evaluate(model, chosen, rejected, loss_fn)
    return dict(validation_loss=metrics['loss'], overall_accuracy=metrics['accuracy'],
                preferred_longer_accuracy=credit(margin[gap > 0]).mean(),
                preferred_shorter_accuracy=credit(margin[gap < 0]).mean(),
                chooses_longer=credit(gap[gap != 0] * margin[gap != 0]).mean())


def show_disagreements(frame, margin, example_ids=None):
    """Use fixed reviewed examples when supplied; otherwise show two model disagreements."""
    rows = frame.loc[margin < 0].copy()
    rows['reward_margin'] = margin[margin < 0]
    if example_ids is not None:
        rows = rows[rows.example_id.isin(example_ids)]
    for _, row in rows.head(2).iterrows():
        print(f'\nModel reward difference (chosen minus rejected): {row.reward_margin:.3f}')
        show_pairs(pd.DataFrame([row]), limit=1)
    if rows.empty:
        print('No prediction disagreements for the selected examples with this fitted model.')


def blind_pairs(data_dir='data'):
    orientation = pd.read_csv(Path(data_dir)/'pilot_orientation.csv')
    chosen_first = np.random.default_rng(42).integers(0, 2, size=len(orientation)).astype(bool)
    blind = orientation[['example_id', 'prompt']].copy()
    blind['response_1'] = np.where(chosen_first, orientation.chosen, orientation.rejected)
    blind['response_0'] = np.where(chosen_first, orientation.rejected, orientation.chosen)
    for number, row in enumerate(blind.itertuples(), 1):
        print(f'\nExample {number}\nPROMPT: {row.prompt}\nRESPONSE 1: {row.response_1}\nRESPONSE 0: {row.response_0}')
    return blind, np.where(chosen_first, '1', '0')


def compare_judgments(choices, blind, recorded, encoder, model):
    if len(choices) != len(blind) or not set(choices) <= {'0', '1'}:
        print('Enter exactly eight digits using 1 for Response 1 and 0 for Response 0.')
        return
    margin = margins(model, encode_pairs(encoder, blind, 'response_1'), encode_pairs(encoder, blind, 'response_0'))
    model_choice = np.where(margin > 0, '1', np.where(margin < 0, '0', 'tie'))
    display(pd.DataFrame({'Example': range(1, len(blind)+1), 'Your choice': list(choices),
                          'Model choice': model_choice, 'Recorded preference': recorded}))


def audit_rule(frame, model_margin, feature_fn, prefer_higher):
    chosen = np.asarray([feature_fn(x, y) for x, y in zip(frame.prompt, frame.chosen)], dtype=float)
    rejected = np.asarray([feature_fn(x, y) for x, y in zip(frame.prompt, frame.rejected)], dtype=float)
    if not np.isfinite(chosen).all() or not np.isfinite(rejected).all():
        raise ValueError('The feature must return finite numeric scores.')
    margin = (chosen-rejected) * (1 if prefer_higher else -1)
    print(f'Validation accuracy: {credit(margin).mean():.2%}; non-tied pairs: {(margin != 0).sum()}')
    rows = frame.loc[(margin > 0) & (model_margin < 0)]
    if rows.empty:
        print('No cases where the rule agrees with the label and the model disagrees.')
    else:
        show_pairs(rows, limit=2)
    return margin
