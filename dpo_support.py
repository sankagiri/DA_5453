"""Supplied DPO runtime. The student functions are used for all scoring and losses.

Training mechanics are preserved from the verified version_3 small-run notebook.
Loading provided/saved results does not load model weights or start training.
"""
from pathlib import Path
from time import perf_counter
from contextlib import contextmanager
import gc, hashlib, json, math, platform, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
from IPython.display import display



def load_results(folder):
    folder = Path(folder)
    if not (folder/'COMPLETED').exists():
        raise FileNotFoundError('No completed run here. Check the folder or choose provided results.')
    config = json.loads((folder/'configuration.json').read_text())
    metrics = pd.read_csv(folder/'metrics.csv')
    answers = pd.read_csv(folder/'answers.csv', keep_default_na=False)
    print('Results source:', folder.resolve())
    print('Model:', config['model'], '; beta:', config['beta'])
    return config, metrics, answers


def show_training(metrics):
    display(metrics[['epoch', 'split', 'loss', 'fraction_margins_improved']].round(4))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for split in ['train', 'validation']:
        rows = metrics[metrics.split == split]
        axes[0].plot(rows.epoch, rows.loss, marker='o', label=split)
        axes[1].plot(rows.epoch, rows.fraction_margins_improved, marker='o', label=split)
    axes[0].set(xlabel='Epoch', ylabel='DPO loss')
    axes[1].set(xlabel='Epoch', ylabel='Fraction of margins improved', ylim=(0, 1))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    plt.show()


def show_answers(answers):
    final_epoch = int(answers.epoch.max())
    for example_id in answers.example_id.drop_duplicates():
        rows = answers[answers.example_id == example_id]
        print('\nPROMPT:', rows.prompt.iloc[0])
        for epoch in [0, final_epoch]:
            row = rows[rows.epoch == epoch].iloc[0]
            print(f'Epoch {epoch} ({row.tokens} tokens; hit generation limit: {row.hit_token_limit}):\n{row.answer}\n')


def run(answer_logps, pair_loss, results_dir, *, n_train=64, epochs=2, beta=0.1,
        model_key="135M", device="auto", show_progress=True):
    dpo_answer_logps = answer_logps
    dpo_pair_loss = pair_loss
    DPO_MODE = "train"                 # "train", "saved", or "provided"
    DPO_MODEL = model_key                 # optional: "360M_sft"
    DPO_RESULTS_DIR = Path(results_dir)  # choose a NEW name for another training run
    DPO_DEVICE_CHOICE = device         # or "cpu", "cuda", "mps"
    DPO_N_TRAIN, DPO_EPOCHS, DPO_BETA = n_train, epochs, beta
    DPO_LR, DPO_SEED = 3e-6, 42
    DPO_MAX_LENGTH, DPO_NEW_TOKENS, DPO_EFFECTIVE_PAIRS = 256, 128, 2
    DPO_SAVE_WEIGHTS = False           # tables and answers are always saved
    DPO_SHOW_PROGRESS = show_progress
    DPO_CACHE = Path("/tmp/da5453-dpo-hf") if sys.platform == "darwin" else Path(".hf_cache")
    DPO_MODELS = {
        "135M": ("HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac"),
        "360M_sft": ("wassname/SmolLM2-360M-sft", "f6fba1d52ca818ff2eff00af64c3c7f5c49cb795"),
    }
    assert DPO_MODE in {"train", "saved", "provided"} and DPO_MODEL in DPO_MODELS
    assert DPO_N_TRAIN > 0 and DPO_EPOCHS > 0 and DPO_BETA > 0
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if DPO_MODE == "train":
        print(f"{DPO_N_TRAIN} pairs; {DPO_EPOCHS} epochs; "
              f"{DPO_EPOCHS*math.ceil(DPO_N_TRAIN/DPO_EFFECTIVE_PAIRS)} optimizer updates.")
        print("Each epoch also scores training/validation pairs and generates three responses.")

    def dpo_progress(items, description, **kwargs):
        return tqdm(items, desc=description, disable=not DPO_SHOW_PROGRESS, **kwargs)

    def dpo_release():
        gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def dpo_sync():
        if dpo_device.type == "mps": torch.mps.synchronize()
        if dpo_device.type == "cuda": torch.cuda.synchronize()

    def dpo_batch(pairs, extra_padding=0):
        sequences = [s for p in pairs for s in p["sequences"]]
        ids = torch.full((len(sequences), DPO_MAX_LENGTH+extra_padding), dpo_tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(ids)
        answer_mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, s in enumerate(sequences):
            n = len(s["ids"])
            ids[i, :n] = torch.tensor(s["ids"])
            attention[i, :n] = 1
            answer_mask[i, s["answer_start"]:n] = True
        return ids.to(dpo_device), attention.to(dpo_device), answer_mask.to(dpo_device)

    def dpo_policy_scores(model, pairs, extra_padding=0):
        ids, attention, mask = dpo_batch(pairs, extra_padding)
        logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
        return dpo_answer_logps(logits, ids, mask).reshape(-1, 2)

    @contextmanager
    def dpo_evaluating(model):
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad(): yield
        finally: model.train(was_training)

    def dpo_score_set(model, pairs, description):
        with dpo_evaluating(model):
            return torch.cat([dpo_policy_scores(model, [p]).cpu()
                for p in dpo_progress(pairs, description, unit="pair", leave=False)])

    def dpo_generate(model, epoch):
        rows = []
        with dpo_evaluating(model):
            for p in dpo_progress(dpo_prompts.to_dict("records"), f"Answers: epoch {epoch}", unit="prompt", leave=False):
                ids = dpo_tokenizer.apply_chat_template([{"role":"user", "content":p["prompt"]}],
                    add_generation_prompt=True, return_tensors="pt").to(dpo_device)
                out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                    max_new_tokens=DPO_NEW_TOKENS, pad_token_id=dpo_tokenizer.pad_token_id,
                    eos_token_id=dpo_tokenizer.eos_token_id, use_cache=True)
                ans = out[0, ids.shape[1]:].cpu().tolist()
                rows.append(dict(epoch=epoch, example_id=p["example_id"], prompt=p["prompt"],
                    answer=dpo_tokenizer.decode(ans, skip_special_tokens=True), tokens=len(ans),
                    hit_token_limit=len(ans)==DPO_NEW_TOKENS and ans[-1]!=dpo_tokenizer.eos_token_id))
        return pd.DataFrame(rows)

    if DPO_MODE == "train":
        if (DPO_RESULTS_DIR/"COMPLETED").exists():
            print(f"A completed run already exists in {DPO_RESULTS_DIR}; not training again. "
                  "Delete that folder if you want to retrain.")
            return DPO_RESULTS_DIR
        if DPO_RESULTS_DIR.exists():
            raise FileExistsError(f"{DPO_RESULTS_DIR} exists but holds no completed run (an earlier attempt was "
                                  "interrupted). Delete the folder and run this cell again.")
        dpo_manifest = json.loads(Path("data/dpo/manifest.json").read_text())
        for filename, expected in dpo_manifest["sha256"].items():
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
        dpo_name, dpo_revision = DPO_MODELS[DPO_MODEL]
        dpo_device = torch.device(DPO_DEVICE_CHOICE if DPO_DEVICE_CHOICE != "auto" else
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        if dpo_device.type == "mps": torch.mps.set_per_process_memory_fraction(.75)
        dpo_preparation_start = perf_counter()
        print("Device:", dpo_device, "— obtaining the pinned model files (first run needs internet).")
        dpo_snapshot = Path(snapshot_download(dpo_name, revision=dpo_revision, cache_dir=DPO_CACHE,
            allow_patterns=["config.json", "generation_config.json", "model.safetensors", "tokenizer_config.json",
                            "special_tokens_map.json", "vocab.json", "merges.txt", "chat_template.jinja"]))
        dpo_tokenizer = AutoTokenizer.from_pretrained(dpo_snapshot, use_fast=False, local_files_only=True)
        if (dpo_snapshot/"chat_template.jinja").exists():
            dpo_tokenizer.chat_template = (dpo_snapshot/"chat_template.jinja").read_text()
        assert dpo_tokenizer.chat_template
        if dpo_tokenizer.pad_token_id is None: dpo_tokenizer.pad_token = dpo_tokenizer.eos_token

        def dpo_encode(prompt, answer):
            messages = [{"role":"user", "content":prompt}]
            context = dpo_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full = dpo_tokenizer.apply_chat_template(messages+[{"role":"assistant", "content":answer}], tokenize=False).rstrip()
            context_ids = dpo_tokenizer(context, add_special_tokens=False)["input_ids"]
            ids = dpo_tokenizer(full, add_special_tokens=False)["input_ids"]
            assert ids[:len(context_ids)] == context_ids and ids[-1] == dpo_tokenizer.eos_token_id
            return dict(ids=ids, answer_start=len(context_ids))

        def dpo_prepare_csv(filename):
            frame = pd.read_csv(filename, keep_default_na=False)
            assert frame.example_id.is_unique
            result = {}
            for row in frame.to_dict("records"):
                seqs = [dpo_encode(row["prompt"], row[k]) for k in ["chosen", "rejected"]]
                if max(len(s["ids"]) for s in seqs) <= DPO_MAX_LENGTH:
                    result[row["example_id"]] = {**row, "sequences":seqs}
            return result

        dpo_training_pool = dpo_prepare_csv("data/pilot_train.csv")
        dpo_validation_pool = dpo_prepare_csv("data/pilot_validation.csv")
        dpo_first = dpo_manifest["train_first64"]
        assert set(dpo_first) <= dpo_training_pool.keys()
        dpo_order = dpo_first + [k for k in dpo_training_pool if k not in set(dpo_first)]
        assert DPO_N_TRAIN <= len(dpo_order), "Not enough complete pairs fit the token limit."
        dpo_train = [dpo_training_pool[k] for k in dpo_order[:DPO_N_TRAIN]]
        dpo_val = [dpo_validation_pool[k] for k in dpo_manifest["validation_ids"]]
        dpo_prompts = pd.read_csv("data/dpo/prompts.csv", keep_default_na=False)
        dpo_norm = lambda x: " ".join(x.lower().split())
        assert {dpo_norm(p["prompt"]) for p in dpo_train}.isdisjoint(dpo_norm(p["prompt"]) for p in dpo_val)
        assert set(dpo_prompts.prompt.map(dpo_norm)).isdisjoint(dpo_norm(p["prompt"]) for p in dpo_training_pool.values())
        DPO_RESULTS_DIR.mkdir(parents=True)
        dpo_config = dict(model=dpo_name, revision=dpo_revision, beta=DPO_BETA, epochs=DPO_EPOCHS,
            train_pairs=len(dpo_train), validation_pairs=len(dpo_val), learning_rate=DPO_LR, seed=DPO_SEED,
            max_length=DPO_MAX_LENGTH, max_new_tokens=DPO_NEW_TOKENS, decoding="greedy",
            effective_pairs=DPO_EFFECTIVE_PAIRS, microbatch_pairs=1, gradient_checkpointing=True,
            device=str(dpo_device), torch=torch.__version__, python=sys.version, save_weights=DPO_SAVE_WEIGHTS)
        (DPO_RESULTS_DIR/"configuration.json").write_text(json.dumps(dpo_config, indent=2))
        (DPO_RESULTS_DIR/"selected_ids.json").write_text(json.dumps(dict(
            train=[p["example_id"] for p in dpo_train], validation=[p["example_id"] for p in dpo_val]), indent=2))
        (DPO_RESULTS_DIR/"data_provenance.json").write_text(json.dumps(dpo_manifest, indent=2))
        print("Prepared:", len(dpo_train), "training pairs and", len(dpo_val), "validation pairs.")
    else:
        print("Reading mode: no model files or training required.")

    def dpo_run_training():
        # Refuse repeated execution even if only this training cell is rerun.
        with (DPO_RESULTS_DIR/"TRAINING_STARTED").open("x") as marker:
            marker.write("Choose a new folder for another run; use saved mode for analysis.\n")
        model = optimiser = scores = loss = None
        history, trace, answer_frames = [], [], []
        start = perf_counter()
        max_driver_gib = 0.
        try:
            torch.manual_seed(DPO_SEED)
            model = AutoModelForCausalLM.from_pretrained(dpo_snapshot, local_files_only=True,
                torch_dtype=torch.float32, attn_implementation="eager", use_safetensors=True).to(dpo_device)
            model.config.use_cache = False
            assert model.config.attention_dropout == 0
            assert all(m.p==0 for m in model.modules() if isinstance(m, torch.nn.Dropout))
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            model.train()  # checkpointing requires training mode; this model has zero dropout
            with dpo_evaluating(model):
                check = dpo_policy_scores(model, dpo_train[:1]).cpu()
                together = dpo_policy_scores(model, dpo_train[:2]).cpu()[:1]
                padded = dpo_policy_scores(model, dpo_train[:1], extra_padding=4).cpu()
            assert torch.allclose(check,together,atol=1e-3,rtol=1e-5)
            assert torch.allclose(check,padded,atol=1e-3,rtol=1e-5)
            ids,att,mask = dpo_batch(dpo_train[:1])
            for i,s in enumerate(dpo_train[0]["sequences"]):
                assert mask[i,len(s["ids"])-1] and not mask[i,len(s["ids"]):].any()
                assert not mask[i,:s["answer_start"]].any()
            del ids,att,mask,check,together,padded
            dpo_release()
            ref_train = dpo_score_set(model,dpo_train,"Reference: training")
            ref_val = dpo_score_set(model,dpo_val,"Reference: validation")
            frozen_train,frozen_val = ref_train.clone(),ref_val.clone()
            np.savez(DPO_RESULTS_DIR/"reference_scores.npz", train=ref_train.numpy(),validation=ref_val.numpy())

            def record(epoch,step):
                for split,pairs,reference in [("train",dpo_train,ref_train),("validation",dpo_val,ref_val)]:
                    current = reference if epoch==0 else dpo_score_set(model,pairs,f"Epoch {epoch}: {split}")
                    changes = current-reference
                    margin = changes[:,0]-changes[:,1]
                    history.append(dict(epoch=epoch,step=step,split=split,
                        loss=dpo_pair_loss(current,reference,DPO_BETA).item(),
                        fraction_margins_improved=((margin>0).float()+.5*(margin==0).float()).mean().item(),
                        chosen_logp_change=changes[:,0].mean().item(),rejected_logp_change=changes[:,1].mean().item()))
                    if epoch == DPO_EPOCHS:
                        np.save(DPO_RESULTS_DIR/f"final_{split}_scores.npy",current.numpy())
                pd.DataFrame(history).to_csv(DPO_RESULTS_DIR/"metrics.csv",index=False)
                answer_frames.append(dpo_generate(model,epoch))
                pd.concat(answer_frames,ignore_index=True).to_csv(DPO_RESULTS_DIR/"answers.csv",index=False)
                dpo_release()

            record(0,0)
            optimiser = torch.optim.AdamW(model.parameters(),lr=DPO_LR,weight_decay=0.,foreach=False)
            rng = torch.Generator().manual_seed(DPO_SEED)
            step=0
            for epoch in range(1,DPO_EPOCHS+1):
                order=torch.randperm(len(dpo_train),generator=rng).tolist()
                updates=[order[i:i+DPO_EFFECTIVE_PAIRS] for i in range(0,len(order),DPO_EFFECTIVE_PAIRS)]
                for batch_indices in dpo_progress(updates,f"Training epoch {epoch}/{DPO_EPOCHS}",unit="update"):
                    dpo_sync();begin=perf_counter();optimiser.zero_grad(set_to_none=True)
                    batch_loss=0.
                    for i in batch_indices:
                        scores=dpo_policy_scores(model,[dpo_train[i]])
                        loss=dpo_pair_loss(scores,ref_train[i:i+1].to(dpo_device),DPO_BETA)
                        assert torch.isfinite(loss)
                        batch_loss+=loss.item()/len(batch_indices)
                        (loss/len(batch_indices)).backward()
                        scores=loss=None
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    assert torch.isfinite(norm)
                    optimiser.step();dpo_sync();step+=1
                    driver=torch.mps.driver_allocated_memory()/2**30 if dpo_device.type=="mps" else 0.
                    max_driver_gib=max(max_driver_gib,driver)
                    trace.append(dict(step=step,epoch=epoch,batch_loss=batch_loss,seconds=perf_counter()-begin,
                                      sampled_mps_driver_gib=driver if dpo_device.type=="mps" else None))
                    if driver>8.2: raise RuntimeError("Apple GPU memory guard reached. Run on a machine with more memory, or use the supplied run in data/dpo/reference_135m; do not raise the limit.")
                optimiser.zero_grad(set_to_none=True)
                pd.DataFrame(trace).to_csv(DPO_RESULTS_DIR/"training_trace.csv",index=False)
                record(epoch,step)
                print(f"Epoch {epoch}: validation loss {history[-1]['loss']:.4f}; "
                      f"fraction of margins improved {history[-1]['fraction_margins_improved']:.1%}")
            assert torch.equal(ref_train,frozen_train) and torch.equal(ref_val,frozen_val)
            assert not np.array_equal(np.load(DPO_RESULTS_DIR/"final_train_scores.npy"),ref_train.numpy())
            optimiser=None;dpo_release()
            if DPO_SAVE_WEIGHTS:
                model.save_pretrained(DPO_RESULTS_DIR/"final_checkpoint")
                dpo_tokenizer.save_pretrained(DPO_RESULTS_DIR/"final_checkpoint")
            summary=dict(total_seconds=perf_counter()-dpo_preparation_start,training_cell_seconds=perf_counter()-start,
                         updates=step,sampled_mps_driver_gib=max_driver_gib if dpo_device.type=="mps" else None)
            (DPO_RESULTS_DIR/"summary.json").write_text(json.dumps(summary,indent=2))
            (DPO_RESULTS_DIR/"COMPLETED").write_text("Training, measurements and answers saved.\n")
            print("Saved results:",DPO_RESULTS_DIR.resolve(),"; elapsed minutes:",round(summary['total_seconds']/60,1))
        except BaseException as exc:
            if trace: pd.DataFrame(trace).to_csv(DPO_RESULTS_DIR/"training_trace.csv",index=False)
            (DPO_RESULTS_DIR/"FAILED.txt").write_text(type(exc).__name__+": "+str(exc))
            raise
        finally:
            model=optimiser=scores=loss=None
            dpo_release()

    if DPO_MODE == "train": dpo_run_training()
    else: print("Training skipped; reading results below.")

    return DPO_RESULTS_DIR


def staff_comparison():
    dpo_staff=pd.read_csv("data/dpo/staff_metrics.csv")
    dpo_staff_config=json.loads(Path("data/dpo/staff_configuration.json").read_text())
    dpo_staff_names=[r['name'] for r in dpo_staff_config['runs']]
    dpo_fig,dpo_axes=plt.subplots(1,2,figsize=(11,3.5))
    for dpo_name in dpo_staff_names:
        for dpo_split,dpo_ax in [("train_subset",dpo_axes[0]),("validation",dpo_axes[1])]:
            dpo_rows=dpo_staff[(dpo_staff.run==dpo_name)&(dpo_staff.split==dpo_split)]
            dpo_ax.plot(dpo_rows.epoch,dpo_rows.common_dpo_loss,marker="o",label=dpo_name)
    for dpo_ax,dpo_title in zip(dpo_axes,["Same 64 training pairs","Same 125 validation pairs"]):
        dpo_ax.set(title=dpo_title,xlabel="Epoch",ylabel="DPO loss evaluated at beta=0.1")
        dpo_ax.legend(fontsize=8);dpo_ax.grid(alpha=.2)
    dpo_fig.tight_layout();plt.show()
    dpo_fig,dpo_axes=plt.subplots(1,3,figsize=(14,3.5))
    for dpo_name,dpo_ax in zip(dpo_staff_names,dpo_axes):
        dpo_run=dpo_staff[dpo_staff.run==dpo_name]
        for dpo_split in ["train_subset","validation"]:
            dpo_rows=dpo_run[dpo_run.split==dpo_split]
            dpo_ax.plot(dpo_rows.epoch,dpo_rows.native_dpo_loss,marker="o",label=dpo_split)
        dpo_ax.set(title=f"{dpo_name}: beta={dpo_run.beta.iloc[0]}",xlabel="Epoch",ylabel="Native DPO loss")
        dpo_ax.legend(fontsize=8);dpo_ax.grid(alpha=.2)
    dpo_fig.tight_layout();plt.show()
    dpo_staff_summary=[]
    for dpo_name in dpo_staff_names:
        dpo_v=dpo_staff[(dpo_staff.run==dpo_name)&(dpo_staff.split=="validation")]
        dpo_best=dpo_v.loc[dpo_v.common_dpo_loss.idxmin()]
        dpo_staff_summary.append(dict(run=dpo_name,best_epoch=int(dpo_best.epoch),
            best_common_loss=dpo_best.common_dpo_loss,final_common_loss=dpo_v.iloc[-1].common_dpo_loss))
    display(pd.DataFrame(dpo_staff_summary).round(4))
    dpo_staff_answers=pd.read_csv("data/dpo/staff_answers.csv",keep_default_na=False)
    for dpo_row in dpo_staff_answers[dpo_staff_answers.example_id=="staff-probe-08"].itertuples():
        print(dpo_row.run,"\n",dpo_row.prompt,"\n",dpo_row.answer,"\n")


def check_response_scores(dpo_answer_logps):
    # Two scored response tokens; the last sequence position is padding.
    dpo_toy_ids = torch.tensor([[0, 1, 2, 3, 0]])
    dpo_toy_mask = torch.tensor([[False, False, True, True, False]])
    dpo_toy_logits = torch.zeros(1, 5, 4)
    assert torch.allclose(dpo_answer_logps(dpo_toy_logits, dpo_toy_ids, dpo_toy_mask),
                          torch.tensor([-2*np.log(4)], dtype=torch.float32))
    # Change predictions that should not contribute to the answer score.
    dpo_toy_logits[:, [0, 3, 4], :] = torch.tensor([8., -3., 1., 0.])
    assert torch.allclose(dpo_answer_logps(dpo_toy_logits, dpo_toy_ids, dpo_toy_mask),
                          torch.tensor([-2*np.log(4)], dtype=torch.float32))
    # Nonuniform predictions at the two scored positions: P(token 2)=0.5, P(token 3)=0.4.
    dpo_toy_logits[:, 1, :] = torch.log(torch.tensor([.1, .2, .5, .2]))
    dpo_toy_logits[:, 2, :] = torch.log(torch.tensor([.1, .2, .3, .4]))
    assert torch.allclose(dpo_answer_logps(dpo_toy_logits, dpo_toy_ids, dpo_toy_mask),
                          torch.tensor([np.log(.5)+np.log(.4)], dtype=torch.float32))
    print("Q17 checks passed: response-token scoring and masking.")


def check_loss(dpo_pair_loss):
    for dpo_test_beta in [.1, .5]:
        dpo_ref = torch.tensor([[-3., -5.], [-7., -4.]])
        dpo_scores = dpo_ref.clone().requires_grad_()
        dpo_initial = dpo_pair_loss(dpo_scores, dpo_ref, dpo_test_beta)
        assert dpo_initial.ndim == 0 and torch.isclose(dpo_initial, torch.tensor(np.log(2), dtype=torch.float32))
        dpo_initial.backward()
        assert (dpo_scores.grad[:, 0] < 0).all() and (dpo_scores.grad[:, 1] > 0).all()
        dpo_better = dpo_ref + torch.tensor([[1., -1.], [2., -1.]])
        assert dpo_pair_loss(dpo_better, dpo_ref, dpo_test_beta) < dpo_initial.detach()
        # Exact value: margins are 2 and 3, so the loss is the mean of -log(sigmoid(beta*2)) and -log(sigmoid(beta*3)).
        expected = -torch.log(torch.sigmoid(dpo_test_beta * torch.tensor([2., 3.]))).mean()
        assert torch.isclose(dpo_pair_loss(dpo_better, dpo_ref, dpo_test_beta), expected, atol=1e-5), "beta is not applied correctly"
        assert dpo_pair_loss(dpo_ref - (dpo_better-dpo_ref), dpo_ref, dpo_test_beta) > dpo_initial.detach()
    print("Q18 checks passed: initial loss, preference direction and score gradients.")
