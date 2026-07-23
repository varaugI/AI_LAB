# Training reference

See the root README for complete commands.

## Choose a path

### Learn every weight yourself

Use the AI LAB transformer. This provides maximum ownership and educational control, but useful language behavior requires a large, clean corpus and meaningful compute.

### Reach useful quality sooner

Start from a licensed open causal language model and continue-pretrain it on your domain material, then perform LoRA or full supervised response tuning.

## Dataset types

### Pretraining JSONL

```json
{"text":"Complete document or section text","title":"Physics","domain":"school"}
```

### Chat SFT JSONL

```json
{"messages":[
  {"role":"system","content":"You are a careful tutor."},
  {"role":"user","content":"Explain inertia."},
  {"role":"assistant","content":"Inertia is..."}
]}
```

The assistant must be the final message.

## Resume

Set `resume_from` in a training configuration to a checkpoint directory containing `training_state.pt`.

## Multi-GPU

The custom trainer reads `WORLD_SIZE`, `RANK`, and `LOCAL_RANK`, so launch it with `torchrun`.

## Checkpoint contents

```text
config.json
model.pt
training_state.pt
tokenizer.json
```

`training_state.pt` includes optimizer, scaler, step, best validation loss, and random-number states for reliable resume.
