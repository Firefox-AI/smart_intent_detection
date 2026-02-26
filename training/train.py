import os
import json
import torch
import argparse
import evaluate
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ----------------------------
# Config
# ----------------------------
LABEL2ID = {"Chat": 0, "Search": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
LABEL_COL = "Label"
TEXT_COL = "Query"

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


# ----------------------------
# Metrics
# ----------------------------
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision_metric.compute(predictions=preds, references=labels, average="binary", pos_label=LABEL2ID["Search"])["precision"],
        "recall": recall_metric.compute(predictions=preds, references=labels, average="binary", pos_label=LABEL2ID["Search"])["recall"],
        "f1": f1_metric.compute(predictions=preds, references=labels, average="binary", pos_label=LABEL2ID["Search"])["f1"],
    }


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-train", dest="data_train", default="./data/training_data.csv", help="Training data in a CSV file")
    parser.add_argument("--data-eval", dest="data_eval", default="./data/evaluation_data.csv", help="Evaluation data in a CSV file")
    parser.add_argument("--base-model", dest="model", default="csarron/mobilebert-uncased-squad-v2", help="Base model")
    parser.add_argument("--model-type", dest="model_type", default="mobilebert", help="Model type")
    parser.add_argument("--output-dir", dest="output_dir", default="./results", help="Output directory.")
    parser.add_argument("--epoch", dest="epoch", type=int, default=2, help="Training epochs")
    parser.add_argument("--freeze", dest="freeze", type=int, default=0, help="Freeze non-classifier layers for n epochs")
    parser.add_argument("--metric", dest="metric", default="eval_precision", help="loss or precision, etc.")
    parser.add_argument("--lr", dest="lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--min-lr", dest="min_lr", type=float, default=9.8e-6)
    parser.add_argument("--seed", dest="seed", type=int, default=89712, help="random seed for training")
    args = parser.parse_args()
    return args


def setup_model(args):
    global TOKENIZER, MODEL

    TOKENIZER = AutoTokenizer.from_pretrained(args.model)
    MODEL = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    print(MODEL.config)
    MODEL.to(DEVICE)


def load_data(args):
    ds = load_dataset("csv", data_files={"train": args.data_train, "test": args.data_eval})
    for _, ds_split in ds.items():
        assert TEXT_COL in ds_split.features
        assert LABEL_COL in ds_split.features

    ds['train'] = upsample_hf_dataset(ds['train'], label_col="Label")
    ds['test'] = upsample_hf_dataset(ds['test'], label_col="Label")

    # preprocessing
    ds['train'] = ds['train'].map(preprocess, batched=True, remove_columns=[TEXT_COL, LABEL_COL])
    ds['test'] = ds['test'].map(preprocess, batched=True, remove_columns=[TEXT_COL, LABEL_COL])

    return ds


def upsample_hf_dataset(ds, label_col="Label", seed=42):
    y = np.array(ds[label_col])
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    rng = np.random.default_rng(seed)
    all_indices = []

    for c, cnt in zip(classes, counts):
        idx_c = np.where(y == c)[0]
        # Repeat whole sets
        reps = max_count // cnt
        rem  = max_count % cnt
        tiled = np.tile(idx_c, reps)
        extra = rng.choice(idx_c, size=rem, replace=True) if rem > 0 else np.array([], dtype=int)
        up_idx = np.concatenate([tiled, extra])
        all_indices.append(up_idx)

    upsampled_indices = np.concatenate(all_indices)
    upsampled_ds = ds.select(upsampled_indices).shuffle(seed=seed)
    return upsampled_ds


def normalize_label(label_str: str) -> int:
    key = str(label_str).strip().lower()
    if key == "chat":
        return LABEL2ID["Chat"]
    if key == "search":
        return LABEL2ID["Search"]
    raise ValueError(f"Unexpected label value: {label_str}")


def preprocess(batch):
    # Tokenize text
    enc = TOKENIZER(batch[TEXT_COL], truncation=True)
    # Map string labels to ids as "labels"
    enc["labels"] = [normalize_label(x) for x in batch[LABEL_COL]]
    return enc


def freeze_layers(model):
    keep_prefixes = ("classifier.",)  # add "pre_classifier." if desired
    for name, p in model.named_parameters():
        p.requires_grad = any(name.startswith(pref) for pref in keep_prefixes)


def run_training_process(args, data):
    output_dir = os.path.join(args.output_dir, "checkpoints")
    if args.min_lr >= args.lr:
        min_lr = 0.1 * args.lr
    else:
        min_lr = args.min_lr

    if args.freeze > 0:
        num_epochs = args.epoch - args.freeze
    else:
        num_epochs = args.epoch

    # Train with Frozen non-classifier layers
    if args.freeze > 0:
        freeze_layers(MODEL)

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            logging_steps=500,
            eval_steps=500,
            lr_scheduler_type="cosine_with_min_lr",
            lr_scheduler_kwargs={"min_lr": min_lr},
            learning_rate=args.lr,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=1,
            num_train_epochs=args.freeze,
            weight_decay=0.0,
            warmup_ratio=0.1,
            load_best_model_at_end=True,
            save_total_limit=10,
            metric_for_best_model=args.metric,
            seed=args.seed,
            fp16=True,
            #report_to="wandb",
        )

        trainer = Trainer(
            model=MODEL,
            args=training_args,
            train_dataset=data['train'],
            eval_dataset=data['test'],
            tokenizer=TOKENIZER,
            data_collator=DataCollatorWithPadding(tokenizer=TOKENIZER),
            compute_metrics=compute_metrics,
        )
        trainer.train()

        # Unfreeze
        for p in MODEL.parameters():
            p.requires_grad = True

    # Train all
    if num_epochs > 0:
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            logging_steps=500,
            eval_steps=500,
            lr_scheduler_type="cosine_with_min_lr",
            lr_scheduler_kwargs={"min_lr": min_lr},
            learning_rate=args.lr,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=1,
            num_train_epochs=num_epochs,
            weight_decay=0.0,
            warmup_ratio=0.0,
            load_best_model_at_end=True,
            save_total_limit=10,
            metric_for_best_model=args.metric,
            seed=args.seed,
            fp16=True,
            #report_to="wandb",
        )

        trainer = Trainer(
            model=MODEL,
            args=training_args,
            train_dataset=data['train'],
            eval_dataset=data['test'],
            tokenizer=TOKENIZER,
            data_collator=DataCollatorWithPadding(tokenizer=TOKENIZER),
            compute_metrics=compute_metrics,
            #callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()

    eval_metrics = trainer.evaluate()
    print("Validation metrics:", eval_metrics)

    # Save best model + tokenizer + label maps
    trainer.save_model(output_dir)
    TOKENIZER.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)


def run_inference_example():
    test_texts = [
        "What’s the weather in San Jose this weekend?",
        "Write a concise summary of this research paper.",
        "Best pizza nearby",
    ]
    batch = TOKENIZER(test_texts, return_tensors="pt", truncation=True, padding=True)
    MODEL.eval()
    with np.errstate(over="ignore"):
        with torch.no_grad():
            out = MODEL(**{k: v.to(DEVICE) for k, v in batch.items() if k in ("input_ids", "attention_mask")})
        preds = out.logits.argmax(dim=-1).cpu().numpy().tolist()

    print("Predictions:")
    print([ID2LABEL[p] for p in preds])


def main():
    args = _get_args()
    setup_model(args)
    data = load_data(args)
    run_training_process(args, data)
    run_inference_example()

if __name__ == "__main__":
    main()
