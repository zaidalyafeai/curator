from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoConfig,
    ModernBertForSequenceClassification
)
from datasets import load_dataset, ClassLabel, load_from_disk
import numpy as np
import evaluate
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix
from models import ModelForSequenceClassification

def compute_metrics(eval_pred):
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    preds = np.round(logits.squeeze()).clip(0, 5).astype(int)
    labels = np.round(labels.squeeze()).astype(int)
    precision = precision_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["precision"]
    recall = recall_metric.compute(
        predictions=preds, references=labels, average="macro"
    )["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]

    report = classification_report(labels, preds)
    cm = confusion_matrix(labels, preds)
    print("Validation Report:\n" + report)
    print("Confusion Matrix:\n" + str(cm))

    return {
        "precision": precision,
        "recall": recall,
        "f1_macro": f1,
        "accuracy": accuracy,
    }


def main(args):
    ckpt_dir = os.path.join(args.checkpoint_dir, "final")
    dataset = load_from_disk(args.dataset_name)
    dataset = dataset.map(
        lambda x: {args.target_column: np.clip(int(x[args.target_column]), 0, 5)},
        num_proc=16,
    )

    dataset = dataset.cast_column(
        args.target_column, ClassLabel(names=[str(i) for i in range(6)])
    )
    dataset = dataset.train_test_split(
        train_size=0.9, seed=42, stratify_by_column=args.target_column
    )
    if args.eval_only:
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt_dir,
            num_labels=1,
            classifier_dropout=0.0,
            hidden_dropout_prob=0.0,
        trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name,
            model_max_length=min(model.config.max_position_embeddings, 512),
            trust_remote_code=True,
        )
    else:
        if "modernbert" in args.base_model_name.lower():
            config = AutoConfig.from_pretrained(args.base_model_name, num_labels=1, classifier_dropout=0.0, hidden_dropout_prob=0.0)   
            model = ModernBertForSequenceClassification(
                config,
            )
        else:
            # config = AutoConfig.from_pretrained(args.base_model_name, num_labels=1, classifier_dropout=0.0, hidden_dropout_prob=0.0)   
            # model = ModelForSequenceClassification(
            #     config,
            #     base_model_name=args.base_model_name,
            # )
            model = AutoModelForSequenceClassification.from_pretrained(
                args.base_model_name,
                num_labels=1,
                classifier_dropout=0.0,
                hidden_dropout_prob=0.0,
                trust_remote_code=True,
            )
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name,
            model_max_length=min(model.config.max_position_embeddings, 512),
            trust_remote_code=True,
        )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(examples):
        batch = tokenizer(examples[args.input_column], truncation=True)
        batch["labels"] = np.float32(examples[args.target_column])
        return batch

    dataset = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    batch_size = 128
    eval_batch_size = 64
    num_train_epochs = 2
    num_steps = len(dataset["train"]) // batch_size
    eval_steps = num_steps // 2
    save_steps = num_steps // 2
    logging_steps = num_steps // 6
    learning_rate = 3e-4
    seed = 0

    
    if "modernbert" in args.base_model_name.lower():
        for param in model.bert.layers.parameters():
            param.requires_grad = False
        
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
    elif "embed-l-v2.0" in args.base_model_name.lower():
        for param in model.roberta.parameters():
            param.requires_grad = False
        
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = False
    else:
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        for param in model.bert.encoder.parameters():
            param.requires_grad = False

    # show the number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")
    print(f"Number of total parameters: {sum(p.numel() for p in model.parameters())}")



    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        hub_model_id=args.output_model_name,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        seed=seed,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        eval_on_start=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        bf16=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    if not args.eval_only:
        trainer.train()
        trainer.save_model(ckpt_dir)
    else:
        trainer.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name", type=str, default="UBC-NLP/MARBERT"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="annotated_dataset-100k",
    )
    
    parser.add_argument("--input_column", type=str, default="input_text")
    parser.add_argument("--target_column", type=str, default="score")
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()
    model_name = args.base_model_name.split("/")[-1].lower()
    args.checkpoint_dir = f"{model_name}_classifier-100k"
    args.output_model_name = f"{model_name}_classifier-100k"
    main(args)
