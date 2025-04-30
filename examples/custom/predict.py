import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModel
from models import ModelForSequenceClassification
from datasets import load_from_disk
from glob import glob
from sklearn.metrics import precision_score, recall_score, f1_score
from datasets import ClassLabel
import numpy as np
def get_score_distribution(dataset, column_name):
    import numpy as np

    # Get the integer scores
    scores = dataset.to_pandas()[column_name]

    # Count occurrences for each score (0 to 5)
    bins = range(0, 7)
    hist, _ = np.histogram(scores, bins=bins)

    print("\nHistogram of int_score:")
    for i, count in enumerate(hist):
        print(f"{i}: {'#' * count} ({count})")


def main(args):

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
    dataset = dataset["test"]

    results = {}
    for model_name in glob("*_classifier/final/"):
        print(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    
        # get_score_distribution(dataset, "score")

        def compute_scores(batch):
            inputs = tokenizer(
                batch[args.text_column],
                return_tensors="pt",
                padding="longest",
                truncation=True,
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)[0]
                logits = outputs.squeeze(-1).float().cpu().numpy()

            batch["predicted_score"] = [int(round(max(0, min(score, 5)))) for score in logits]
            return batch

        dataset = dataset.map(compute_scores, batched=True, batch_size=512)

        # get_score_distribution(dataset, "predicted_score")

        # calculate the accuracy
        df = dataset.to_pandas()
        accuracy = (df["predicted_score"] == df["score"]).mean()
        precision = precision_score(df["score"], df["predicted_score"], average="macro")
        recall = recall_score(df["score"], df["predicted_score"], average="macro")
        f1 = f1_score(df["score"], df["predicted_score"], average="macro")

        model_name = model_name.split("_")[0]

        # create and save the confusion matrix image
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(df["score"], df["predicted_score"])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(6), yticklabels=range(6))
        plt.savefig(f"{model_name}_confusion_matrix.png")
        plt.close()

        results[model_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    # show a table in markdown format
    from tabulate import tabulate
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    table_data = []
    for model_name, result in results.items():
        table_data.append([
            model_name,
            f"{result['accuracy']:.4f}",
            f"{result['precision']:.4f}", 
            f"{result['recall']:.4f}",
            f"{result['f1']:.4f}"
        ])
    # sort by f1 score
    table_data.sort(key=lambda x: x[4], reverse=True)
    print(tabulate(table_data, headers=headers, tablefmt="pipe"))

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="annotated_dataset")
    parser.add_argument("--dataset_config", type=str, default="default")

    parser.add_argument("--output_dataset_config", type=str, default="default")
    parser.add_argument("--text_column", type=str, default="input_text")
    parser.add_argument("--target_column", type=str, default="score")
    args = parser.parse_args()
    main(args)

