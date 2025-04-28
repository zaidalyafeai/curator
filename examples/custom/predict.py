import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModel
from models import ModelForSequenceClassification
from datasets import load_from_disk


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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = ModelForSequenceClassification.from_pretrained(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = load_from_disk(args.dataset_name).select(range(9000))
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
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", type=str, default="arbert_classifier/final"
    )
    parser.add_argument("--dataset_name", type=str, default="annotated_dataset")
    parser.add_argument("--dataset_config", type=str, default="default")

    parser.add_argument("--output_dataset_config", type=str, default="default")
    parser.add_argument("--text_column", type=str, default="input_text")

    args = parser.parse_args()
    main(args)

