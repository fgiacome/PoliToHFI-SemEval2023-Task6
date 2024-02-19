import datasets
from utils.dataset import LegalNERTokenDataset, INDIAN_LABELS
from utils.german_dataset import get_german_dataset
from utils import conversion


def get_combined_dataset(
    path_to_indian: str,
    indian_split: str,
    german_split: str,
    german_subsampling: float,
    seed: int,
):
    indian = LegalNERTokenDataset(
        path_to_indian,
        "roberta-base",
        labels_list=INDIAN_LABELS,
        split=indian_split,
        use_roberta=True,
    )
    # let's work with huggingface datasets
    indian = datasets.Dataset.from_list(indian)
    german = get_german_dataset(german_split)
    indian = indian.map(
        lambda s: {"labels": [conversion.INDIAN_TO_COMMON[i] for i in s["labels"]]}
    )
    german = german.map(
        lambda s: {"labels": [conversion.GERMAN_TO_COMMON[i] for i in s["labels"]]}
    )
    prob_indian = 1 / (german_subsampling + 1)
    prob_german = 1 - prob_indian
    combined: datasets.Dataset = datasets.interleave_datasets(
        [german, indian], probabilities=[prob_german, prob_indian], seed=seed
    )
    combined = combined.flatten_indices()
    return combined
