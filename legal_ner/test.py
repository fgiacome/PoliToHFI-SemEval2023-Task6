from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments
from main import LegalNERTokenDataset,compute_metrics  
import argparse
from utils.dataset import INDIAN_LABELS
from utils.german_dataset import get_german_dataset
from utils import conversion
from datasets import Dataset

def test(model_path, test_data_path, dataset, label_type):
    assert dataset=='german' or (test_data_path is not None), 'Please indicate a test_data_path for the indian dataset if not using German only.'
    assert label_type in {'original', 'combined'}, 'Please specify a supported label_type'

    # load model
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    # load the test dataset
    if dataset == 'indian':
        indian_original_labels = LegalNERTokenDataset(test_data_path, model_path, INDIAN_LABELS, split='test', use_roberta=True)
        indian_original_labels = Dataset.from_list(indian_original_labels)
        if label_type == 'original':
            test_ds = indian_original_labels
        if label_type == 'combined':
            test_ds = indian_original_labels.map(
                lambda s: {"labels": [conversion.INDIAN_TO_COMMON[i] for i in s["labels"]]}
            )
    if dataset == 'german' and label_type == 'original':
        test_ds = get_german_dataset('test')
    if dataset == 'german' and label_type == 'combined':
        test_ds = get_german_dataset('test')
        test_ds = test_ds.map(
            lambda s: {"labels": [conversion.GERMAN_TO_COMMON[i] for i in s["labels"]]}
        )


    # set Trainer
    training_args = TrainingArguments(
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
    )

    # test and output the results
    print("Testing...")
    predictions = trainer.predict(test_ds)
    metrics = compute_metrics(predictions)
    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NER model")
    parser.add_argument("--model_path", help="Path to the trained model", required=True)
    parser.add_argument("--test_data_path", help="Path to the test dataset (Indian dataset only)", required=False, type=str, default=None)
    parser.add_argument("--dataset", help="'german', 'indian', 'combined'", required=True)
    parser.add_argument("--label_type", help="'original' or 'combined'", required=True)
    args = parser.parse_args()

    test(args.model_path, args.test_data_path, args.dataset, args.label_type)
