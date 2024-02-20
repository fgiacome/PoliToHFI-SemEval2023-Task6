from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments
from main import LegalNERTokenDataset,compute_metrics  
import argparse

def test(model_path, test_data_path):
    # load model
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    # load the test dataset
    test_ds = LegalNERTokenDataset(test_data_path, model_path, ...)

    # set Trainer
    training_args = TrainingArguments(...)
    trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics)

    # test and output the results
    print("Testing...")
    predictions = trainer.predict(test_ds)
    metrics = compute_metrics(predictions)
    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NER model")
    parser.add_argument("--model_path", help="Path to the trained model", required=True)
    parser.add_argument("--test_data_path", help="Path to the test dataset", required=True)
    args = parser.parse_args()

    test(args.model_path, args.test_data_path)
