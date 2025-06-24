import pandas as pd
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def create_demo_files_if_not_exist():
    """
    checking CSV files: creating them if they do not exist
    """
    train_file = "reviews_train.csv"
    test_file = "reviews_test.csv"

    if os.path.exists(train_file) and os.path.exists(test_file):
        print("csv files do exist")
        return

    print("creating csv files")
    train_data = {
        "text": [
            "The food was absolutely wonderful, from preparation to presentation, very pleasing.","I highly recommend this place, the staff is amazing.","A great place to have a quiet and delicious meal.","The ambiance is very charming and the service is impeccable.","Best burger I have ever had in my life!","The dessert was divine, a perfect end to a perfect meal.","We had a delightful experience, will definitely come back.","Service was quick and the food was fresh and tasty.","A hidden gem! The quality of food is outstanding.","Perfect for a date night. Cozy and romantic.","The pasta was cooked to perfection.","Their brunch menu is fantastic, so many options.","I loved the creative cocktails and the friendly bartender.","The steak was juicy and flavorful.","An unforgettable dining experience.","The portions were generous and the prices were reasonable.","Everything we ordered was delicious.","The decor is modern and stylish.","Staff went above and beyond to make our evening special.","The seafood is incredibly fresh.","A must-visit for any food lover.","The coffee is the best in town.","I appreciate the attention to detail in every dish.","The tasting menu is a journey of flavors.","Exceptional quality and taste.",
            "The food was cold and tasteless.","Service was incredibly slow, we waited 45 minutes for our drinks.","The waiter was rude and seemed annoyed by our questions.","Overpriced for the quality of food you get.","I found a hair in my soup, it was disgusting.","The place was dirty and the tables were sticky.","I would not recommend this restaurant to anyone.","The music was too loud, we couldn't hear each other talk.","My order was wrong, and they didn't even apologize.","A complete disappointment from start to finish.","The chicken was dry and overcooked.","The portions are tiny and not worth the money.","This place is overhyped. It did not live up to expectations.","The bathroom was filthy.","We felt rushed by the staff to finish our meal and leave.","The menu is very limited and uninspired.","I've had better food from a frozen dinner.","The atmosphere is dull and depressing.","They charge for water, which is ridiculous.","A tourist trap with mediocre food.","The restaurant smelled weird.","My steak was tough as leather.","The vegetables were soggy and bland.","Avoid this place at all costs.","It was a waste of time and money.",
        ] * 2,
        "label": ([1] * 25 + [0] * 25) * 2,
    }
    train_df = pd.DataFrame(train_data)
    train_df.to_csv(train_file, index=False)
    print(f"file '{train_file}' is created (100 records)")
    test_data = {
        "text": [
            "The pizza was fantastic, some of the best I've had.","A wonderful experience, the staff made us feel so welcome.","Truly authentic flavors, it felt like I was in Italy.","The view from the terrace is breathtaking.","I will be dreaming about that cheesecake for weeks.",
            "The restaurant was freezing cold and uncomfortable.","My fish was undercooked and inedible.","The manager was unprofessional and handled our complaint poorly.","Extremely disappointing, I expected much more.","The location is great, but the food is terrible.",
        ],
        "label": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    }
    test_df = pd.DataFrame(test_data)
    test_df.to_csv(test_file, index=False)
    print(f"file '{test_file}' is created (10 records)")

def load_and_prepare_data(train_file, test_file):
    dataset = load_dataset("csv", data_files={"train": train_file, "test": test_file})
    print("\ndataset loaded:")
    print(dataset)
    return dataset

def tokenize_data(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print("\nTOKENIZED")
    return tokenized_datasets

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary")
    return {"accuracy": acc, "f1": f1} # extra

if __name__ == "__main__":
    create_demo_files_if_not_exist()

    MODEL_NAME = "distilbert-base-uncased"
    TRAIN_FILE = "reviews_train.csv"
    TEST_FILE = "reviews_test.csv"

    raw_datasets = load_and_prepare_data(TRAIN_FILE, TEST_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_datasets = tokenize_data(raw_datasets, tokenizer)

    print("\nstarting fine-tuning")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("\ntraining finished.")