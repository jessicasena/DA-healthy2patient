import torch
from transformers import PegasusTokenizerFast, BigBirdPegasusForSequenceClassification, Trainer, TrainingArguments


class PegasusDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])  # torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)  # len(self.labels)


def prepare_fine_tuning(model_name, train_dataset, val_dataset=None, freeze_encoder=False, num_labels=2,
                        output_dir='/home/jsenadesouza/DA-healthy2patient/results/outcomes/classifier_results/bigbird/results'):
  """
  Prepare configurations and base model for fine-tuning
  """
  torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = BigBirdPegasusForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(torch_device)

  if freeze_encoder:
    for param in model.model.encoder.parameters():
      param.requires_grad = False

  if val_dataset is not None:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=2000,           # total number of training epochs
      per_device_train_batch_size=1,   # batch size per device during training, can increase if memory allows
      per_device_eval_batch_size=1,    # batch size for evaluation, can increase if memory allows
      save_steps=500,                  # number of updates steps before checkpoint saves
      save_total_limit=5,              # limit the total amount of checkpoints and deletes the older checkpoints
      evaluation_strategy='steps',     # evaluation strategy to adopt during training
      eval_steps=100,                  # number of update steps before evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='/home/jsenadesouza/DA-healthy2patient/results/outcomes/classifier_results/bigbird/logs',            # directory for storing logs
      logging_steps=10,
    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      eval_dataset=val_dataset,            # evaluation dataset
    )

  else:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=2000,           # total number of training epochs
      per_device_train_batch_size=1,   # batch size per device during training, can increase if memory allows
      save_steps=500,                  # number of updates steps before checkpoint saves
      save_total_limit=5,              # limit the total amount of checkpoints and deletes the older checkpoints
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='/home/jsenadesouza/DA-healthy2patient/results/outcomes/classifier_results/bigbird/logs',            # directory for storing logs
      logging_steps=10,
    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
    )

  return trainer


def prepare_data(model_name,
                 train_texts, train_labels,
                 val_texts=None, val_labels=None,
                 test_texts=None, test_labels=None):
  """
  Prepare input data for model fine-tuning
  """
  tokenizer = PegasusTokenizerFast.from_pretrained(model_name)

  prepare_val = False if val_texts is None or val_labels is None else True
  prepare_test = False if test_texts is None or test_labels is None else True

  def tokenize_data(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True)
    decodings = tokenizer(labels, truncation=True, padding=True)
    dataset_tokenized = PegasusDataset(encodings, decodings)
    return dataset_tokenized

  train_dataset = tokenize_data(train_texts, train_labels)
  val_dataset = tokenize_data(val_texts, val_labels) if prepare_val else None
  test_dataset = tokenize_data(test_texts, test_labels) if prepare_test else None

  return train_dataset, val_dataset, test_dataset, tokenizer

 #
 # import torch
 #  from transformers import PegasusTokenizerFast, BigBirdPegasusForSequenceClassification
 #
 #  tokenizer = PegasusTokenizerFast.from_pretrained("hf-internal-testing/tiny-random-bigbird_pegasus")
 #  model = BigBirdPegasusForSequenceClassification.from_pretrained("hf-internal-testing/tiny-random-bigbird_pegasus")
 #
 #  inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
 #
 #  with torch.no_grad():
 #    logits = model(**inputs).logits
 #
 #  predicted_class_id = logits.argmax().item()
 #  model.config.id2label[predicted_class_id]
 #
 #  # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
 #  num_labels = len(model.config.id2label)
 #  model = BigBirdPegasusForSequenceClassification.from_pretrained("hf-internal-testing/tiny-random-bigbird_pegasus",
 #                                                                  num_labels=num_labels)
 #
 #  labels = torch.tensor([1])
 #  loss = model(**inputs, labels=labels).loss
 #  round(loss.item(), 2)