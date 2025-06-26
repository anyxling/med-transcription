from datasets import load_dataset
from data_pipeline import *
from transformers import WhisperForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, LoraModel
import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from ipdb import set_trace
from transformers import BatchFeature
import wandb

processor = WhisperProcessor.from_pretrained("openai/whisper-large")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Pad features
        input_features = [{"input_features": feat["input_features"]} for feat in features]
        batch = self.processor.feature_extractor.pad(
            input_features, 
            padding=True,  # Dynamic padding for audio
            return_tensors="pt"
        )

        # Pad labels
        labels = [{"input_ids": feat["labels"]} for feat in features]
        labels_batch = self.processor.tokenizer.pad(
            labels,
            padding=True,  # Dynamic padding for text
            return_tensors="pt"
        )
        # set_trace()

        # Replace padding tokens with -100 (ignored in the loss)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove bos token if necessary
        if (
            labels.size(1) > 1
            and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

wandb.init(
    # set the wandb project where this run will be logged
    project="whisper",
)
training_args = Seq2SeqTrainingArguments(
    output_dir="checkpoints", 
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-3,
    warmup_steps=0,
    num_train_epochs=3,
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_first_step=True,
    logging_nan_inf_filter=False,
    eval_steps=500,
    report_to=["wandb"],
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=128,
    logging_steps=1,
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],  # same reason as above
)

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large", load_in_8bit=True, device_map="auto")

model = prepare_model_for_kbit_training(model)

config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

model = get_peft_model(model, config)
model.print_trainable_parameters()

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     return {"eval_loss": trainer.evaluate()["eval_loss"]}
    
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=AudioTextDataset(json_path="train_data_clean.json", processor=processor),
    eval_dataset=AudioTextDataset(json_path="test_data_clean.json", processor=processor),
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

trainer.train(resume_from_checkpoint=True)
# trainer.train()




