import re
import spacy
# from TTS.api import TTS
import json
import os
from transformers import WhisperProcessor, BatchFeature
import torchaudio
from torch.utils.data import Dataset, DataLoader
from gtts import gTTS
from IPython.display import Audio
import time
from torch.nn.utils.rnn import pad_sequence
import pyttsx3
import random
from peft import PeftModel
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

# Initialize WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def split_into_sentence(text: str):
    """
    Splits a given text into sentences.

    Args:
        text (str): The input text to be split into sentences.

    Returns:
        list: A list of sentences extracted from the input text.
    """
    # t1 = time.time()
    
    # Add space after punctuation if not already present
    text = re.sub(r'(?<=[.!?])(?=\S)', r' ', text)
    # Remove space between numbers in numerical ranges (e.g., "12. 5" -> "12.5")
    text = re.sub(r'(?<=\d)\. (?=\d)', r'.', text)
    # Remove "\n"
    text = re.sub(r"\n", "", text)

    # t2 = time.time()
    
    # Perform sentence segmentation
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]

    # t3 = time.time()
    # print(len(text), len(sentences), t2-t1, t3-t2)
    
    return sentences

def text2speech(sentence_ls: list, start_from: int, audio_folder_name: str, audio_text_dict: dict[str, str]):
    """
    Converts a list of sentences into speech files using gTTS.

    Args:
        sentence_ls (list): List of sentences to convert to speech.
        start_from (int): The starting index for naming audio files.

    Returns:
        None
    """
    # Create a folder named audios if not exist
    if not os.path.exists(audio_folder_name):
        os.makedirs(audio_folder_name)

    idx = start_from
    # Generate audio files for each sentence
    while idx < len(sentence_ls):
        try:
            if idx % 500 == 0:
                print("current index:", idx)
            sent = sentence_ls[idx]
            if len(sent.split()) == 1: # edge case: when the sentence only contains one single element
                # print("skip this one")
                idx += 1
                continue
            tts = gTTS(text=sent, lang='en')
            tts.save(f"{audio_folder_name}/sent_{idx}.wav")
            audio_text_dict[f"{audio_folder_name}/sent_{idx}.wav"] = sent
            idx += 1
            time.sleep(random.uniform(1, 5)) # Avoid error 429 (Too Many Requests) from TTS API
        except Exception as e:
            print(f"Error occurred at index {idx}: {e}")
            return idx
        
    return idx, audio_text_dict

def audio_text_data(audio_folder: str, transcript: list, output_file_name: str):
    """
    Prepares a JSON dataset mapping audio file paths to corresponding transcripts.

    Args:
        audio_folder (str): Folder containing audio files.
        transcript (list): List of transcripts(sentences) corresponding to the audio files.
        output_file_name (str): Name of the output JSON file.

    Returns:
        None
    """
    # Get all audio file paths from the folder
    audio_paths = [os.path.join(audio_folder, file) for file in os.listdir(
        audio_folder) if os.path.isfile(os.path.join(audio_folder, file))]
    # Sort the path by the order of the sentence
    sorted_audio_paths = sorted(audio_paths, key=lambda x: int(x.split('/')[1].split('.')[0].split('_')[1]))
    # Create the dataset structure
    data = [{"audio_path": path, "transcript": trans}
            for path, trans in zip(sorted_audio_paths, transcript)]

    # Save the dataset to a JSON file
    output_path = f"{output_file_name}.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print("finished")


class AudioTextDataset(Dataset):
    def __init__(self, json_path, processor=processor, sampling_rate=16000):
        """
        Initializes the dataset with the path to the JSON file and processor.

        Args:
            json_path (str): Path to the JSON file containing audio-transcript pairs.
            processor: Processor object for tokenizing text and preparing audio features.
            sampling_rate (int): Sampling rate for audio data.
        """
        with open(json_path, "r", encoding="utf-8") as file:
            self.data = json.load(file)
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an item by index and processes it.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing input features and labels for the model.
        """
        item = self.data[idx]
        audio_path = item["audio_path"]
        transcript = item["transcript"]

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sampling_rate)(waveform)
        waveform = waveform.squeeze(0).numpy()

        # Process inputs and labels
        inputs = self.processor(waveform, sampling_rate=self.sampling_rate, return_tensors="pt")
        labels = self.processor.tokenizer(transcript, return_tensors="pt").input_ids

        return {
            "input_features": inputs.input_features.squeeze(0),
            "labels": labels.squeeze(0),
            "transcript": transcript
        }


def collate_fn(batch):
    features = [item['input_features'] for item in batch]
    labels = [item['labels'] for item in batch]

    features_padded = pad_sequence(features, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_features": features_padded,
        "labels": labels_padded
    }


class WhisperPeftModel(PeftModel):
    def __init__(self, base_model, peft_config):
        super().__init__(base_model, peft_config)

    def forward(self, input_features=None, labels=None, **kwargs):
        # Remove `input_ids` from kwargs if passed
        kwargs.pop("input_ids", None)

        # Explicitly pass `input_features` to the base model
        return self.base_model(input_features=input_features, labels=labels, **kwargs)



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
            and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item()
        ):
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

        

