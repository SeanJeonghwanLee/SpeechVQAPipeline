from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import time


# load model and processor
processor = WhisperProcessor.from_pretrained("jiwon65/whisper-small_korean-zeroth")
model = WhisperForConditionalGeneration.from_pretrained("jiwon65/whisper-small_korean-zeroth")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="korean", task="transcribe")


# load dummy dataset and read audio files

ds = load_dataset("audiofolder", data_dir="/home/seanlee/class/deeplearning/whisper/sample", drop_labels=True)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
print(f"ds:{ds}")
for i in range(ds['train'].num_rows):
    sample = ds['train'][i]["audio"]
    print(f"sample:{sample}")
    # print(f'sample["sampling_rate"]:{sample["sampling_rate"]}')

    st = time.time()
    input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

    # generate token ids
    predicted_ids = model.generate(input_features, forced_decoder_ids = forced_decoder_ids)
    # decode token ids to text
    # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    end = time.time()


    print(f"result ({end-st}): {transcription}")
