import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

class ASR:
    def __init__(self, model_path="ibm-granite/granite-4.0-1b-speech") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        )

    def _get_chat(self, prmt):
        return [{"role": "user", "content": prmt},]

    def transcribe_text(self, audio_file_path):
        chat = self._get_chat("<|audio|>transcribe the speech")
        return self._helper_fun(chat, audio_file_path)

    def translate_to_english(self, audio_file_path):
        chat = self._get_chat("<|audio|>Translate to English")
        return self._helper_fun(chat, audio_file_path)

    def _helper_fun(self, chat, audio_file_path):
        wav, sr = torchaudio.load(audio_file_path)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav_16k = resampler(wav)
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        model_inputs = self.processor(prompt, wav_16k, device=self.device, return_tensors="pt").to(self.device)
        model_outputs = self.model.generate(
            **model_inputs, max_new_tokens=200, do_sample=False, num_beams=1
        )

        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
        output_text = self.tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )
        return output_text[0]