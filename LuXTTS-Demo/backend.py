from zipvoice.luxvoice import LuxTTS as LuxModel
import soundfile as sf

class LuxTTSBackend:
    def __init__(self) -> None:
        self.lux_tts = LuxModel('YatharthS/LuxTTS', device='cuda')
        self.rms = 0.01 
        self.t_shift = 0.9 
        self.num_steps = 4
        self.speed = 1.0 
        self.return_smooth = False 
        self.ref_duration = 10000

    def tts(self, ref_audio_path, text, out_path):
        encoded_prompt = self.lux_tts.encode_prompt(
            ref_audio_path,
            duration=self.ref_duration,
            rms=self.rms
        )

        final_wav = self.lux_tts.generate_speech(
            text,
            encoded_prompt,
            num_steps=self.num_steps,
            t_shift=self.t_shift,
            speed=self.speed,
            return_smooth=self.return_smooth
        )

        final_wav = final_wav.numpy().squeeze()
        sf.write(out_path, final_wav, 48000)
        return out_path
