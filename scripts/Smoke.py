# smoke_test.py
import torchaudio
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")
wav = model.generate(
    "The quick brown fox jumps over the lazy dog.",
    audio_prompt_path="Test.wav",
)
torchaudio.save("out.wav", wav, model.sr)
print("done")