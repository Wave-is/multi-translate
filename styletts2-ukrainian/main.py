import argparse
import soundfile as sf
from app import synthesize_multi, prompts_list

def save_wav_from_np(data, sample_rate, output_file):
    sf.write(output_file, data, sample_rate)

def main():
    parser = argparse.ArgumentParser(description="Synthesize speech from text using StyleTTS2 Ukrainian.")
    parser.add_argument('--text', type=str, required=True, help='Text to synthesize')
    parser.add_argument('--output', type=str, required=True, help='Output audio file path')
    parser.add_argument('--voice', type=str, choices=prompts_list, default=prompts_list[0], help='Voice prompt to use for synthesis')
    parser.add_argument('--speed', type=float, default=1.0, help='Speed of the synthesized speech')

    args = parser.parse_args()

    text = args.text
    output_file = args.output
    voice = args.voice
    speed = args.speed

    print(f"Synthesizing text: {text}")
    print(f"Using voice: {voice}")
    print(f"Output file: {output_file}")
    print(f"Speed: {speed}")

    sample_rate, audio_data = synthesize_multi(text, voice, speed)
    save_wav_from_np(audio_data, sample_rate, output_file)
    print(f"Audio saved to {output_file}")

if __name__ == "__main__":
    main()