"""
RVC Voice Changing AI Tool for Kali Linux Terminal
---------------------------------------------------

Developer: Marttin Saji
Contact: Martinsaji26@gmail.com | +971-XXXXXXXXXX

Description:
This tool converts audio using Retrieval-based Voice Conversion (RVC) 
to modify the input voice. The program is designed to work in the Kali Linux terminal.

Features:
- Supports audio file input
- Allows model selection for voice conversion
- Converts voice using RVC algorithm
- Saves the modified audio output

Usage:
    python3 rvc_tool.py --input <input_audio.wav> --output <output_audio.wav> --model <rvc_model.pth>

Dependencies:
    - Python 3.6+
    - PyTorch
    - torchaudio
    - numpy
    - soundfile
    - argparse
"""
import argparse
import torch
import torchaudio
import soundfile as sf
import numpy as np

def load_rvc_model(model_path):
    """
    Load the pre-trained RVC model.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is not available. Please run this on a system with a GPU.")
    print(f"Loading RVC model from {model_path}...")
    model = torch.load(model_path, map_location='cuda')
    model.eval()
    return model

def convert_audio(model, input_audio, sample_rate):
    """
    Apply voice conversion using the loaded model.
    """
    print("Converting audio...")
    input_tensor = torch.tensor(input_audio, dtype=torch.float32).unsqueeze(0).cuda()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_audio = output_tensor.squeeze(0).cpu().numpy()
    return output_audio

def main():
    parser = argparse.ArgumentParser(description="RVC Voice Conversion Tool")
    parser.add_argument("--input", required=True, help="Path to input audio file")
    parser.add_argument("--output", required=True, help="Path to save converted audio file")
    parser.add_argument("--model", required=True, help="Path to RVC pre-trained model")
    
    args = parser.parse_args()

    # Load input audio
    print("Loading input audio...")
    input_audio, sample_rate = torchaudio.load(args.input)
    input_audio = input_audio.squeeze(0).numpy()  # Convert to numpy array if multi-channel

    # Load RVC model
    model = load_rvc_model(args.model)

    # Convert audio
    converted_audio = convert_audio(model, input_audio, sample_rate)

    # Save the converted audio
    print(f"Saving converted audio to {args.output}...")
    sf.write(args.output, converted_audio, sample_rate)
    print("Voice conversion completed successfully!")

if __name__ == "__main__":
    main()
