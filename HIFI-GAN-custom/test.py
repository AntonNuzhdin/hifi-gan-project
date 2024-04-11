import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from librosa.util import normalize

import src.model as module_model
from src.trainer import GANTrainer
from src.utils.parse_config import ConfigParser
from src.model.mel_spectrogram import MelSpectrogram, MelSpectrogramConfig
from wv_mos_metric.calculate_wv_mos import calculate_all_metrics
from wv_mos_metric.metric import MOSNet


TEST_ROOT_DIR = "LJSpeech-1.1/wavs"
SPLIT_PATH = "HIFI-GAN-custom/data/Split-LJSpeech-1.1/validation.txt"

def get_dataset_filelist(split_path):
        with open(split_path, 'r', encoding='utf-8') as fi:
            files = [x.split('|')[0] + '.wav'
                     for x in fi.read().split('\n') if len(x) > 0]
        return files

    
def main(config, output_dir, device):
    logger = config.get_logger("test")

    device = torch.device(device)

    model = config.init_obj(config["arch"], module_model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    logger.info(f"Device {device}")
    model = model.to(device)
    model.eval()
    model.generator.remove_norm()

    os.makedirs(output_dir, exist_ok=True)
    # test_dir = Path(test_dir)
    output_dir = Path(output_dir)

    sampling_rate = 22050
    mel_spec_config = MelSpectrogramConfig()
    mel_spec_transform = MelSpectrogram(mel_spec_config).to(device)
    fake_wavs = []
    with torch.no_grad():
        for wav_path in tqdm(get_dataset_filelist(SPLIT_PATH), "Generating wavs"):
            wav = torchaudio.load(os.path.join(TEST_ROOT_DIR, wav_path))[0].to(device)
            mel_spec = mel_spec_transform(wav)
            wav_pred = model.generator(mel_spec).squeeze(0).cpu()
            fake_wavs.append(wav_pred.squeeze().cpu().numpy())
            torchaudio.save(output_dir / wav_path, wav_pred, sample_rate=sampling_rate)
        
        logger.info(f"WV_MOS metric value: {calculate_all_metrics(fake_wavs, [MOSNet()])}")

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default='cuda',
        type=str,
        help="indices of GPUs to enable (default: cuda)",
    )
    # args.add_argument(
    #     "-t",
    #     "--test-dir",
    #     default="test_audio",
    #     type=str,
    #     help="Directory with test audio wav files",
    # )
    args.add_argument(
        "-o",
        "--output-dir",
        default="output",
        type=str,
        help="Output directory",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    model_config = Path(args.config)
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    main(config, args.output_dir, args.device)