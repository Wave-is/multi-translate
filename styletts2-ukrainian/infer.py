import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)
import librosa
from copy import deepcopy
from huggingface_hub import hf_hub_download

import spaces
import yaml
import re
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from ipa_uk import ipa
from unicodedata import normalize
from ukrainian_word_stress import Stressifier, StressSymbol
stressify = Stressifier()



from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask


def load_state_dict(model, params):
    for key in model:
        if key in params:
            print('%s loaded' % key)
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v

                model[key].load_state_dict(new_state_dict, strict=False)

 
config = yaml.safe_load(open('config.yml')) 
config_path = os.path.join(os.path.dirname(__file__), 'config.yml')

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
ASR_path = os.path.join(os.path.dirname(__file__), ASR_path)
ASR_config = os.path.join(os.path.dirname(__file__), ASR_config)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
F0_path = os.path.join(os.path.dirname(__file__), F0_path)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert

plbert = load_plbert(os.path.join(os.path.dirname(__file__), 'weights/plbert.bin'), os.path.join(os.path.dirname(__file__), 'Utils/PLBERT/config.yml'))

model_single = build_model(recursive_munch(config['model_params_single']), text_aligner, pitch_extractor, plbert)
model_multi = build_model(recursive_munch(config['model_params_multi']), deepcopy(text_aligner), deepcopy(pitch_extractor), deepcopy(plbert))


multi_path = hf_hub_download(repo_id='patriotyk/styletts2_ukrainian_multispeaker', filename="pytorch_model.bin")
params_multi = torch.load(multi_path, map_location='cpu')


single_path = hf_hub_download(repo_id='patriotyk/styletts2_ukrainian_single', filename="pytorch_model.bin")
params_single = torch.load(single_path, map_location='cpu')


load_state_dict(model_single, params_single)
_ = [model_single[key].eval() for key in model_single]
_ = [model_single[key].to(device) for key in model_single]


load_state_dict(model_multi, params_multi)
_ = [model_multi[key].eval() for key in model_multi]
_ = [model_multi[key].to(device) for key in model_multi]



models = {
    'multi': model_multi,
    'single': model_single
}



def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(voice_audio):
    wave, sr = librosa.load(voice_audio, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = models['multi'].style_encoder(mel_tensor.unsqueeze(1))
        ref_p = models['multi'].predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)


def split_to_parts(text):
    split_symbols = '.?!:'
    parts = ['']
    index = 0
    for s in text:
        parts[index] += s
        if s in split_symbols and len(parts[index]) > 150:
            index += 1
            parts.append('')
    return parts
    


def _inf(model, text, ref_s, speed, s_prev, noise, alpha, beta, diffusion_steps, embedding_scale):
    model = models[model]
    text = text.strip()
    text = text.replace('"', '')
    text = text.replace('+', 'ˈ')
    text = normalize('NFKC', text)

    text = re.sub(r'[᠆‐‑‒–—―⁻₋−⸺⸻]', '-', text)
    text = re.sub(r' - ', ': ', text)
    ps = ipa(stressify(text))
    print(ps)

    tokens = textclenaer(ps)
    tokens.insert(0, 0)

    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
        
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 

        
        if ref_s is None:
            s_pred = model.sampler(noise, 
                  embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
                  embedding_scale=embedding_scale).squeeze(0)
        else:
            s_pred = model.sampler(noise = noise,
                            embedding=bert_dur,
                            embedding_scale=embedding_scale,
                            features=ref_s, # reference from the same speaker as the embedding
                            num_steps=diffusion_steps).squeeze(1)
        
        if s_prev is not None:
            # convex combination of previous and current style
            s_pred = alpha * s_prev + (1 - alpha) * s_pred
        
        s = s_pred[:, 128:]
        ref = s_pred[:, :128]
        
        if ref_s is not None:
            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)/speed
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
               
        if ref_s is not None:
            pred_dur[0] = 30


        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))

        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        if ref_s is not None:
            out = out[:,:, 14500:]
    return out.squeeze().cpu().numpy(), s_pred, ps


@spaces.GPU
def inference(model, text, voice_audio, progress, speed=1, alpha=0.4, beta=0.4, diffusion_steps=10, embedding_scale=1.2):

    wavs = []
    s_prev = None

    #sentences = text.split('|')
    sentences = split_to_parts(text)

    phonemes = ''
    noise = torch.randn(1,1,256).to(device)
    ref_s = compute_style(voice_audio) if voice_audio else None
    for text in sentences:
        if text.strip() == "": continue
        wav, s_prev, ps = _inf(model, text, ref_s, speed, s_prev, noise, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
        wavs.append(wav)
        phonemes += ' ' + ps
    return  np.concatenate(wavs), phonemes
