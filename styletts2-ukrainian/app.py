import glob
import os
import gradio as gr
from infer import inference, split_to_parts
import onnxruntime
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import numpy as np

prompts_dir = os.path.join(os.path.dirname(__file__), 'voices')
prompts_list = sorted(glob.glob(os.path.join(prompts_dir, '*.wav')))
prompts_list = ['.'.join(p.split('/')[-1].split('.')[:-1]) for p in prompts_list]

verbalizer_model_name = "skypro1111/mbart-large-50-verbalization"

def cache_model_from_hf(repo_id, model_dir="./"):
    """Download ONNX models from HuggingFace Hub."""
    files = ["onnx/encoder_model.onnx", "onnx/decoder_model.onnx", "onnx/decoder_model.onnx_data"]
    
    for file in files:
        hf_hub_download(
            repo_id=repo_id,
            filename=file,
            local_dir=model_dir,
        )


def create_onnx_session(model_path, use_gpu=True):
    """Create an ONNX inference session."""
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_mem_pattern = True
    session_options.enable_mem_reuse = True
    session_options.intra_op_num_threads = 8
    session_options.log_severity_level = 1
    
    cuda_provider_options = {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 0,  # 0 means no limit
        'cudnn_conv_algo_search': 'DEFAULT',
        'do_copy_in_default_stream': True,
    }
    
    if use_gpu and 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        providers = [('CUDAExecutionProvider', cuda_provider_options)]
    else:
        providers = ['CPUExecutionProvider']
    
    session = onnxruntime.InferenceSession(
        model_path,
        providers=providers,
        sess_options=session_options
    )
    
    return session

def init_verbalizer():
    cache_model_from_hf(verbalizer_model_name)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(verbalizer_model_name)
    tokenizer.src_lang = "uk_UA"
    tokenizer.tgt_lang = "uk_UA"

    print("Creating ONNX sessions...")
    encoder_session = create_onnx_session("onnx/encoder_model.onnx")
    decoder_session = create_onnx_session("onnx/decoder_model.onnx")
    return tokenizer, encoder_session, decoder_session

tokenizer, encoder_session, decoder_session = init_verbalizer()


def generate_text(text):
    """Generate text for a single input."""
    # Prepare input
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    
    # Run encoder
    encoder_outputs = encoder_session.run(
        output_names=["last_hidden_state"],
        input_feed={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )[0]
    
    # Initialize decoder input
    decoder_input_ids = np.array([[tokenizer.pad_token_id]], dtype=np.int64)
    
    # Generate sequence
    for _ in range(512):
        # Run decoder
        decoder_outputs = decoder_session.run(
            output_names=["logits"],
            input_feed={
                "input_ids": decoder_input_ids,
                "encoder_hidden_states": encoder_outputs,
                "encoder_attention_mask": attention_mask,
            }
        )[0]
        
        # Get next token
        next_token = decoder_outputs[:, -1:].argmax(axis=-1)
        decoder_input_ids = np.concatenate([decoder_input_ids, next_token], axis=-1)
        
        # Check if sequence is complete
        if tokenizer.eos_token_id in decoder_input_ids[0]:
            break
    
    # Decode sequence
    output_text = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
    return output_text

def verbalize(text):
    parts = split_to_parts(text)
    verbalized = ''
    for part in parts:
        verbalized += generate_text(part)
    return verbalized

description = f'''
<h1 style="text-align:center;">StyleTTS2 ukrainian demo</h1><br>
Програма може не коректно визначати деякі наголоси і не перетворює цифри, акроніми і різні скорочення в словесну форму.
Якщо наголос не правильний, використовуйте символ + після наголошеного складу.
Також дуже маленькі речення можуть крешати, тому пишіть щось більше а не одне-два слова.

'''

examples = [
    ["Решта окупантів звернула на Вокзальну — центральну вулицю Бучі. Тільки уявіть їхній настрій, коли перед ними відкрилася ця пасторальна картина! Невеличкі котеджі й просторіші будинки шикуються обабіч, перед ними вивищуються голі липи та електро-стовпи, тягнуться газони й жовто-чорні бордюри. Доглянуті сади визирають із-поза зелених парканів, гавкотять собаки, співають птахи… На дверях будинку номер тридцять шість досі висить різдвяний вінок.", 1.0],
    ["Одна дівчинка стала королевою Франції. Звали її Анна, і була вона донькою Ярослава Му+дрого, великого київського князя. Він опі+кувався літературою та культурою в Київській Русі+, а тоді переважно про таке не дбали – більше воювали і споруджували фортеці.", 1.0],
    ["Одна дівчинка народилася і виросла в Америці, та коли стала дорослою, зрозуміла, що дуже любить українські вірші й найбільше хоче робити вистави про Україну. Звали її Вірляна. Дід Вірляни був український мовознавець і педагог Кость Кисілевський, котрий навчався в Лейпцизькому та Віденському університетах і, після Другої світової війни виїхавши до США, започаткував систему шкіл українознавства по всій Америці. Тож Вірляна зростала в українському середовищі, а окрім того – в середовищі вихідців з інших країн.", 1.0],
    ["За інформацією від Державної служби з надзвичайних ситуацій станом на 7 ранку 15 липня.", 1.0],
    ["Очікується, що цей застосунок буде запущено 22.08.2025.", 1.0],
]

def synthesize_multi(text, voice_audio, speed, progress=gr.Progress()):
    prompt_audio_path = os.path.join(prompts_dir, voice_audio+'.wav')
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    if len(text) > 50000:
        raise gr.Error("Text must be <50k characters")
    print("*** saying ***")
    print(text)
    print("*** end ***")
    
    return 24000, inference('multi', text, prompt_audio_path, progress, speed=speed, alpha=0, beta=0, diffusion_steps=20, embedding_scale=1.0)[0]



def synthesize_single(text, speed,  progress=gr.Progress()):
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    if len(text) > 50000:
        raise gr.Error("Text must be <50k characters")
    print("*** saying ***")
    print(text)
    print("*** end ***")
    
    return 24000, inference('single',  text, None, progress, speed=speed, alpha=1, beta=0, diffusion_steps=4, embedding_scale=1.0)[0]

def select_example(df, evt: gr.SelectData):
    return evt.row_value   
    
with gr.Blocks() as single:
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Text(label='Text:', lines=5, max_lines=10)
            verbalize_button = gr.Button("Вербалізувати(beta)")
            speed = gr.Slider(label='Швидкість:', maximum=1.3, minimum=0.7, value=1.0)
            verbalize_button.click(verbalize, inputs=[input_text], outputs=[input_text])
            
        with gr.Column(scale=1):
            output_audio = gr.Audio(
                    label="Audio:",
                    autoplay=False,
                    streaming=False,
                    type="numpy",
                )
            synthesise_button = gr.Button("Синтезувати")
            
            synthesise_button.click(synthesize_single, inputs=[input_text, speed], outputs=[output_audio])
    
    with gr.Row():
        examples_table = gr.Dataframe(wrap=True, headers=["Текст", "Швидкість"], datatype=["str", "number"], value=examples, interactive=False)
        examples_table.select(select_example, inputs=[examples_table], outputs=[input_text, speed])
    
with gr.Blocks() as multy:
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Text(label='Text:', lines=5, max_lines=10)
            verbalize_button = gr.Button("Вербалізувати(beta)")
            speed = gr.Slider(label='Швидкість:', maximum=1.3, minimum=0.7, value=1.0)
            speaker = gr.Dropdown(label="Голос:", choices=prompts_list, value=prompts_list[0])
            verbalize_button.click(verbalize, inputs=[input_text], outputs=[input_text])

        with gr.Column(scale=1):
            output_audio = gr.Audio(
                    label="Audio:",
                    autoplay=False,
                    streaming=False,
                    type="numpy",
                )
            synthesise_button = gr.Button("Синтезувати")
            
            synthesise_button.click(synthesize_multi, inputs=[input_text, speaker, speed], outputs=[output_audio])
    with gr.Row():
        examples_table = gr.Dataframe(wrap=True, headers=["Текст", "Швидкість"], datatype=["str", "number"], value=examples, interactive=False)
        examples_table.select(select_example, inputs=[examples_table], outputs=[input_text, speed])




with gr.Blocks(title="StyleTTS2 ukrainian demo", css="") as demo:
    gr.Markdown(description)
    gr.TabbedInterface([multy, single], ['Multі speaker', 'Single speaker'])
    

if __name__ == "__main__":
    demo.queue(api_open=True, max_size=15).launch(show_api=True)