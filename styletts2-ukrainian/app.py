import glob
import os
import gradio as gr
from infer import inference


prompts_dir = os.path.join(os.path.dirname(__file__), 'voices')
prompts_list = sorted(glob.glob(os.path.join(prompts_dir, '*.wav')))
prompts_list = ['.'.join(p.split('/')[-1].split('.')[:-1]) for p in prompts_list]


description = f'''
<h1 style="text-align:center;">StyleTTS2 ukrainian demo</h1><br>
Програма може не коректно визначати деякі наголоси і не перетворює цифри, акроніми і різні скорочення в словесну форму.
Якщо наголос не правильний, використовуйте символ + після наголошеного складу.
Також дуже маленькі речення можуть крешати, тому пишіть щось більше а не одне-два слова.

'''

examples = [
    ["Решта окупантів звернула на Вокзальну — центральну вулицю Бучі. Тільки уявіть їхній настрій, коли перед ними відкрилася ця пасторальна картина! Невеличкі котеджі й просторіші будинки шикуються обабіч, перед ними вивищуються голі липи та електро-стовпи, тягнуться газони й жовто-чорні бордюри. Доглянуті сади визирають із-поза зелених парканів, гавкотять собаки, співають птахи… На дверях будинку номер тридцять шість досі висить різдвяний вінок.", 1.0],
    ["Одна дівчинка стала королевою Франції. Звали її Анна, і була вона донькою Ярослава Му+дрого, великого київського князя. Він опі+кувався літературою та культурою в Київській Русі+, а тоді переважно про таке не дбали – більше воювали і споруджували фортеці.", 1.0],
    ["Одна дівчинка народилася і виросла в Америці, та коли стала дорослою, зрозуміла, що дуже любить українські вірші й найбільше хоче робити вистави про Україну. Звали її Вірляна. Дід Вірляни був український мовознавець і педагог Кость Кисілевський, котрий навчався в Лейпцизькому та Віденському університетах і, після Другої світової війни виїхавши до США, започаткував систему шкіл українознавства по всій Америці. Тож Вірляна зростала в українському середовищі, а окрім того – в середовищі вихідців з інших країн.", 1.0]
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

    return 24000, inference('single', text, None, progress, speed=speed, alpha=1, beta=0, diffusion_steps=4, embedding_scale=1.0)[0]

def select_example(df, evt: gr.SelectData):
    return evt.row_value

with gr.Blocks() as single:
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Text(label='Text:', lines=5, max_lines=10)
            speed = gr.Slider(label='Швидкість:', maximum=1.3, minimum=0.7, value=1.0)
            synthesise_button = gr.Button("Синтезувати")
        with gr.Column(scale=1):
            output_audio = gr.Audio(
                    label="Audio:",
                    autoplay=False,
                    streaming=False,
                    type="numpy",
                )

            synthesise_button.click(synthesize_single, inputs=[input_text, speed], outputs=[output_audio])

    with gr.Row():
        examples_table = gr.Dataframe(wrap=True, headers=["Текст", "Швидкість"], datatype=["str", "number"], value=examples, interactive=False)
        examples_table.select(select_example, inputs=[examples_table], outputs=[input_text, speed])

with gr.Blocks() as multy:
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Text(label='Text:', lines=5, max_lines=10)
            speed = gr.Slider(label='Швидкість:', maximum=1.3, minimum=0.7, value=1.0)
            speaker = gr.Dropdown(label="Голос:", choices=prompts_list, value=prompts_list[0])

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
