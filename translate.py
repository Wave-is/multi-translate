import os
import logging
import subprocess
import shutil
from app_rvc import SoniTranslate
from soni_translate.utils import remove_directory_contents, create_directories

# Настройка логгирования
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("translation_log.txt"),
        logging.StreamHandler()
    ]
)

# Очистка папки /VOICE-TRANSLATED перед началом работы
audio_dir_tr = "../VOICE-TRANSLATED"
if os.path.exists(audio_dir_tr):
    remove_directory_contents(audio_dir_tr)
else:
    create_directories(audio_dir_tr)

# Очистка папки /FINAL перед началом работы
final_dir = "../FINAL"
if os.path.exists(final_dir):
    remove_directory_contents(final_dir)
else:
    create_directories(final_dir)

# Очистка папки /FINAL-ZIP перед началом работы
final_zip_dir = "../FINAL-ZIP"
if os.path.exists(final_zip_dir):
    remove_directory_contents(final_zip_dir)
else:
    create_directories(final_zip_dir)

# Очистка папки /INPUT-VIDEO после завершения работы
videos_dir = "../INPUT-VIDEO"
if not os.path.exists(videos_dir):
    create_directories(videos_dir)

# Очистка папки /MUSIC после завершения работы
music_dir = "../MUSIC"
if not os.path.exists(music_dir):
    create_directories(music_dir)

# Инициализация SoniTranslate с использованием GPU
soni_translate = SoniTranslate(cpu_mode=False)  # Используем GPU

# Загрузка видео и SRT-файла
video_file = None
srt_file = None
for file in os.listdir(videos_dir):
    if file.endswith(".mp4") or file.endswith(".mp3"):
        video_file = os.path.join(videos_dir, file)
    elif file.endswith(".srt"):
        srt_file = os.path.join(videos_dir, file)

if not video_file:
    logging.error("Видео или аудио файл не найден в папке /INPUT-VIDEO.")
    exit(1)

# Настройки перевода
settings = {
    "media_file": video_file,
    "subtitle_file": srt_file,
    "YOUR_HF_TOKEN": "",  # Если требуется Hugging Face Token
    "preview": False,
    "transcriber_model": "large-v3",
    "batch_size": 4,
    "compute_type": "float32",
    "origin_language": "Russian (ru)",
    "target_language": "English (en)",  # Для первого перевода
    "min_speakers": 1,
    "max_speakers": 1,
    "tts_voice00": "en-US-AvaNeural-Female",  # Голос для английского
    "tts_voice01": "de-DE-KatjaNeural-Female",  # Голос для немецкого
    "video_output_name": "",
    "mix_method_audio": "Adjusting volumes and mixing audio",
    "max_accelerate_audio": 1.9,
    "acceleration_rate_regulation": True,
    "volume_original_audio": 0,
    "volume_translated_audio": 2.5,
    "output_format_subtitle": "srt",
    "get_translated_text": False,
    "get_video_from_text_json": False,
    "text_json": "{}",
    "avoid_overlap": True,
    "vocal_refinement": False,
    "literalize_numbers": True,
    "segment_duration_limit": 15,
    "diarization_model": "disable",
    "translate_process": "google_translator_batch",
    "output_type": "mp3",
    "voiceless_track": True,
    "voice_imitation": False,
    "voice_imitation_max_segments": 3,
    "voice_imitation_vocals_dereverb": False,
    "voice_imitation_remove_previous": True,
    "voice_imitation_method": "freevc",
    "dereverb_automatic_xtts": True,
    "text_segmentation_scale": "sentence",
    "divide_text_segments_by": "",
    "soft_subtitles_to_video": True,
    "burn_subtitles_to_video": False,
    "enable_cache": True,
    "custom_voices": True,
    "custom_voices_workers": 1,
    "is_gui": False,
}

# Настройки Custom Voice R.V.C.
custom_voice_settings = {
    "file_model": "weights/mary-v1.pth",
    "pitch_algo": "rmvpe",
    "pitch_lvl": 0,
    "file_index": None,
    "index_influence": 0.75,
    "respiration_median_filtering": 3,
    "envelope_ratio": 0.25,
    "consonant_breath_protection": 0.5,
}

# Применение настроек Custom Voice R.V.C. для TTS Speaker 01
soni_translate.vci.apply_conf(
    tag="TTS Speaker 01",  # Указываем, что настройки применяются к первому спикеру
    file_model=custom_voice_settings["file_model"],
    pitch_algo=custom_voice_settings["pitch_algo"],
    pitch_lvl=custom_voice_settings["pitch_lvl"],
    file_index=custom_voice_settings["file_index"],
    index_influence=custom_voice_settings["index_influence"],
    respiration_median_filtering=custom_voice_settings["respiration_median_filtering"],
    envelope_ratio=custom_voice_settings["envelope_ratio"],
    consonant_breath_protection=custom_voice_settings["consonant_breath_protection"],
)

# Список языков и голосов
languages_and_voices = {
    "English (en)": "en-US-AvaNeural-Female",
    "German (de)": "de-DE-SeraphinaMultilingualNeural-Female",
    "French (fr)": "fr-FR-VivienneMultilingualNeural-Female",
    "Italian (it)": "it-IT-ElsaNeural-Female",
    "Spanish (es)": "es-ES-XimenaNeural-Female",
    "Portuguese (pt)": "pt-BR-ThalitaMultilingualNeural-Female",
    "Chinese - Simplified (zh-CN)": "zh-CN-XiaoxiaoNeural-Female",
    "Polish (pl)": "pl-PL-ZofiaNeural-Female",
    "Turkish (tr)": "tr-TR-EmelNeural-Female",
    "Bulgarian (bg)": "bg-BG-KalinaNeural-Female",
    "Hindi (hi)": "hi-IN-SwaraNeural-Female",
    "Romanian (ro)": "ro-RO-AlinaNeural-Female",
    "Indonesian (id)": "id-ID-GadisNeural-Female",
    "Vietnamese (vi)": "vi-VN-HoaiMyNeural-Female",
    "Thai (th)": "th-TH-PremwadeeNeural-Female",
    "Bengali (bn)": "bn-IN-TanishaaNeural-Female",
    "Korean (ko)": "ko-KR-SunHiNeural-Female",
    "Japanese (ja)": "ja-JP-NanamiNeural-Female",
}

# Переменная для хранения всех выходных файлов
all_output_files = []

# Перевод на каждый язык
for lang, voice in languages_and_voices.items():
    logging.info(f"Начинаем перевод на {lang}...")
    settings["target_language"] = lang
    settings["tts_voice00"] = voice  # Устанавливаем голос для текущего языка
    output = soni_translate.multilingual_media_conversion(**settings)
    logging.info(f"Результат перевода: {output}")
    all_output_files.extend(output)

# Перемещение переведенных файлов в папку /VOICE-TRANSLATED
for output_file in all_output_files:
    if isinstance(output_file, str) and os.path.exists(output_file):
        os.rename(output_file, os.path.join(audio_dir_tr, os.path.basename(output_file)))
    else:
        logging.warning(f"Файл не найден: {output_file}")

# Проверка содержимого папки /VOICE-TRANSLATED
if os.path.exists(audio_dir_tr):
    logging.info(f"Содержимое папки {audio_dir_tr}: {os.listdir(audio_dir_tr)}")
else:
    logging.error(f"Папка {audio_dir_tr} не существует.")
    exit(1)

# Добавление музыкальной дорожки и сохранение в формате MP3
music_files = [os.path.join(music_dir, f) for f in os.listdir(music_dir) if f.endswith(".mp3") or f.endswith(".mp4")]

for audio_file in os.listdir(audio_dir_tr):
    if audio_file.endswith(".mp3") or audio_file.endswith(".mp4"):  # Обрабатываем как .mp3, так и .mp4 файлы
        audio_path = os.path.join(audio_dir_tr, audio_file)
        for music_file in music_files:
            output_file = os.path.join(final_dir, f"final_{os.path.splitext(audio_file)[0]}.mp3")  # Всегда сохраняем как .mp3
            # Смешиваем аудио с музыкой и сохраняем в формате MP3
            cmd = [
                "ffmpeg", "-y", "-i", audio_path, "-i", music_file,
                "-filter_complex", "[0:a][1:a]amerge=inputs=2[out]",  # Убрано увеличение громкости
                "-map", "[out]", "-ac", "2", "-f", "mp3", output_file
            ]
            try:
                logging.info(f"Обработка файла: {audio_file} с музыкой: {os.path.basename(music_file)}")
                subprocess.run(cmd, check=True, timeout=1200)  # Увеличьте тайм-аут для больших файлов
                logging.info(f"Файл {output_file} создан с добавлением музыкальной дорожки.")
            except subprocess.CalledProcessError as e:
                logging.error(f"Ошибка при обработке файла {audio_file}: {e}")
            except subprocess.TimeoutExpired:
                logging.error(f"Тайм-аут при обработке файла {audio_file}.")
            except Exception as e:
                logging.error(f"Неизвестная ошибка при обработке файла {audio_file}: {e}")

# Проверка содержимого папки /FINAL
if os.path.exists(final_dir):
    final_files = os.listdir(final_dir)
    logging.info(f"Содержимое папки {final_dir}: {final_files}")
    if not final_files:
        logging.error("Папка /FINAL пуста. Склеивание не выполнено.")
        exit(1)
else:
    logging.error(f"Папка {final_dir} не существует.")
    exit(1)

# Архивирование папки /FINAL
try:
    shutil.make_archive(os.path.join(final_zip_dir, "final"), 'zip', final_dir)
    logging.info("Финальные файлы заархивированы и сохранены в /FINAL-ZIP")
except Exception as e:
    logging.error(f"Ошибка при архивировании: {e}")

# Очистка папки /INPUT-VIDEO после завершения работы
if os.path.exists(videos_dir):
    remove_directory_contents(videos_dir)
    logging.info("Папка /INPUT-VIDEO очищена.")

# Очистка папки /MUSIC после завершения работы
if os.path.exists(music_dir):
    remove_directory_contents(music_dir)
    logging.info("Папка /MUSIC очищена.")

logging.info("Процесс завершен.")