import os
import subprocess
from IPython.display import display, HTML
from base64 import b64encode

def play_video_universal(video_path: str):
    """
    Универсальная функция, которая работает везде:
    - VS Code (скрипты)
    - VS Code (Jupyter) 
    - Google Colab
    - Jupyter Lab
    - Локальные скрипты
    """
    
    # Проверка существования файла
    if not os.path.exists(video_path):
        print(f"❌ Файл не найден: {video_path}")
        return
    
    abs_path = os.path.abspath(video_path)
    
    # Определяем среду выполнения
    try:
        from IPython import get_ipython
        in_notebook = get_ipython() is not None
    except:
        in_notebook = False
    
    # ВЕБ-СРЕДЫ (Colab, Jupyter Lab)
    if in_notebook:
        try:
            # Сжимаем видео для веба
            compressed_path = video_path.replace(".mp4", "_web.mp4")
            result = subprocess.run([
                "ffmpeg", "-i", video_path, 
                "-vcodec", "libx264", 
                "-crf", "23", 
                "-preset", "fast", 
                "-movflags", "faststart",  # для быстрой загрузки
                compressed_path,
                "-y"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if result.returncode != 0:
                print("❌ Ошибка при сжатии видео")
                return

            # Читаем сжатое видео
            with open(compressed_path, "rb") as f:
                video_bytes = f.read()

            # Кодируем в base64
            video_base64 = b64encode(video_bytes).decode()

            # Отображаем в HTML
            display(HTML(f"""
            <video width="800" controls style="border-radius: 8px; margin: 10px 0;">
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                Ваш браузер не поддерживает видео тег.
            </video>
            <p><small>Файл: {os.path.basename(video_path)}</small></p>
            """))

            # Удаляем сжатый файл
            os.remove(compressed_path)
            print(f"✅ Видео загружено в ноутбук")
            return
        except Exception as e:
            print(f"❌ Не удалось отобразить видео: {e}")
    
    # ЛОКАЛЬНЫЕ СРЕДЫ (VS Code скрипты, терминал)
    else:
        try:
            if sys.platform == "win32":
                os.startfile(abs_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", abs_path], check=True)
            else:
                subprocess.run(["xdg-open", abs_path], check=True)
            print(f"✅ Видео открыто в системном плеере: {abs_path}")
        except Exception as e:
            print(f"❌ Не удалось открыть видео автоматически: {e}")
            print(f"📁 Откройте вручную: {abs_path}")
