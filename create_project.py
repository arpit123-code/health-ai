from pathlib import Path

# Current working directory
root = Path(".")

# Folders to create

folders = [
    "config",
    "certs",
    "data/knowledge_base/healthcare",
    "data/vector_store",
    "data/audio_cache/input",
    "data/audio_cache/output",
    "src/audio",
    "src/speech",
    "src/rag",
    "src/pipeline",
    "app",
    "tests",
]

files=[
    "requirements.txt",
    "main.py",
    ".env",
]

for folder in folders:
    (root/folder).mkdir(parents=True, exist_ok=True)

for file in files:
    (root/file).touch(exist_ok=True)

print("Project structure created successfully.")