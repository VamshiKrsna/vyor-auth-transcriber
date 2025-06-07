from setuptools import setup, find_packages

setup(
    name="vyor-auth-transcribe",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyaudio",
        "whisper",
        "torchaudio",
        "nemo_toolkit[all]",
        "torch",
        "openai-whisper",
    ],
    entry_points={
        'console_scripts': [
            'voiceauth-transcribe = voiceauth.main:main'
        ]
    },
    author="VamZZZ",
    description="Voice Authentication and Transcription system using PyTorch, Whisper, and NeMo.",
    classifiers=[
        "Programming Language :: Python :: 3.12",
    ],
    python_requires='>=3.9',
)
