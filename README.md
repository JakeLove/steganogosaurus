# Steganogosaurus
Python program to hide monochrome bitmaps in audio files.

## Setup
Requires a few dependencies:
```
pip install scipy
pip install pillow
```

## Basic Usage
To encode your bitmap (must be monochrome) into an audio file run the following command
```
python steganogasaurus.py image.bmp
```

The program should produce a file called image.wav, the spectrogram of the produced file will contain the image.bmp. To view the spectrogram use a tool such as [SonicVisualizer](https://www.sonicvisualiser.org/) or [Audacity](https://www.audacityteam.org/).

## Results
Below is a comparison a spectrogram of the IBM logo encoded into a 44100Hz wav file. It is a good as the difficulty of creating good square waves is highlighted for a loq frequency signal.

![Image](https://i.imgur.com/sZU5oKt.png)

## Options
Some parameters can be specified explicitly. For more info use:
```
python steganogasaurus.py -h
```
