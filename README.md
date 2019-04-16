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
Below is a comparison a spectrogram of your logo encoded into a 44100Hz wav file. Unfortunately the IBM logo does not encode very well into an audio file. This is due to the fact it is composed entirely of square waves, which are not represented well by a low frequency DFT.

![Image](https://i.imgur.com/sZU5oKt.png)

## Options
Some parameters can be specified explicitly. For more info use:
```
python steganogasaurus.py -h
```
