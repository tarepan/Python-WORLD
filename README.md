# PYTHON WORLD VOCODER
Clone of *PYTHON WORLD VOCODER*.  

This is a line-by-line implementation of WORLD vocoder (Matlab, C++) in python.

## Installation

```bash
pip install -r requirements.txt
```

## Example

For quick demo, run ```example/prodosy.py```:
```bash
python example.prosody
```

This demo contains analysis, modification (pitch, duration, spectrum) and synthesis.  
Below is step-by-step examples.  

### Encode
Prepare audio:  

```python
from scipy.io.wavfile import read as wavread
wav_path = "path/to/your/audio.wav"
fs, x_int16 = wavread(wav_path) # `fs`         - sampling frequency
x = x_int16 / (2 ** 15 - 1)     # `x` :: float - speech signal
```

Then, decode the speech with `World` instance:

```python
from world import main
vocoder = main.World()
dat = vocoder.encode(fs, x, f0_method='dio') # `dat` - WORLD parameter dictionary (fo, spec, ap, etc...)
```

### Manipulation
We can scale the pitch:

```python
scale = 1.5 # Be careful when you scale the pich because there is upper limit and lower limit.
dat = vocoder.scale_pitch(dat, scale)
```

We can make speech faster or slower:

```python
speed = 2.0
dat = vocoder.scale_duration(dat, speed)
```

### Decode
todo

### Speed Test
In ```test/speed.py```, we estimate the time of analysis.

### Others
To extract log-filterbanks, MCEP-40, VAE-12 as described in the paper `Using a Manifold Vocoder for Spectral Voice and Style Conversion`, check ```test/spectralFeatures.py```.  
You need Keras 2.2.4 and TensorFlow 1.14.0 to extract VAE-12.  
Check out [speech samples](https://tuanad121.github.io/samples/2019-09-15-Manifold/).  


## Options

Supported *fo* analysis methods:

- `SWIPE`
- `DIO`
- `Harvest`: slowest (multi-core optimized by ```numba``` and ```python multiprocessing```)
- Raw *f<sub>o</sub>* contour: any method you prefer

You can use 'd4c requiem' mode (d4c_requiem analysis & requiem_synthesis in WORLD version 0.2.2):

```python
dat = vocoder.encode(fs, x, f0_method='harvest', is_requiem=True)
```


## Note

### pitch-synchronous analysis
The vocoder use pitch-synchronous analysis, the size of each window is determined by fundamental frequency ```F0```.  
The centers of the windows are equally spaced with the distance of ```frame_period``` ms.  

### Limit of Parameters
You must satisfy equation `f0_floor = 3.0 * fs / fft_size`.  
`f0_floor` is assumed lowest *f<sub>o</sub>*, `fs` is sampling frequency and `fft_size` is The Fourier transform size.  

Under fixed `fs`, decrease of ```fft_size``` results in ```f0_floor``` increases.  
But, a high ```f0_floor``` might be not good for the analysis of male voices.  
So there is a kind of limitation in `fft_size`.  


## CITATION
```
Dinh, T., Kain, A., & Tjaden, K. (2019). Using a manifold vocoder for spectral voice and style conversion. Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, 2019-September, 1388-1392.
```