# INVERSYNTH: parameter learning for computer FM synthesizers

Python implementation of [INVERSYNTH](https://arxiv.org/abs/1812.06349) with Keras.

## Description

In this repository two CNN structures are implemented for the purpose of parameter learning for a predefined computer FM synthesizer. 

## Requirements

1. Clone this repo.
2. Python 3.7
3. Install your preferred version of TensorFlow 1.4.0 (for CPU, GPU; from PyPI, compiled, etc).
4. Install the rest of the requirements: `pip install -r requirements.txt`

## Data Generation

```bash
cd models
python synth_generator.py [--length --num_classes --size --out_dir --wav_dir --sample_rate]
```

This generates `size` wav sound files using a FM synthesizer defined the `models/synth_generator.py` with sample rate of `sample_rate` of length `length` saved in `wav_dir`, and associated meta data in `out_dir`. 

## Running the INVERSYNTH model

Currently two CNN models are available: E2E and CONV6XL. Please adjust the corresponding macro in `models/cnn.py` to run the intended model. 
 
```bash
cd models
python cnn.py 
```

More parameters which can be tuned in `models/cnn.py` are listed in the table below.

<div itemscope="" itemtype="http://schema.org/Organization" itemprop="provider">
  <table>
    <tbody><tr>
      <th>Parameters</th>
      <th>Default Value</th>
    </tr>
    <tr>
      <td>NUM_EPOCH</td>
      <td>100</td>
    </tr>
    <tr>
      <td>BATCH_SIZE</td>
      <td>64</td>
    </tr>
    <tr>
      <td>SAMPLE_RATE</td>
      <td>16384 <code>this should match that of input data</code></td>
    </tr>
    <tr>
      <td>OUT_DIR</td>
      <td><code>../data</code></td>
    </tr>
    <tr>
      <td>MODEL_DIR</td>
      <td><code>../saved_models/</code></td>
    </tr>
    <tr>
      <td>MODEL_NAME</td>
      <td>E2E</td>
    </tr>
  </tbody></table>
</div>

## Generate and Evaluate Individual Samples

To generate a sample
```bash
cd models
python generate_sample.py
```
Command line arguments can be used to tune the parameters of the FM synthesizer.

To evaluate a sample (any wav file works)
```bash
cd models
python evaluate_sampe.py 
```