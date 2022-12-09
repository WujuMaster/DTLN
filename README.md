 # Dual-signal Transformation LSTM Network

 Tensorflow 2.x implementation of the stacked dual-signal transformation LSTM network (DTLN) for real-time noise suppression.

---
The original repository can be found [here](https://github.com/breizhn/DTLN.git). 

The DTLN model was handed in to the deep noise suppression challenge ([DNS-Challenge](https://github.com/microsoft/DNS-Challenge)) and the paper was presented at Interspeech 2020. 

For more information see the [paper](https://www.isca-speech.org/archive/interspeech_2020/westhausen20_interspeech.html). The results of the DNS-Challenge are published [here](https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-interspeech-2020/#!results).
---

---

Author: Nils L. Westhausen ([Communication Acoustics](https://uol.de/en/kommunikationsakustik) , Carl von Ossietzky University, Oldenburg, Germany)

This code is licensed under the terms of the MIT license.


---
### Citing:

If you are using the DTLN model, please cite:

```BibTex
@inproceedings{Westhausen2020,
  author={Nils L. Westhausen and Bernd T. Meyer},
  title={{Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression}},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={2477--2481},
  doi={10.21437/Interspeech.2020-2631},
  url={http://dx.doi.org/10.21437/Interspeech.2020-2631}
}
```


---
### Contents of the README:

- [Dual-signal Transformation LSTM Network](#dual-signal-transformation-lstm-network)
  - [For more information see the paper. The results of the DNS-Challenge are published here.](#for-more-information-see-the-paper-the-results-of-the-dns-challenge-are-published-here)
    - [Citing:](#citing)
    - [Contents of the README:](#contents-of-the-readme)
    - [Contents of the repository:](#contents-of-the-repository)
    - [Python dependencies:](#python-dependencies)
    - [Training data preparation:](#training-data-preparation)
    - [Run a training of the DTLN model:](#run-a-training-of-the-dtln-model)
    - [Measuring the execution time of the DTLN model with the SavedModel format:](#measuring-the-execution-time-of-the-dtln-model-with-the-savedmodel-format)
    - [Real time processing with the SavedModel format:](#real-time-processing-with-the-savedmodel-format)
    - [Real time processing with tf-lite:](#real-time-processing-with-tf-lite)
    - [Model conversion and real time processing with ONNX:](#model-conversion-and-real-time-processing-with-onnx)


---
### Contents of the repository:

*  **DTLN_model.py** \
  This file is containing the model, data generator and the training routine.
*  **run_training.py** \
  Script to run the training. Before you can start the training with `$ python run_training.py`you have to set the paths to you training and validation data inside the script. The training script uses a default setup.
* **run_evaluation.py** \
  Script to process a folder with optional subfolders containing .wav files with a trained DTLN model. With the pretrained model delivered with this repository a folder can be processed as following: \
  `$ python run_evaluation.py -i /path/to/input -o /path/for/processed -m ./pretrained_model/model.h5` \
  The evaluation script will create the new folder with the same structure as the input folder and the files will have the same name as the input files.
* **measure_execution_time.py** \
  Script for measuring the execution time with the saved DTLN model in `./pretrained_model/dtln_saved_model/`. For further information see this [section](#measuring-the-execution-time-of-the-dtln-model-with-the-savedmodel-format).
* **real_time_processing.py** \
  Script, which explains how real time processing with the SavedModel works. For more information see this [section](#real-time-processing-with-the-savedmodel-format).
+  **./pretrained_model/** \
   * `DTLN_model_8khz_42epoch.h5`: Model weights as described in thesis, with (frame_length, frame_shift) = (512, 128)
   * `DTLN_8khz_model_1.tflite` together with `DTLN_8khz_model_2.tflite`: same as `DTLN_model_8khz_42epoch.h5` but as TF-lite model with external state handling.
   * `my_custom_model_1.tflite` together with `my_custom_model_2.tflite`: Modified version with (frame_length, frame_shift) = (256, 64)
   
[To contents](#contents-of-the-readme)
   
---
### Python dependencies:

The following packages will be required for this repository:
* TensorFlow (2.x) - some files may require different versions, see the comments in each file for more details
* librosa
* wavinfo 


All additional packages (numpy, soundfile, etc.) should be installed on the fly when using conda or pip. I recommend using conda environments or [pyenv](https://github.com/pyenv/pyenv) [virtualenv](https://github.com/pyenv/pyenv-virtualenv) for the python environment. For training a GPU with at least 5 GB of memory is required. I recommend at least Tensorflow 2.1 with Nvidia driver 418 and Cuda 10.1. If you use conda Cuda will be installed on the fly and you just need the driver. For evaluation-only the CPU version of Tensorflow is enough. 

The tf-lite runtime must be downloaded from [here](https://www.tensorflow.org/lite/guide/python).

[To contents](#contents-of-the-readme)

---
### Training data preparation:

1. Data folders can be retrieved from [here](https://drive.google.com/drive/folders/1235mzYy0ZN6kUvmb2_AApwYlW15GgplE?usp=sharing)
2. After decompressed, all folders should be merged as following:
    .  
    ├── ...  
    ├── training_set  
    │&nbsp;&nbsp;├── train  
    │&nbsp;&nbsp;│&nbsp;&nbsp;├── clean  
    │&nbsp;&nbsp;│&nbsp;&nbsp;└── noisy  
    │&nbsp;&nbsp;├── val  
    │&nbsp;&nbsp;│&nbsp;&nbsp;├── clean  
    │&nbsp;&nbsp;│&nbsp;&nbsp;└── noisy  
    └── ...  
[To contents](#contents-of-the-readme)  

---
### Run a training of the DTLN model:

1. Make sure all dependencies are installed in your python environment.

2. Change the paths to your training and validation dataset in `run_training.py`.

3. Run `$ python run_training.py`. 


[To contents](#contents-of-the-readme)

---
### Measuring the execution time of the DTLN model with the SavedModel format:

In total there are three ways to measure the execution time for one block of the model: Running a sequence in Keras and dividing by the number of blocks in the sequence, building a stateful model in Keras and running block by block, and saving the stateful model in Tensorflow's SavedModel format and calling that one block by block. In the following I will explain how running the model in the SavedModel format, because it is the most portable version and can also be called from Tensorflow Serving.

A Keras model can be saved to the saved model format:
```python
import tensorflow as tf
'''
Building some model here
'''
tf.saved_model.save(your_keras_model, 'name_save_path')
```
Important here for real time block by block processing is, to make the LSTM layer stateful, so they can remember the states from the previous block.

The model can be imported with 
```python
model = tf.saved_model.load('name_save_path')
```

For inference we now first call this for mapping signature names to functions
```python
infer = model.signatures['serving_default']
```

and now for inferring the block `x` call
```python
y = infer(tf.constant(x))['conv1d_1']
```
This command gives you the result on the node `'conv1d_1'`which is our output node for real time processing. For more information on using the SavedModel format and obtaining the output node see this [Guide](https://www.tensorflow.org/guide/saved_model).

For making everything easier this repository provides a stateful DTLN SavedModel. 
For measuring the execution time call:
```
$ python measure_execution_time.py
```

[To contents](#contents-of-the-readme)

---

### Real time processing with the SavedModel format:

For explanation look at `real_time_processing.py`. 

Here some consideration for integrating this model in your project:
* The sampling rate of this model is fixed at 16 kHz. It will not work smoothly with other sampling rates.
* The block length of 32 ms and the block shift of 8 ms are also fixed. For changing these values, the model must be retrained.
* The delay created by the model is the block length, so the input-output delay is 32 ms.
* For real time capability on your system, the execution time must be below the length of the block shift, so below 8 ms. 
* If can not give you support on the hardware side, regarding soundcards, drivers and so on. Be aware, a lot of artifacts can come from this side.

[To contents](#contents-of-the-readme)

---
### Real time processing with tf-lite:

With TF 2.3 it is finally possible to convert LSTMs to tf-lite. It is still not perfect because the states must be handled seperatly for a stateful model and tf-light does not support complex numbers. That means that the model is splitted in two submodels when converting it to tf-lite and the calculation of the FFT and iFFT is performed outside the model. I provided an example script for explaining, how real time processing with the tf light model works (```real_time_processing_tf_lite.py```). In this script the tf-lite runtime is used. The runtime can be downloaded [here](https://www.tensorflow.org/lite/guide/python). Quantization works now.


[To contents](#contents-of-the-readme)

---
### Model conversion and real time processing with ONNX:

Finally I got the ONNX model working. 
For converting the model TF 2.1 and keras2onnx is required. keras2onnx can be downloaded [here](https://github.com/onnx/keras-onnx) and must be installed from source as described in the README. When all dependencies are installed, call:
```
$ python convert_weights_to_onnx.py -m /name/of/the/model.h5 -t onnx_model_name
```
to convert the model to the ONNX format. The model is split in two parts as for the TF-lite model. The conversion does not work on MacOS.
The real time processing works similar to the TF-lite model and can be looked up in following file: ```real_time_processing_onnx.py ```
The ONNX runtime required for this script can be installed with:
```
$ pip install onnxruntime
```
The execution time on the Macbook Air mid 2012 is around 1.13 ms for one block.
