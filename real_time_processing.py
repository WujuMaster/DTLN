#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 16:23:15 2020

@author: nils
"""

import soundfile as sf
import numpy as np
from tensorflow import saved_model, constant
import time

sample_path = './samples/sample_noisy.wav'
enhanced = './samples/enhanced_sample_noisy.wav'

block_len = 512
block_shift = 128
# load model
model = saved_model.load('./pretrained_model/test/SavedModel_8khz')
infer = model.signatures["serving_default"]
# load audio file at 16k fs (please change)
audio, fs = sf.read(sample_path)
# check for sampling rate
if fs != 8000:
    raise ValueError('This model only supports 8k sampling rate.')
# preallocate output audio
out_file = np.zeros((len(audio)))
# create buffer
in_buffer = np.zeros((block_len))
out_buffer = np.zeros((block_len))
# calculate number of blocks
num_blocks = (audio.shape[0] - (block_len-block_shift)) // block_shift
# iterate over the number of blcoks
time_array = []
for idx in range(num_blocks):
    start_time = time.time()

    # shift values and write to buffer
    in_buffer[:-block_shift] = in_buffer[block_shift:]
    in_buffer[-block_shift:] = audio[idx *
                                     block_shift:(idx*block_shift)+block_shift]
    # create a batch dimension of one
    in_block = np.expand_dims(in_buffer, axis=0).astype('float32')
    # process one block
    out_block = infer(constant(in_block))['conv1d_1']
    # shift values and write to buffer
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = np.zeros((block_shift))
    out_buffer += np.squeeze(out_block)
    # write block to output file
    out_file[idx*block_shift:(idx*block_shift) +
             block_shift] = out_buffer[:block_shift]
    time_array.append(time.time()-start_time)

# write to .wav file
sf.write(enhanced, out_file, fs)

print('Processing Time [ms]:')
print(np.mean(np.stack(time_array))*1000)
print('Processing finished.')
