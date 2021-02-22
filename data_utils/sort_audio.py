import os
import scipy.io.wavfile as wav 
import librosa
import soundfile
import numpy as np
from pipes import quote

from config import nn_config

from typing import List


def convert_mp3_to_wav(fname: str, sample_freq: int) -> str:

    curr = fname[-4:]
    if curr != '.mp3': return
    
    files = fname.split('/')

    orig_fname = files[-1][0:-4]
    orig_path = fname[0:-len(files[-1])]

    n_path = ''

    if fname[0] == '/':

        n_path = '/'

    for idx in range(len(files)-1):

        n_path += f'{files[idx]}/'

    tmp_path = f'{n_path}tmp'
    n_path += 'wave'

    if not os.path.exists(n_path):

        os.makedirs(n_path)

    if not os.path.exists(tmp_path):

        os.makedirs(tmp_path)

    fname_tmp = f'{tmp_path}/{orig_fname}.mp3'
    n_name = f'{n_path}/{orig_fname}.wav'

    sample_freq_str = "{0:1f}".format(float(sample_freq)/1000.0)
    cmd = f"lame -a -m m {quote(fname)} {quote(fname_tmp)}"
    os.system(cmd)

    cmd = f"lame --decode {quote(fname_tmp)} {quote(n_name)} --resample {sample_freq_str}"
    os.system(cmd)

    return n_name


def convert_flac_to_wav(fname: str, sample_freq: int) -> str:

    curr = fname[-5:]
    if curr != '.flac': return 

    files = fname.split('/')
    orig_fname = files[-1][0:-5]
    orig_path = fname[0:-len(files[-1])]

    n_path = ''

    if fname[0] == '/':

        n_path = '/'

    for idx in range(len(files-1)):

        n_path += f'{files[idx]}/'

    n_path += 'wave'

    if not os.path.exists(n_path):

        os.makedirs(n_path)
    
    n_name = f'{n_path}/{orig_fname}.wav'

    cmd = f"sox {quote(fname)} {quote(n_name)} channels 1 rate {sample_freq}"
    os.system(cmd)

    return n_name


def convert_folder_to_wav(directory: str, sample_rate: int = 44100) -> str:

    for f in os.listdir(directory):
        full_f = f'{directory}{f}'

        if f.endswith('.mp3'):

            convert_mp3_to_wav(fname=full_f, sample_freq=sample_rate)

        if f.endswith('.flac'):

            convert_flac_to_wav(fname=full_f, sample_freq=sample_rate)

    return f'{directory}wave/'


def read_wav_as_np(fname: str) -> (List[float], int):

    try:
        data = wav.read(fname)
    except:
        y, sr = librosa.load(fname)
        soundfile.write(fname, y, sr)
        data = wav.read(fname)

    np_arr = np.array(data[1].astype('float32') / 32767.0)
    
    return np_arr, data[0]

def write_np_as_wav(X: List[float] , sample_rate: int, fname: str) -> None:

    n_X = (X * 32767.0).astype('int16')
    wav.write(fname, sample_rate, n_X)

    return

def convert_np_audio_to_sample_blocks(song_np, block_size):

    block_list = []
    total_samples = song_np.shape[0]

    num_samples_so_far = 0

    while num_samples_so_far < total_samples:

        block = song_np[num_samples_so_far:num_samples_so_far+int(block_size)]

        if block.shape[0] < block_size:

            padding = np.zeros((block_size - block.shape[0], ))
            block = np.concatenate((block, padding))

        block_list.append(block)
        num_samples_so_far += block_size

    return block_list


def convert_sample_blocks_to_np_audio(blocks):

    song_np = np.concatenate(blocks)

    return song_np

def time_blocks_to_fft_blocks(blocks_time_domain):

    fft_blocks = []

    for block in blocks_time_domain:

        fft_block = np.fft.fft(block)
        n_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
        fft_blocks.append(n_block)

    return fft_blocks

def fft_blocks_to_time_blocks(blocks_ft_domain):

    time_blocks = []

    for block in blocks_ft_domain:

        num_elems = int(block.shape[0] / 2)
        real_chunk = block[0:num_elems]
        imag_chunk = block[num_elems:]
        n_block = real_chunk + 1.0 * imag_chunk

        time_block = np.fft.ifft(n_block)
        time_blocks.append(time_block)

    return time_blocks


def convert_wav_to_nptensor(directory, block_size, max_seq_len, out_file, max_files=20, useTimeDomain = False):

    files = []

    for f in os.listdir(directory):

        if f.endswith('.wav'):

            files.append(directory+f)

    chunks_X = []
    chunks_Y = []

    num_files = len(files)

    if num_files > max_files:

        num_files = max_files

    for idx in range(num_files):

        fi = files[idx]
        print(f'Processing: {idx+1}/{num_files}')
        print(f'File: {fi}')

        X,Y = load_training_example(fi, int(block_size), useTimeDomain=useTimeDomain)

        cur_seq = 0
        total_seq = len(X)

        print(total_seq)
        print(max_seq_len)

        while cur_seq + max_seq_len < total_seq:

            chunks_X.append(X[cur_seq:cur_seq+max_seq_len])
            chunks_Y.append(Y[cur_seq:cur_seq+max_seq_len])
            cur_seq += max_seq_len

    num_examples = len(chunks_X)
    num_dims_out = int(block_size) * 2

    if useTimeDomain:
        num_dims_out = int(block_size)

    out_shape = num_examples, max_seq_len, num_dims_out

    x_data = np.zeros(out_shape, dtype=int)
    y_data = np.zeros(out_shape, dtype=int)

    for n in range(num_examples):

        for i in range(max_seq_len):

            x_data[n][i] = chunks_X[n][i]
            y_data[n][i] = chunks_X[n][i]

        print(f'Saved Example: {n+1}/{num_examples}')

    print('Flushing to disk....')

    mean_x = np.mean(np.mean(x_data, axis=0, dtype=int), axis=0, dtype=int) #Mean across num examples and num timesteps
    print(mean_x)
    std_x = np.sqrt(np.mean(np.mean(np.abs(x_data-mean_x)**2, axis=0, dtype=int), axis=0, dtype=int)) # STD across num examples and num timesteps
    print(std_x)
    std_x = np.maximum(1.0e-8, std_x.astype(float)) #Clamp variance if too tiny
    std_x = std_x.astype(int)
    print(std_x)

    x_data[:][:] -= mean_x #Mean 0
    x_data[:][:] //= std_x #Variance 1
    y_data[:][:] -= mean_x #Mean 0
    y_data[:][:] //= std_x #Variance 1

    np.save(out_file+'_mean', mean_x)
    np.save(out_file+'_var', std_x)
    np.save(out_file+'_x', x_data)
    np.save(out_file+'_y', y_data)

    print('Done!')



def convert_nptensor_to_wav_files(tensor, indices, filename, useTimeDomain=False):

	num_seqs = tensor.shape[1]

	for i in indices:

		chunks = []
		for x in range(num_seqs):

			chunks.append(tensor[i][x])
		save_generated_example(filename+str(i)+'.wav', chunks,useTimeDomain=useTimeDomain)

def load_training_example(filename, block_size=2048, useTimeDomain=False):

	data, bitrate = read_wav_as_np(filename)
	x_t = convert_np_audio_to_sample_blocks(data, int(block_size))
	y_t = x_t[1:]
	y_t.append(np.zeros(block_size)) #Add special end block composed of all zeros

	if useTimeDomain:
		return x_t, y_t

	X = time_blocks_to_fft_blocks(x_t)
	Y = time_blocks_to_fft_blocks(y_t)

	return X, Y

def save_generated_example(filename, generated_sequence, useTimeDomain=False, sample_frequency=44100):

	if useTimeDomain:

		time_blocks = generated_sequence
	else:

		time_blocks = fft_blocks_to_time_blocks(generated_sequence)

	song = convert_sample_blocks_to_np_audio(time_blocks)
	write_np_as_wav(song, sample_frequency, filename)

	return

def audio_unit_test(filename, filename2):

	data, bitrate = read_wav_as_np(filename)
	time_blocks = convert_np_audio_to_sample_blocks(data, 1024)
	ft_blocks = time_blocks_to_fft_blocks(time_blocks)
	time_blocks = fft_blocks_to_time_blocks(ft_blocks)
	song = convert_sample_blocks_to_np_audio(time_blocks)
	write_np_as_wav(song, bitrate, filename2)
    
	return












    





