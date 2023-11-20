import os
from tqdm import tqdm
import math
from PIL import Image
import io
import numpy as np
import cv2
import zlib
import reedsolo
import array
import wave

def file_to_binary(file_name):
    # get file size
    file_size = os.path.getsize(file_name)

    # read file as binary and convert to string of 0's and 1's
    binary_string = ""
    with open(file_name, "rb") as f:
        for chunk in tqdm(iterable=iter(lambda: f.read(1024), b""), total=math.ceil(file_size/1024), desc="Converting to binary data", unit="KB"):
            binary_string += "".join(f"{byte:08b}" for byte in chunk)

    return binary_string

def binary_to_video(bin_string, output, width, height, pixel_size=4, fps=25):
    # Calculate the total number of pixels needed to represent the binary string
    num_pixels = len(bin_string)

    # Calculate the number of pixels that can fit in one image
    pixels_per_image = (width // pixel_size) * (height // pixel_size)

    # Calculate the number of images needed to represent the binary string
    num_images = math.ceil(num_pixels / pixels_per_image)

    # Create an array to store the frames
    frames = []

    # Loop through each image
    for i in tqdm(range(num_images), desc="Converting video frames"):
        # Calculate the range of binary digits needed for this image
        start_index = i * pixels_per_image
        end_index = min(start_index + pixels_per_image, num_pixels)
        binary_digits = bin_string[start_index:end_index]

        # Create a new image object with the given size
        img = Image.new('RGB', (width, height), color='white')

        # Loop through each row of binary digits
        for row_index in range(height // pixel_size):

            # Get the binary digits for the current row
            start_index = row_index * (width // pixel_size)
            end_index = start_index + (width // pixel_size)
            row = binary_digits[start_index:end_index]

            # Loop through each column of binary digits
            for col_index, digit in enumerate(row):

                # Determine the color of the pixel based on the binary digit
                if digit == '1':
                    color = (0, 0, 0)  # Black
                else:
                    color = (255, 255, 255)  # White

                # Calculate the coordinates of the pixel
                x1 = col_index * pixel_size
                y1 = row_index * pixel_size
                x2 = x1 + pixel_size
                y2 = y1 + pixel_size

                # Draw the pixel on the image
                img.paste(color, (x1, y1, x2, y2))

        # Add the frame to the list of frames
        with io.BytesIO() as f:
            img.save(f, format='PNG')
            frame = np.array(Image.open(f))
        frames.append(frame)

    # Create a video from the frames using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output, fourcc, fps, (width, height))

    for frame in tqdm(frames, desc="Writing video frames"):
        video.write(frame)
    
    video.release()

def binary_to_images(bin_string, output, width, height, pixel_size=4):
    # Calculate the total number of pixels needed to represent the binary string
    num_pixels = len(bin_string)

    # Calculate the number of pixels that can fit in one image
    pixels_per_image = (width // pixel_size) * (height // pixel_size)

    # Calculate the number of images needed to represent the binary string
    num_images = math.ceil(num_pixels / pixels_per_image)

    # Create output directory if it doesn't exist
    if not os.path.exists(output):
        os.makedirs(output)

    # Loop through each image
    for i in tqdm(range(num_images), desc="Converting images"):
        # Calculate the range of binary digits needed for this image
        start_index = i * pixels_per_image
        end_index = min(start_index + pixels_per_image, num_pixels)
        binary_digits = bin_string[start_index:end_index]

        # Create a new image object with the given size
        img = Image.new('RGB', (width, height), color='white')

        # Loop through each row of binary digits
        for row_index in range(height // pixel_size):

            # Get the binary digits for the current row
            start_index = row_index * (width // pixel_size)
            end_index = start_index + (width // pixel_size)
            row = binary_digits[start_index:end_index]

            # Loop through each column of binary digits
            for col_index, digit in enumerate(row):

                # Determine the color of the pixel based on the binary digit
                if digit == '1':
                    color = (0, 0, 0)  # Black
                else:
                    color = (255, 255, 255)  # White

                # Calculate the coordinates of the pixel
                x1 = col_index * pixel_size
                y1 = row_index * pixel_size
                x2 = x1 + pixel_size
                y2 = y1 + pixel_size

                # Draw the pixel on the image
                img.paste(color, (x1, y1, x2, y2))

        # Save the image
        img.save(f"{output}/{i}.png")

def extract_frames(input):
    am = []

    # Open the video file
    vid = cv2.VideoCapture(input)

    # Get the total number of frames in the video
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use tqdm to create a progress bar
    with tqdm(total=num_frames, desc="Extracting video frames") as pbar:
        # Iterate over every frame of the video
        while True:
            ret, frame = vid.read()
            if not ret:
                break

            # Append the frame to the list
            am.append(frame)

            # Update the progress bar
            pbar.update(1)

    # Release the video capture object
    vid.release()

    # Return the list of frames
    return am

def process_images(frames, pixel_size=4):
    # Define the threshold value for determining black pixels
    threshold = 128

    # Create an empty string to store the binary digits
    binary_digits = ''

    # Loop through each frame in the list
    for frame in tqdm(frames, desc="Processing frames"):
        # Convert the frame to grayscale
        gray_frame = np.mean(frame, axis=2).astype(np.uint8)

        # Loop through each row of pixels
        for y in range(0, gray_frame.shape[0], pixel_size):
            # Loop through each column of pixels
            for x in range(0, gray_frame.shape[1], pixel_size):
                # Get the color of the current pixel
                color = gray_frame[y:y+pixel_size, x:x+pixel_size]

                # Determine the binary digit based on the color of the pixel
                if color.mean() < threshold:
                    binary_digits += '1'
                else:
                    binary_digits += '0'

    # Store the binary string in a single text file
    return binary_digits

def binary_to_file(output, binary_filename):
    # convert binary string to binary data
    binary_data = bytes(int(binary_filename[i:i+8], 2)
                        for i in range(0, len(binary_filename), 8))

    # write binary data to output file
    with open(output, "wb") as f:
        with tqdm(total=len(binary_data), unit='B', unit_scale=True, desc="Writing binary data") as pbar:
            for chunk in range(0, len(binary_data), 1024):
                f.write(binary_data[chunk:chunk+1024])
                pbar.update(1024)

        print(f"Binary data converted to example_reverse.zip")

def get_images(directory):
    image_list = []

    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # Use tqdm to create a progress bar
    with tqdm(total=len(image_files), desc="Reading images") as pbar:
        # Iterate over every image in the directory
        for image_file in image_files:
            # Construct the full path to the image
            image_path = os.path.join(directory, image_file)

            # Read the image using OpenCV
            frame = cv2.imread(image_path)

            # Append the frame to the list
            image_list.append(frame)

            # Update the progress bar
            pbar.update(1)

    # Return the list of frames
    return image_list

def compress(data):
    # Compress the data using zlib
    compressed_data = zlib.compress(data)

    return compressed_data

def decompress(data):
    # Decompress the data using zlib
    decompressed_data = zlib.decompress(data)

    return decompressed_data

def reedsolomon_encode(data):
    # Create a Reed-Solomon encoder object
    rs = reedsolo.RSCodec(10)

    # Encode the data
    encoded_data = rs.encode(data)

    return encoded_data

def reedsolomon_decode(data):
    # Create a Reed-Solomon decoder object
    rs = reedsolo.RSCodec(10)

    # Decode the data
    decoded_data = rs.decode(data)

    return decoded_data

def calculate_required_audio_length(binary_hash, bit_duration=0.1, sample_rate=44100):
    # Calculate the number of samples per bit
    samples_per_bit = int(sample_rate * bit_duration)

    # Calculate the total number of samples needed for the entire binary hash
    total_samples = len(binary_hash) * samples_per_bit

    # Calculate the total time duration in seconds
    total_time_seconds = total_samples / sample_rate

    return total_time_seconds

def embed_hash_into_audio_fsk(audio_data, binary_hash, frequency_0, frequency_1, sample_rate=44100, bit_duration=0.1):
    # Calculate the number of samples per bit
    samples_per_bit = int(sample_rate * bit_duration)

    # Create an array to store the modified audio data
    modified_audio_data = array.array('h', [0] * len(audio_data))

    # Embed the binary hash into the audio data using FSK
    for i in tqdm(range(len(binary_hash)), desc="Embedding hash into audio"):
        frequency = frequency_1 if binary_hash[i] == '1' else frequency_0

        for t in range(samples_per_bit):
            modified_audio_data[i * samples_per_bit + t] = int(32767.0 * np.sin(2.0 * np.pi * frequency * t / sample_rate))

    return modified_audio_data.tobytes()

def calculate_required_audio_length(binary_hash, bit_duration=0.1, sample_rate=44100):
    # Calculate the number of samples per bit
    samples_per_bit = int(sample_rate * bit_duration)

    # Calculate the total number of samples needed for the entire binary hash
    total_samples = len(binary_hash) * samples_per_bit

    # Calculate the total time duration in seconds
    total_time_seconds = total_samples / sample_rate

    return total_time_seconds / 2

def generate_silence(duration_seconds, sample_rate=44100):
    # Calculate the number of samples for the specified duration
    num_samples = int(duration_seconds * sample_rate)

    # Create an array of zeros to represent silence
    silence_data = array.array('h', [0] * num_samples)

    return silence_data.tobytes()

def hex_encode(string):
    return ''.join([hex(ord(c))[2:] for c in string])

def hed_decode(string):
    return ''.join([chr(int(string[i:i+2], 16)) for i in range(0, len(string), 2)])

# Save the modified audio data to a new audio file
def save_audio_file(modified_audio_data, output_audio_file_path, sample_width=2, sample_rate=44100):
    with wave.open(output_audio_file_path, 'wb') as output_wave_file:
        output_wave_file.setsampwidth(sample_width)
        output_wave_file.setframerate(sample_rate)
        output_wave_file.setnframes(len(modified_audio_data) // sample_width)
        output_wave_file.setnchannels(1)  # Mono audio
        output_wave_file.writeframes(modified_audio_data)

def to_binary_string(string):
    return ''.join(format(int(c, 16), '04b') for c in string)

def decode_audio_fsk(audio_data, frequency_0, frequency_1, sample_rate=44100, bit_duration=0.1):
    samples_per_bit = int(sample_rate * bit_duration)
    binary_hash = ''

    # Convert bytes to NumPy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    for i in tqdm(range(0, len(audio_array), samples_per_bit), desc="Decoding audio"):
        chunk = audio_array[i:i+samples_per_bit]

        # Ensure chunk is one-dimensional
        if len(chunk.shape) > 1:
            chunk = chunk[:, 0]

        fft_values = np.fft.fft(chunk.real)
        freq = np.fft.fftfreq(len(chunk), d=1/sample_rate)

        # Find the dominant frequency in the chunk
        dominant_freq = freq[np.argmax(np.abs(fft_values))]

        # If the dominant frequency is closer to frequency_1 than frequency_0, the bit is '1', otherwise it is '0'
        if abs(dominant_freq) < (frequency_0 + frequency_1) / 2:
            binary_hash += '0'
        else:
            binary_hash += '1'

    return binary_hash

# Read the modified audio file
def read_audio_file(input_audio_file_path):
    with wave.open(input_audio_file_path, 'rb') as wave_file:
        sample_width = wave_file.getsampwidth()
        sample_rate = wave_file.getframerate()
        audio_data = wave_file.readframes(wave_file.getnframes())

    return sample_width, sample_rate, audio_data

def binary_to_hex(binary_string):
    return hex(int(binary_string, 2))[2:]