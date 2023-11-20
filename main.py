import fire

from functions import file_to_binary, binary_to_video, extract_frames, process_images, binary_to_file, binary_to_images, get_images, calculate_required_audio_length, generate_silence, embed_hash_into_audio_fsk, save_audio_file, read_audio_file, decode_audio_fsk

class InfiniteStorageYoutube(object):#
    """Infinite Storage Youtube"""

    def filetovideo(self, input, output, width=640, height=480, fps=25, pixel_size=4):
        # Print the config
        print('Input file: {}'.format(input))
        print('Output file: {}'.format(output))
        print('Width: {}'.format(width))
        print('Height: {}'.format(height))
        print('FPS: {}'.format(fps))
        print('Pixel size: {}'.format(pixel_size))

        binary_string = file_to_binary(input)

        binary_to_video(binary_string, output, width=width, height=height, pixel_size=pixel_size, fps=fps)

        return output

    def videotofile(self, input, output, width=640, height=480, fps=25, pixel_size=4):
        # Print the config
        print('Input file: {}'.format(input))
        print('Output file: {}'.format(output))
        print('Width: {}'.format(width))
        print('Height: {}'.format(height))
        print('FPS: {}'.format(fps))
        print('Pixel size: {}'.format(pixel_size))

        am = extract_frames(input)

        binary_digits = process_images(am, pixel_size=pixel_size)

        binary_to_file(output, binary_digits)

        return output
    
    def filetoimages(self, input, output, width=640, height=480, pixel_size=4):
        # Print the config
        print('Input file: {}'.format(input))
        print('Output directory: {}'.format(output))
        print('Width: {}'.format(width))
        print('Height: {}'.format(height))
        print('Pixel size: {}'.format(pixel_size))

        binary_string = file_to_binary(input)

        binary_to_images(binary_string, output, width=width, height=height, pixel_size=pixel_size)

        return output

    def imagestofile(self, input, output, width=640, height=480, pixel_size=4):
        # Print the config
        print('Input directory: {}'.format(input))
        print('Output file: {}'.format(output))
        print('Width: {}'.format(width))
        print('Height: {}'.format(height))
        print('Pixel size: {}'.format(pixel_size))

        am = get_images(input)

        binary_digits = process_images(am, pixel_size=pixel_size)

        binary_to_file(output, binary_digits)

        return output
    
    def filetoaudio(self, input, output, frequency_0=1000, frequency_1=2000, sample_rate=44100, bit_duration=0.1):
        # Print the config
        print('Input file: {}'.format(input))
        print('Output file: {}'.format(output))
        print('Frequency 0: {}'.format(frequency_0))
        print('Frequency 1: {}'.format(frequency_1))
        print('Sample rate: {}'.format(sample_rate))

        binary_string = file_to_binary(input)

        duration_seconds = calculate_required_audio_length(binary_string, sample_rate=sample_rate, bit_duration=bit_duration)

        audio_data = generate_silence(duration_seconds, sample_rate=sample_rate)

        modified_audio_data = embed_hash_into_audio_fsk(audio_data, binary_string, frequency_0=frequency_0, frequency_1=frequency_1, sample_rate=sample_rate, bit_duration=bit_duration)

        save_audio_file(modified_audio_data, output, sample_rate=sample_rate)

        return output

    def audiotofile(self, input, output, frequency_0=1000, frequency_1=2000, bit_duration=0.1):
        # Print the config
        print('Input file: {}'.format(input))
        print('Output file: {}'.format(output))
        print('Frequency 0: {}'.format(frequency_0))
        print('Frequency 1: {}'.format(frequency_1))
        print('Bit duration: {}'.format(bit_duration))

        sample_width, sample_rate, audio_data = read_audio_file(input)

        binary_digits = decode_audio_fsk(audio_data, frequency_0=frequency_0, frequency_1=frequency_1, bit_duration=bit_duration, sample_rate=sample_rate)

        binary_to_file(output, binary_digits)

        return output

if __name__ == '__main__':
  fire.Fire(InfiniteStorageYoutube)