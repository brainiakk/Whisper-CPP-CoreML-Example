import os
import subprocess
import time
import wave
import sox
import speech_recognition as sr

class VoiceService:
    
    def __init__(self) -> None:
        self._output_dir = "outputs"
        os.makedirs(self._output_dir, exist_ok=True)
    
    def listen(self):
        print('Listening (mode: offline)...')
        try:
            transcribed_file = f"{self._output_dir}/transcribed-audio.wav"
            # args = ['-d', '-r',  '16000', '-c', '1', '-b', '16', transcribed_file, 'silence' ,'1' , '0.1', '3%' , '1', '1.0', '3%']
            # sox.core.sox(args)
            r = sr.Recognizer()
            with sr.Microphone(sample_rate=16000) as source:
                print("Say something!")
                audio = r.listen(source)

            # write audio to a RAW file
            with open(transcribed_file, "wb") as f:
                f.write(audio.get_wav_data())
                
            start = time.time()
            output = self.process_audio(transcribed_file)
            end = time.time()
            print('⏰ Listen Offline Transcription in %s seconds' % (end - start))
            # os.remove(transcribed_file)
            return output 
        except Exception as e:
            print(f"Error while listening: {e}")
            return False


    def process_audio(self, wav_file, model_name="base.en"):
        """
        Processes an audio file using a specified model and returns the processed string.

        :param wav_file: Path to the WAV file
        :param model_name: Name of the model to use
        :return: Processed string output from the audio processing
        :raises: Exception if an error occurs during processing
        """

        model = f"modules/whisper.cpp/models/ggml-{model_name}.bin"

        # Check if the file exists
        if not os.path.exists(model):
            raise FileNotFoundError(f"Model file not found: {model} \n\nDownload a model with this command:\n\n> bash ./models/download-ggml-model.sh {model_name}\n\n")

        if not os.path.exists(wav_file):
            raise FileNotFoundError(f"WAV file not found: {wav_file}")

        full_command = f"modules/whisper.cpp/main -m {model} -f {wav_file} -np -nt"

        # Execute the command
        process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Get the output and error (if any)
        output, error = process.communicate()

        if error:
            raise Exception(f"Error processing audio: {error.decode('utf-8')}")

        # Process and return the output string
        decoded_str = output.decode('utf-8').strip()
        processed_str = decoded_str.replace('[BLANK_AUDIO]', '').strip()

        return processed_str
