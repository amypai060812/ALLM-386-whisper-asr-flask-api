##############server_path.py##############

import re
import time
import logging
import argsparser
from flask_restx import *
from flask import *

gpu_device_id = 2

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from datasets import Audio

print('loading the model')
processor = WhisperProcessor.from_pretrained(
	"openai/whisper-large-v2",
	)
model = WhisperForConditionalGeneration.from_pretrained(
	"openai/whisper-large-v2",
	device_map = gpu_device_id,
	)

forced_decoder_ids_ar = processor.get_decoder_prompt_ids(
	language="arabic", 
	task="transcribe",
	)


forced_decoder_ids_en = processor.get_decoder_prompt_ids(
	language="english", 
	task="transcribe",
	)



print('loading example data')

## english

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]

speech_signal_array = list(sample["array"])[0:2048]
speech_sampling_rate = 16000

'''



transcript(
	speech_signal_array,
	speech_sampling_rate,
	)

# ' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.'

## arabic

ds = load_dataset("common_voice", "ar", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
input_speech = next(iter(ds))["audio"]

speech_signal_array = list(input_speech["array"])
speech_sampling_rate = 16000

transcript(
	speech_signal_array,
	speech_sampling_rate,
	language = 'ar'
	)

# ' ما أطول عودك'

'''

##

def transcript(
	speech_signal_array,
	speech_sampling_rate,
	language = 'en',
	):
	try:
		input_features = processor(
			speech_signal_array, 
			sampling_rate = speech_sampling_rate,
			return_tensors="pt").input_features.cuda(gpu_device_id)
		if language == 'ar':
			predicted_ids = model.generate(
				input_features,
				forced_decoder_ids = forced_decoder_ids_ar,
				)
		else:
			predicted_ids = model.generate(
				input_features,
				forced_decoder_ids = forced_decoder_ids_en,
				)
		transcription = processor.batch_decode(
			predicted_ids, 
			skip_special_tokens=True,
			)
		return transcription[0]
	except:
		return None

'''
[' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.']
'''

ns = Namespace(
	'tonomus_llm', 
	description='Tomonus LLM',
	)

args = argsparser.prepare_args()

#############

llama2_qa_parser = ns.parser()
llama2_qa_parser.add_argument('speech_signal_array', type=list, location='json')
llama2_qa_parser.add_argument('speech_sampling_rate', type=float, location='json')
llama2_qa_parser.add_argument('language', type=str, location='json')

llama2_qa_inputs = ns.model(
	'qa', 
		{
			'speech_signal_array': fields.List(fields.Float(), example = speech_signal_array),
			'speech_sampling_rate': fields.Float(example = 16000.0),
			'language': fields.String(example = 'en'),
		}
	)

@ns.route('/asr_wisper')
class llama2_qa_api(Resource):
	def __init__(self, *args, **kwargs):
		super(llama2_qa_api, self).__init__(*args, **kwargs)
	@ns.expect(llama2_qa_inputs)
	def post(self):		
		start = time.time()
		try:			
			args = llama2_qa_parser.parse_args()	

			output = {}
			output['transcription'] = transcript(
				speech_signal_array = args['speech_signal_array'],
				speech_sampling_rate = args['speech_sampling_rate'],
				language = args['language'],
				)
			output['transcription'] = output['transcription'].strip()
			output['status'] = 'success'
			output['running_time'] = float(time.time()- start)
			return output, 200
		except Exception as e:
			output = {}
			output['status'] = str(e)
			output['running_time'] = float(time.time()- start)
			return output

##############server_path.py##############