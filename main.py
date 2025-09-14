language_metrics = {
	"Bikol": {
		"word_error_rate": 10.70,
		"accuracy": "Good",
	},
	"Cebuano": {
		"word_error_rate": 12.77,
		"accuracy": "Good",
	},
	"Hiligaynon": {
		"word_error_rate": 4.44,
		"accuracy": "Excellent",
	},
	"Ilocano": {
		"word_error_rate": 7.73,
		"accuracy": "High",
	},
	"Maranao": {
		"word_error_rate": 11.39,
		"accuracy": "Good",
	},
	"Pangasinan": {
		"word_error_rate": 8.69,
		"accuracy": "High",
	},
	"Tagalog": {
		"word_error_rate": 12.16,
		"accuracy": "Good",
	},
	"Waray": {
		"word_error_rate": 15.05,
		"accuracy": "Good",
	},
	"Pampanga": {
		"word_error_rate": 11.68,
		"accuracy": "Good",
	},
	"Bisaya": {
		"word_error_rate": 8.31,
		"accuracy": "High",
	},
}

from typing import Union
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome! This API provides evaluation metrics for Whisper Turbo, fine-tuned on Philippine languages. The models were trained using publicly available data from the web.",
         "languages": list(language_metrics.keys()),
         "endpoints": {"/asr_metrics/": "View ASR performance metrics for all supported languages",
                       "/asr_accuracy/": "See languages grouped by accuracy levels"}}

@app.get("/asr_metrics/")
def read_all_metrics():
    return {"language_metrics": language_metrics}

@app.get("/asr_accuracy/")
def read_accuracy():
    language_accuracy = {
		"Excellent": [],
		"Good": [],
		"High": [],
		"Poor": [],
	}
    for lang, metrics in language_metrics.items():
        language_accuracy[metrics["accuracy"]].append(lang)
    return {"language_accuracy": language_accuracy}
