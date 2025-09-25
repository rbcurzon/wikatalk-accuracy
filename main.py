language_wer_metrics = {
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

from typing import Dict, Any

mt_bleu_metrics: Dict[str, Dict[str, Any]] = {
	"Tagalog": {
		"bleu_score": 34.07,
	},
	"Cebuano": {
		"bleu_score": 36.36,
	},
	"Hiligaynon": {
		"bleu_score": 35.27,
	},
	"Ilocano": {
		"bleu_score": 33.94,
	},
	"Pampanga": {
		"bleu_score": 22.22,
	},
	"Pangasinan": {
		"bleu_score": 37.56,
	},
	"Waray": {
		"bleu_score": 35.96,
	},        
	"Bikol": {
		"bleu_score": 38.69,
	},
	"Maguindanao": {
		"bleu_score": 26.66,
	},
	"Bisaya": {
		"bleu_score": 35.86,
	},
	"English": {
		"bleu_score": 39.03,
	},

}

def classify_accuracy(wer: float) -> str:
	assert wer >= 0.0, "WER should be a non-negative value."
 
	if wer <= 5.0:
		return "Excellent"
	elif wer <= 10.0:
		return "High"
	elif wer <= 25.0:
		return "Good"
	else:
		return "Poor"

def interpret_bleu(bleu: float) -> str:
	assert bleu >= 0.0, "BLEU score should be a non-negative value."
	
	if bleu < 10.0:
		return "ALMOST_USELESS"
	elif bleu < 20.0:
		return "HARD_TO_GET_GIST"
	elif bleu < 30.0:
		return "SIGNIFICANT_ERRORS"
	elif bleu < 40.0:
		return "UNDERSTANDABLE"
	else:
		return "HIGH_QUALITY"

interpretations_explanations = {
    "ALMOST_USELESS": "Almost useless",
    "HARD_TO_GET_GIST": "Hard to get the gist",
    "SIGNIFICANT_ERRORS": "The gist is clear, but has significant grammatical errors",
    "UNDERSTANDABLE": "Understandable to good translations",
    "HIGH_QUALITY": "High quality translations",
}

from typing import Union
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
	return {"message": "Welcome! This API provides evaluation metrics for Whisper Turbo, fine-tuned on Philippine languages. The models were trained using publicly available data from the web.",
		 "languages": list(language_wer_metrics.keys()),
		 "endpoints": {"/asr_metrics/": "View ASR performance metrics for all supported languages",
					   "/asr_accuracy/": "See languages grouped by accuracy levels",
					   "/mt_metrics/": "View MT performance metrics for all supported languages",
					   "/mt_interpretation/": "View MT interpretation for all supported languages"}
	}

@app.get("/asr_metrics/")
def read_asr_all_metrics():
	asr_wer_metrics_classified = {}
	for lang, metrics in language_wer_metrics.items():
		wer = metrics["word_error_rate"]
		accuracy = classify_accuracy(wer)
		asr_wer_metrics_classified[lang] = {
			"word_error_rate": wer,
			"accuracy": accuracy
		}

	asr_wer_metrics_classified = dict(sorted(asr_wer_metrics_classified.items(), key=lambda item: item[1]["word_error_rate"]))
	
	return {"language_metrics": asr_wer_metrics_classified}

@app.get("/asr_accuracy/")
def read_asr_accuracy():
	language_accuracy = {
		"Excellent": [],
		"Good": [],
		"High": [],
		"Poor": [],
	}
	for lang, metrics in language_wer_metrics.items():
		language_accuracy[metrics["accuracy"]].append(lang)
	
	return {"language_accuracy": language_accuracy}

@app.get("/mt_metrics/")
def read_mt_metrics():
	results = {}
	for lang, metrics in mt_bleu_metrics.items():
		bleu = metrics["bleu_score"]
		interpretation = interpret_bleu(bleu)
		results[lang] = {
			"bleu_score": bleu,
			"interpretation": interpretation
		}
	return {"results": results}

@app.get("/mt_interpretation/")
def read_mt_interpretation():
	interpretations = {
		"ALMOST_USELESS": [],
		"HARD_TO_GET_GIST": [],
		"SIGNIFICANT_ERRORS": [],
		"UNDERSTANDABLE": [],
		"HIGH_QUALITY": [],
	}
	for lang, metrics in mt_bleu_metrics.items():
		interpretation = interpret_bleu(metrics["bleu_score"])
		interpretations[interpretation].append(lang)

	return {"results": interpretations, "explanations": interpretations_explanations}