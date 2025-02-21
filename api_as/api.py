import torch
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from fastapi.middleware.cors import CORSMiddleware
from collections import defaultdict
from typing import List, Dict

# FastAPI and CORS Setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Model running on {device}")

# Linguistic Analysis Constants
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "should", "now"
}

IDIOMS = {
    "cold shoulder": {"meaning": "deliberate disregard", "emotions": ["rejection", "sadness"], "scenario": "emotional",
                      "weight": 4.5},
    "piece of cake": {"meaning": "easy task", "emotions": ["approval", "joy"], "scenario": "social", "weight": 3.8},
    "break the ice": {"meaning": "initiate conversation", "emotions": ["excitement", "anticipation"],
                      "scenario": "social", "weight": 4.0},
    "spill the beans": {"meaning": "reveal secret", "emotions": ["surprise", "disappointment"], "scenario": "social",
                        "weight": 3.5},
    "raining cats and dogs": {"meaning": "heavy rain", "emotions": ["annoyance", "discomfort"], "scenario": "travel",
                              "weight": 3.2},
}

IRONY_PATTERNS = {
    r"as fun as a root canal": {"weight": 4.5, "emotions": ["sarcasm", "disappointment"]},
    r"clear as mud": {"weight": 4.0, "emotions": ["confusion", "frustration"]},
    r"about as useful as a (chocolate teapot|screen door on a submarine)": {"weight": 4.2,
                                                                            "emotions": ["annoyance", "anger"]},
    r"([nN]ow that's )?([aA]wesome|[gG]reat)(... said no one ever)": {"weight": 4.8,
                                                                      "emotions": ["sarcasm", "disapproval"]},
}

compiled_irony_patterns = {re.compile(k): v for k, v in IRONY_PATTERNS.items()}


def detect_linguistic_features(text: str) -> Dict[str, List]:
    text_lower = text.lower()
    features = {
        "idioms": [],
        "irony": [],
        "context_boost": 0.0
    }

    # Detect idioms
    for idiom, data in IDIOMS.items():
        if idiom in text_lower:
            features["idioms"].append({
                "phrase": idiom,
                "meaning": data["meaning"],
                "data": data
            })
            features["context_boost"] += data["weight"] * 0.3

    # Detect irony
    for pattern, data in compiled_irony_patterns.items():
        if pattern.search(text_lower):
            features["irony"].append({
                "pattern": pattern.pattern,
                "data": data
            })
            features["context_boost"] += data["weight"] * 0.5

    return features


def enhanced_extract_context(text: str) -> str:
    words = re.findall(r'\b\w+(?:-\w+)*\b', text.lower())
    keywords = [
        word for word in words
        if word not in STOP_WORDS and len(word) > 2
    ]
    bigrams = [' '.join(words[i:i + 2]) for i in range(len(words) - 1)]
    keywords += bigrams
    unique_keywords = list(set(keywords))[:7]

    features = detect_linguistic_features(text)
    context_addons = []

    if features["idioms"]:
        idiom_meanings = [f"{i['phrase']} ({i['meaning']})" for i in features["idioms"]]
        context_addons.append("Idioms: " + ", ".join(idiom_meanings))

    if features["irony"]:
        irony_flags = [i["pattern"] for i in features["irony"]]
        context_addons.append("Irony detected: " + ", ".join(irony_flags))

    base_context = ", ".join(unique_keywords) if unique_keywords else "general context"
    return f"{base_context} | {' | '.join(context_addons)}" if context_addons else base_context


def enhanced_detect_scenario(text: str, emotions: dict) -> str:
    scenario_weights = {
        "health": {"keywords": {"hungry": 3.0, "pain": 4.0, "sick": 4.5}, "emotion_map": {"fear": 1.2}},
        "emotional": {"keywords": {"heartbroken": 4.5, "lonely": 4.0}, "emotion_map": {"sadness": 1.5}},
        "travel": {"keywords": {"travel": 3.0, "trip": 3.5}, "emotion_map": {"joy": 1.3}},
        "social": {"keywords": {"party": 3.2, "wedding": 3.8}, "emotion_map": {"joy": 1.1}},
        "gift": {"keywords": {"gift": 4.0, "present": 4.5}, "emotion_map": {"joy": 1.4}},
        "entertainment": {"keywords": {"amusement": 4.0, "park": 3.5}, "emotion_map": {"joy": 1.5}},
    }

    # Original detection logic
    words = re.findall(r'\b\w+\b', text.lower())
    word_scores = defaultdict(float)
    for word in words:
        for scenario, data in scenario_weights.items():
            if word in data["keywords"]:
                word_scores[scenario] += data["keywords"][word]

    # Emotion boost
    emotion_boost = defaultdict(float)
    for scenario, data in scenario_weights.items():
        for emotion, boost in data["emotion_map"].items():
            if emotion in emotions:
                emotion_boost[scenario] += emotions[emotion] * boost

    # Metaphor/idiom boost
    features = detect_linguistic_features(text)
    for idiom in features["idioms"]:
        scenario = idiom["data"]["scenario"]
        word_scores[scenario] += idiom["data"]["weight"]

    total_scores = {
        scenario: word_scores.get(scenario, 0) + emotion_boost.get(scenario, 0)
        for scenario in scenario_weights
    }

    max_score = max(total_scores.values(), default=0)
    if max_score < 5.0:
        return "general"

    return max(total_scores, key=total_scores.get)


class MessageRequest(BaseModel):
    text: str


def is_greeting(text: str) -> bool:
    greetings = {"hello", "hi", "hey", "hola", "bonjour", "ciao", "privet", "greetings"}
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    return len(set(cleaned.split()) & greetings) > 0


def generate_enhanced_advice(text: str, emotions: dict) -> List[str]:
    try:
        context = enhanced_extract_context(text)
        scenario = enhanced_detect_scenario(text, emotions)
        features = detect_linguistic_features(text)

        prompt_template = f"""Generate TWO {scenario} suggestions considering:
        - Input: {text}
        - Emotions: {', '.join([f"{k}({v:.2f})" for k, v in emotions.items()])}
        - Context: {context}

        Requirements:
        - Address any detected idioms/irony
        - Provide culturally appropriate actions
        - Suggest practical steps

        Examples:"""

        if scenario == "emotional":
            prompt_template += "\n1. Write unsent letters\n2. Practice mindfulness"
        elif scenario == "social":
            prompt_template += "\n1. Plan low-pressure activities\n2. Use ice-breakers"
        else:
            prompt_template += "\n1. Break into steps\n2. Set success criteria"

        prompt_template += "\n\nSuggestions:"

        inputs = tokenizer(prompt_template, return_tensors="pt", max_length=512, truncation=True).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.75,
            top_p=0.92,
            repetition_penalty=1.6,
            num_return_sequences=1
        )

        raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Split at the "Suggestions:" marker and take the new content
        generated_part = raw_text.split("Suggestions:")[-1]

        # Extract numbered suggestions from generated content only
        suggestions = [s.strip() for s in re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$)', generated_part)]

        return suggestions[:2] if suggestions else ["Consider different perspectives", "Break into manageable steps"]

    except Exception as e:
        print(f"Advice error: {str(e)}")
        return ["Try reframing positively", "Identify actionable steps"]


# Sentiment and Emotion Models
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if device != torch.device("cpu") else -1
)

emotion_analyzer = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=5,
    device=0 if device != torch.device("cpu") else -1
)

# GPT-2 Configuration fakirlikten bunu kullanıyorum
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large", padding_side="left")
model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device)
tokenizer.pad_token = tokenizer.eos_token


@app.post("/analyze")
async def analyze_message(request: MessageRequest):
    try:
        text = request.text[:512].strip()

        if is_greeting(text):
            return {
                "sentiment": {"label": "NEUTRAL", "score": 1.0},
                "emotions": {"happy": 1.0},
                "advice": ["Hello! Let's explore your thoughts together. How can I help?"],
                "linguistic_features": []
            }

        # Sentiment analysis
        sentiment = sentiment_analyzer(text)[0]

        # Emotion analysis
        base_emotions = emotion_analyzer(text)[0]
        features = detect_linguistic_features(text)

        # Emotion processing
        emotion_profile = defaultdict(float)
        for emotion in base_emotions:
            emotion_profile[emotion['label']] = emotion['score']

        for idiom in features["idioms"]:
            for emotion in idiom['data']['emotions']:
                emotion_profile[emotion] += 0.15

        for irony in features["irony"]:
            for emotion in irony['data']['emotions']:
                emotion_profile[emotion] += 0.25

        # Normalize
        total = sum(emotion_profile.values())
        top_emotions = {k: round(v / total, 3) for k, v in sorted(
            emotion_profile.items(), key=lambda x: x[1], reverse=True
        )[:3]}

        # Generate advice
        advice = generate_enhanced_advice(text, top_emotions)

        return {
            "sentiment": {
                "label": sentiment['label'].upper(),
                "score": round(sentiment['score'], 3)
            },
            "emotions": top_emotions,
            "advice": advice,
            "linguistic_features": {
                "idioms": [i['phrase'] for i in features["idioms"]],
                "irony_patterns": [i['pattern'] for i in features["irony"]]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))










