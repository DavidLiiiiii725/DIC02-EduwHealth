from transformers import pipeline

class EmotionDetector:
    def __init__(self):
        print("[APU] Loading emotion model... (first time may take a while)")
        self.model = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        print("[APU] Emotion model loaded.")

    def detect(self, text):
        raw = self.model(text)
        # Newer versions of transformers return a nested list: [[{...}, ...]]
        # Older versions return a flat list: [{...}, ...]
        scores = raw[0] if isinstance(raw[0], list) else raw
        return {s["label"]: s["score"] for s in scores}