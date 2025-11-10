import json, random, os

os.makedirs("outputs", exist_ok=True)

emotions = ["angry","disgust","fear","happy","sad","surprise","neutral"]
templates = {
    "angry": ["User looked irritated.", "Frustrated by delay."],
    "disgust": ["User seemed displeased.", "Expression showed rejection."],
    "fear": ["User looked anxious.", "Uneasy during the task."],
    "happy": ["User smiled warmly.", "Positive and cheerful."],
    "sad": ["User looked disappointed.", "Low energy noted."],
    "surprise": ["User appeared astonished.", "Unexpected reaction."],
    "neutral": ["User stayed calm.", "No visible emotion."]
}

try:
    preds = json.load(open("outputs/predictions.json"))["preds"]
except:
    preds = [0,1,2,3,4,5,6]*3 

reviews = [random.choice(templates[emotions[p % len(emotions)]]) for p in preds]
with open("outputs/reviews.json","w") as f:
    json.dump(reviews,f,indent=2)

print(f"✅ Saved {len(reviews)} synthetic reviews to outputs/reviews.json")
