from deepface import DeepFace

# obj = DeepFace.analyze(img_path = "/Users/Anan/LAHacks/tony.jpeg", actions = ['emotion'])
# print(obj.get("dominant_emotion"))

from textblob import TextBlob
print(TextBlob('I am sad').sentiment.polarity)