from deepface import DeepFace

obj = DeepFace.analyze(img_path = "https://firebasestorage.googleapis.com/v0/b/moody-255ba.appspot.com/o/posts%2Ftonyjiang02%40gmail.com%2Ftony.jpeg?alt=media&token=4ebeb1bc-67ca-45ba-85b5-b2b97b68aaec", actions = ['emotion'])
print(obj.get("dominant_emotion"))

# from textblob import TextBlob
# print(TextBlob('I am sad').sentiment.polarity)