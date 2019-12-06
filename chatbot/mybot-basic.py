import spoonacular as sp
import json, requests, os
import aiml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from xml.etree import ElementTree as ET
import tensorflow as tf
from tensorflow.keras import models, backend, layers
import urllib.request
import numpy
import cv2

kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="chatbot/mybot-basic.xml")

model = models.load_model("cnn/model.h5")

APIkey = "13ffaf56553e48faab0c6ce7762f2e1c" 
spoon_api = sp.API(APIkey)

print("Welcome to food bot")


patterns = []
tree = ET.parse("chatbot/mybot-basic.xml")
all_pattern = tree.findall("*/pattern")
for pattern in all_pattern:
    if "*" not in pattern.text:
        patterns.append(pattern.text)

classnames = []
with open("cnn/food41/meta/meta/classes.txt") as reader:
    for line in reader:
        classnames.append(line.strip())

def get_similar(phrase):
    corpus = list(patterns)
    corpus.append(phrase)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity = list(cosine_similarity(tfidf_matrix, tfidf_matrix[-1]).flatten()[:-1])
    if max(similarity) < 0.1:
        print("Sorry I do not understand")
    else:
        similar_index = similarity.index(max(similarity))
        # print(max(similarity))
        # print(patterns[similar_index])
        handle_answer(kern.respond(patterns[similar_index]))

def handle_answer(answer):
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            return True
        elif cmd == 1:
            response = spoon_api.search_recipes_complex(params[1])
            data = response.json()
            if data["results"]:
                recipe_id = data["results"][0]["id"]
                response = spoon_api.get_recipe_information(recipe_id)
                data = response.json()
                print(data["sourceUrl"])
            else:
                print("There is no recipe for this")

        elif cmd == 2:
            response = spoon_api.get_a_random_food_joke()
            data = response.json()
            print(data["text"])
        
        elif cmd == 3:
            response = spoon_api.get_random_food_trivia()
            data = response.json()
            print(data["text"])
        
        elif cmd == 4:
            q = params[1]
            response = spoon_api.quick_answer("How much " + q)
            data = response.json()
            print(data["answer"])

        elif cmd == 5:
            q = params[1]
            name = 0
            for part in q.split(" "):
                if "jpg" in part or "png" in part:
                    name = part
            if name == 0:
                return "Input is neither a valid file nor url"
            if os.path.exists(name):
                npImage = cv2.imread(name)
            else:
                try:
                    filename = name.split("/")[-1]
                    urllib.request.urlretrieve(name, filename)
                    npImage = cv2.imread(filename)
                    os.remove(filename)
                except Exception as e:
                    print(e)
                    return "Input is neither a valid file nor url"
            rows = 100
            cols = 100
            color_bands = 3
            input_shape = (rows, cols, color_bands)
            scaledImage = cv2.resize(npImage, dsize=(rows, cols), interpolation=cv2.INTER_CUBIC)
            scaledImage = scaledImage/255
            imageArray = scaledImage.reshape(1, rows, cols, color_bands)

            predictions = model.predict(imageArray)
            
            print(classnames[numpy.argmax(predictions)])


        elif cmd == 99:
            get_similar(answer)

    else:
        print(answer)

while True:
    #get user input
    try:
        userInput = input("> ")
        userInput = userInput.split("?")[0]
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
        answer = answer.replace("  #99$", ".")

    if handle_answer(answer):
        break
