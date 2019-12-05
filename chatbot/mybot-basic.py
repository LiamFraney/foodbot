import spoonacular as sp
import json, requests
import aiml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from xml.etree import ElementTree as ET
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="C:/Users/LiamF/Documents/FinalYear/ArtificialIntelligence/Lab2/mybot-basic.xml")

APIkey = "13ffaf56553e48faab0c6ce7762f2e1c" 
spoon_api = sp.API(APIkey)

print("Welcome to food bot")


patterns = []
tree = ET.parse("mybot-basic.xml")
all_pattern = tree.findall("*/pattern")
for pattern in all_pattern:
    if "*" not in pattern.text:
        patterns.append(pattern.text)

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

        elif cmd == 4:
                q = params[1]
                

        elif cmd == 99:
            get_similar(answer)

    else:
        print(answer)

while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)

    if handle_answer(answer):
        break
