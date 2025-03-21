import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import re
from dotenv import load_dotenv
import numpy as np

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
LOCAL_MYSQL_USER = "root"
LOCAL_MYSQL_USER_PASSWORD = os.getenv("DB_PASSWORD")
LOCAL_MYSQL_PORT = 3306
LOCAL_MYSQL_DATABASE = "pokemon_database"

mysql_engine = MySQLDatabaseHandler(LOCAL_MYSQL_USER,LOCAL_MYSQL_USER_PASSWORD,LOCAL_MYSQL_PORT,LOCAL_MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

# Sample search, the LIKE operator in this case is hard-coded, 
# but if you decide to use SQLAlchemy ORM framework, 
# there's a much better and cleaner way to do this
def sql_search(episode):
    if True:
        return new_sql_search(episode)
    query_sql = f"""SELECT * FROM allcards WHERE LOWER( name ) LIKE '%%{episode.lower()}%%' limit 100"""
    keys = ["id","title","descr"]
    data = mysql_engine.query_selector(query_sql)
    return json.dumps([dict(zip(keys,[i[0],i[7],i[22]])) for i in data])

def sql_like(column, word):
    return f"""LOWER( """ + column + f""" ) LIKE '%%{word.lower()}%%'"""

def new_sql_search(query):
    format_query = query.lower().strip()
        
    query_sql = """SELECT * FROM allcards WHERE"""
    
    for word in re.split(r"[ -]", format_query):
        if len(word) > 1:
            query_sql = query_sql + " " + sql_like('name', word) + " OR "
        if len(word) > 2:
            query_sql = query_sql + " " + sql_like('rules', word) + " OR "
            query_sql = query_sql + " " + sql_like('attacks', word) + " OR "
    
    query_sql = query_sql[:-4] + " limit 1000"
    
    keys = ["id","title","descr"]
    data = mysql_engine.query_selector(query_sql)
    
    ranked_data = []
    for i in data:
        ranked_data.append((rank_simple(re.split(r"[ -]", format_query), i), i))
    ranked_data = sorted(ranked_data, reverse=True)
    
    available_names = {item[1][7]: True for item in ranked_data}
    i = 0
    while i < len(ranked_data):
        if available_names[ranked_data[i][1][7]]:
            available_names[ranked_data[i][1][7]] = False
            i+=1
        else:
            del ranked_data[i]
            
    for i in range(0, min(20, len(ranked_data))):
        if not (ranked_data[i][1][14] is None):
            if ranked_data[i][1][14] in [j[1][7] for j in ranked_data[:20]]:
                continue
            ranked_data.insert(i+1, get_prevolution(ranked_data[i][1][14], format_query))

    return json.dumps([dict(zip(keys,[i[1][0],str(i[0]) + " " + str(i[1][7]),i[1][17]])) for i in ranked_data[:20]])

def get_prevolution(name, format_query):
    name = name.replace("'", "\\'")
    query_sql = f"""SELECT * FROM allcards WHERE LOWER( name ) = '{name.lower()}' limit 50"""
    data = mysql_engine.query_selector(query_sql)
    ranked_data = []
    for i in data:
        ranked_data.append((rank_simple(re.split(r"[ -]", format_query), i), i))
    ranked_data = sorted(ranked_data, reverse=True)
    try:
        return ranked_data[0]
    except IndexError:
        raise IndexError(name)
    

def rank_simple(query: list, card: list):
    name_val = simple_jaccard(re.split(r"[ -]", card[7].lower()), query)
        
    card_type = card[10]
    
    if card_type == 'Energy':
        rules_score = -10
    elif card_type == 'Trainer':
        json_list = card[26].replace("':", "\":").replace("',", "\",").replace("{'","{\"").replace(" '", " \"").replace("['", "[\"")
        json_list = json_list.replace("']", "\"]").replace("'}", "\"}").replace("""\n""", """\\n""")
        try:
            rule_data = re.split(r"[ -]", " ".join(json.loads(json_list)).lower())
        except json.decoder.JSONDecodeError:
            print("===================================")
            print(card[26])
            print(json_list)
            print("===================================")
            raise ValueError("Error parsing Rules")
        rules_score = simple_jaccard(rule_data, query)
    else: # Pokemon
        if card[17] is None:
            return -10
        json_list = card[17].replace("\'n\'", "n").replace("':", "\":").replace("',", "\",").replace("{'","{\"").replace(" '", " \"").replace("['", "[\"")
        json_list = json_list.replace("']", "\"]").replace("'}", "\"}").replace(": None}", ": \"\"}")
        try:
            attacks = json.loads(json_list)
            for attack in attacks:
                if not "text" in attack:
                    attack["text"] = ""
            attack_data = re.split(r"[ -]", " ".join([i['text'].lower() for i in attacks]))
        except json.decoder.JSONDecodeError:
            print("===================================")
            print(card[17])
            print(json_list)
            print("===================================")
            raise ValueError("Error parsing Attacks")
        except KeyError:
            print("===================================")
            print(card[17])
            print(json_list)
            print("===================================")
            raise KeyError("Error parsing Attacks")
        rules_score = simple_jaccard(attack_data, query)
    
    return name_val + rules_score

def simple_jaccard(list1, list2):
    if list1 is None or list2 is None or len(list1) * len(list2) == 0:
        return 0
    words1 = set(list1)
    words2 = set(list2)
    return float(len(words1.intersection(words2))) / len(words1.union(words2))

def cos_similarity(vector1, vector2):
    numerator = np.dot(vector1, vector2)
    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    if denominator == 0:
        return 0.0
    else:
        return numerator / denominator
    
def type_cos_sim_search(selected_types):
    query_sql = "SELECT * FROM allcards"
    data = mysql_engine.query_selector(query_sql)
    data_list = list(data)
    
    type_list = ['Colorless', 'Fire', 'Water', 'Grass', 'Lightning', 'Psychic', 
                 'Fighting', 'Darkness', 'Metal', 'Fairy', 'Dragon']
    
    query_vec = np.zeros(len(type_list))

    for type_name in selected_types:
        if type_name in type_list:
            query_vec[type_list.index(type_name)] = 1
    
    ranked_data = []
    
    for row in data_list:
        card_id = row[0]
        name = row[7]
        supertype = row[10] 
        types_str = row[9] 
        
        if supertype not in ['Pokemon', 'Pokémon']:
            continue
        
        if not types_str:
            continue
            
        types = types_str.strip('[]').replace("'", "").split(',')
        types = [t.strip() for t in types]
        
        card_vec = np.zeros(len(type_list))

        for card_type in types:
            if card_type in type_list:
                card_vec[type_list.index(card_type)] = 1

        score = cos_similarity(query_vec, card_vec)
        
        if score > 0:
            ranked_data.append((score, card_id, name))
    
    ranked_data.sort(reverse=True)
    
    result = json.dumps([
        {
            "id": card_id,
            "title": name,
            "descr": f"Type Match Score: {score:.3f}"
        }
        for score, card_id, name in ranked_data[:20]
    ])
    
    return result


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return sql_search(text)

@app.route("/type_search")
def type_search():
    selected_types = request.args.getlist('types[]')
    return type_cos_sim_search(selected_types)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
