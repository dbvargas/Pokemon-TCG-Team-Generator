import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import re
import requests
from dotenv import load_dotenv
import numpy as np
from sqlalchemy import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, normalize
from scipy.sparse.linalg import svds

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

    return json.dumps([dict(zip(keys,[i[1][0],str(i[1][7]),i[0]])) for i in ranked_data[:20]])

def search_trainer_cards(query):
    query = query.lower().strip()
    words = re.split(r"[ -]", query)
    
    base_sql = "SELECT id, name, supertype, rules FROM allcards WHERE LOWER(supertype) = 'trainer' AND ("
    
    filters = []
    for word in words:
        if len(word) > 1:
            filters.append(sql_like('name', word))
        if len(word) > 2:
            filters.append(f"(rules IS NOT NULL AND {sql_like('rules', word)})")
    
    if not filters:
        return json.dumps([])

    query_sql = base_sql + " OR ".join(filters) + ") LIMIT 20"
    print("TRAINER SQL:", query_sql)

    data = mysql_engine.query_selector(query_sql)

    return json.dumps([
        {
            "id": row[0],
            "title": row[1],
            "descr": row[3] or "No description"
        }
        for row in data
    ])

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

def create_deck_table():
    query = """
    CREATE TABLE IF NOT EXISTS decks (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255) NOT NULL,
        cards JSON NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    mysql_engine.query_executor(query)

create_deck_table()

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/type_search")
def type_search():
    selected_types = request.args.getlist('types[]')
    return type_cos_sim_search(selected_types)

@app.route("/search_trainers")
def search_trainers():
    query = request.args.get("title", "")
    return search_trainer_cards(query)

@app.route("/save_deck", methods=['POST'])
def save_deck():
    try:
        data = request.get_json()
        deck_name = data.get('name', 'Untitled Deck')
        cards = json.dumps(data.get('cards', []))
        
        query = text("INSERT INTO decks (name, cards) VALUES (:name, :cards)")
        mysql_engine.query_executor(query.bindparams(name=deck_name, cards=cards))
        
        return json.dumps({"success": True, "message": "Deck saved successfully"})
    except Exception as e:
        return json.dumps({"success": False, "message": str(e)}), 500

@app.route("/get_decks", methods=['GET'])
def get_decks():
    try:
        query = text("SELECT * FROM decks ORDER BY created_at DESC")
        results = mysql_engine.query_selector(query)
        decks = []
        for row in results:
            decks.append({
                "id": row[0],
                "name": row[1],
                "cards": json.loads(row[2]),
                "created_at": row[3].isoformat() if row[3] else None
            })
        return json.dumps({"success": True, "decks": decks})
    except Exception as e:
        return json.dumps({"success": False, "message": str(e)}), 500

@app.route("/export_deck/<int:deck_id>", methods=['GET'])
def export_deck(deck_id):
    try:
        query = text("SELECT * FROM decks WHERE id = :deck_id")
        results = mysql_engine.query_selector(query.bindparams(deck_id=deck_id))
        deck = None
        for row in results:
            deck = {
                "id": row[0],
                "name": row[1],
                "cards": json.loads(row[2]),
                "created_at": row[3].isoformat() if row[3] else None
            }
        
        if not deck:
            return json.dumps({"success": False, "message": "Deck not found"}), 404
            
        return json.dumps({"success": True, "deck": deck})
    except Exception as e:
        return json.dumps({"success": False, "message": str(e)}), 500

@app.route("/delete_deck/<int:deck_id>", methods=['DELETE'])
def delete_deck(deck_id):
    try:
        query = text("DELETE FROM decks WHERE id = :deck_id")
        mysql_engine.query_executor(query.bindparams(deck_id=deck_id))
        return json.dumps({"success": True, "message": "Deck deleted successfully"})
    except Exception as e:
        return json.dumps({"success": False, "message": str(e)}), 500

@app.route("/card_details/<card_id>")
def get_card_details(card_id):
    query_sql = f"""SELECT * FROM allcards WHERE id = '{card_id}'"""
    print(f"Executing query: {query_sql}")  # Debug log
    data = list(mysql_engine.query_selector(query_sql))  # Convert to list
    if data:
        row = data[0]
        print(f"Full row data: {row}")  # Debug log
        
        # HP is at index 13 for Pokemon cards
        hp = row[13] if row[13] else "N/A"
        types_str = row[9] if row[9] else "[]"
        types = types_str.strip('[]').replace("'", "").split(',')
        types = [t.strip() for t in types if t.strip()]
        
        # Calculate similarity score
        similarity_score = calculate_ranked_deck_similarity(card_id)
        
        response = {
            "hp": hp,
            "types": types if types else ["N/A"],
            "similarity_score": f"{similarity_score:.1f}%"
        }
        print(f"Final response: {response}")  # Debug log
        return json.dumps(response)
    return json.dumps({"error": "Card not found"}), 404

def get_top_limitless_decks():
    print("Fetching tournaments...")
    try:
        # Try the new API endpoint
        response = requests.get("https://play.limitlesstcg.com/api/tournaments/recent")
        print("Status:", response.status_code)
        
        if response.status_code != 200:
            print("Failed to fetch recent tournaments, trying alternative endpoint...")
            # Try alternative endpoint
            response = requests.get("https://play.limitlesstcg.com/api/tournaments")
            print("Alternative endpoint status:", response.status_code)
            
            if response.status_code != 200:
                print("Both endpoints failed, using fallback data")
                # Return some fallback data for testing
                return [
                    {
                        "player": "Test Player 1",
                        "placing": "1",
                        "deck_archetype": "Fire Deck",
                        "event": "Test Tournament",
                        "link": "https://limitlesstcg.com/decks/test1",
                        "cards": ["sv2-258", "sv2-259", "sv2-260"]  # Example card IDs
                    },
                    {
                        "player": "Test Player 2",
                        "placing": "2",
                        "deck_archetype": "Water Deck",
                        "event": "Test Tournament",
                        "link": "https://limitlesstcg.com/decks/test2",
                        "cards": ["sv2-261", "sv2-262", "sv2-263"]  # Example card IDs
                    }
                ]
        
        tournaments = response.json()
        print(f"Total tournaments: {len(tournaments)}")
        
        top_decks = []
        checked = 0
        
        for t in tournaments:
            if not t.get("id") or not t.get("name") or t.get("status") != "completed":
                continue
            
            tid = t["id"]
            print(f"\nChecking tournament: {t['name']} (ID: {tid})")
            
            standings_url = f"https://play.limitlesstcg.com/api/tournaments/{tid}/standings"
            r = requests.get(standings_url)
            print("Standings response:", r.status_code)
            
            if r.status_code != 200:
                continue
            
            standings = r.json()
            print(f"Found {len(standings)} players in standings.")
            
            if not standings:
                continue
            
            for entry in standings[:5]:  # Try 5 per tournament
                if not entry.get("deck_id"):
                    continue
                
                print(f"Adding deck: {entry.get('player')} - {entry.get('archetype')}")
                
                # Get deck details
                deck_url = f"https://play.limitlesstcg.com/api/decks/{entry['deck_id']}"
                deck_response = requests.get(deck_url)
                if deck_response.status_code != 200:
                    continue
                    
                deck_data = deck_response.json()
                if not deck_data or 'cards' not in deck_data:
                    continue
                
                top_decks.append({
                    "player": entry.get("player", "Unknown"),
                    "placing": entry.get("placing", "?"),
                    "deck_archetype": entry.get("archetype", "Unknown"),
                    "event": t["name"],
                    "link": f"https://limitlesstcg.com/decks/{entry['deck_id']}",
                    "cards": [card.get('id') for card in deck_data['cards'] if card.get('id')]
                })
                
                if len(top_decks) >= 10:
                    print("Collected 10 decks, returning now.")
                    return top_decks
            
            checked += 1
            if checked >= 10:
                break
        
        print(f"Returning {len(top_decks)} decks.")
        return top_decks
        
    except Exception as e:
        print(f"Error in get_top_limitless_decks: {e}")
        # Return fallback data if API fails
        return [
            {
                "player": "Test Player 1",
                "placing": "1",
                "deck_archetype": "Fire Deck",
                "event": "Test Tournament",
                "link": "https://limitlesstcg.com/decks/test1",
                "cards": ["sv2-258", "sv2-259", "sv2-260"]  # Example card IDs
            },
            {
                "player": "Test Player 2",
                "placing": "2",
                "deck_archetype": "Water Deck",
                "event": "Test Tournament",
                "link": "https://limitlesstcg.com/decks/test2",
                "cards": ["sv2-261", "sv2-262", "sv2-263"]  # Example card IDs
            }
        ]

def extract_and_preprocess_card_features_with_flavor_text():
    query_sql = """
    SELECT types, abilities, attacks, subtypes, weaknesses, resistances, supertype, flavorText FROM allcards
    """
    data = mysql_engine.query_selector(query_sql)
    
    card_features = []
    for row in data:
        types, abilities, attacks, subtypes, weaknesses, resistances, supertype, flavor_text = row
        
        abilities_text = parse_json_field(abilities)
        attacks_text = parse_json_field(attacks)
        weaknesses_text = parse_json_field(weaknesses)
        resistances_text = parse_json_field(resistances)
        
        combined_text = " ".join(filter(None, [abilities_text, attacks_text, weaknesses_text, resistances_text, flavor_text]))
        
        types_list = parse_list_field(types)
        subtypes_list = parse_list_field(subtypes)
        
        card_features.append({
            "types": types_list,
            "subtypes": subtypes_list,
            "supertype": supertype,
            "text": combined_text
        })
    
    return card_features

def parse_json_field(json_str):
    if json_str is None:
        return "" 
    try:
        json_str = json_str.replace("'", "\"")
        data = json.loads(json_str)
        if isinstance(data, list):
            return " ".join(item.get('text', '') for item in data if isinstance(item, dict))
        return ""
    except (json.JSONDecodeError, TypeError):
        return ""

def parse_list_field(list_str):
    if list_str is None:
        return []
    try:
        list_str = list_str.replace("'", "\"")
        data = json.loads(list_str)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, TypeError):
        return []


def preprocess():
    query_sql = """
    SELECT id, name, types, abilities, attacks, subtypes, weaknesses, resistances, supertype, flavorText, hp FROM allcards
    """
    data = mysql_engine.query_selector(query_sql)
    
    card_ids = []
    card_names = []
    card_features = []
    hp_values = []
    for row in data:
        card_id, name, types, abilities, attacks, subtypes, weaknesses, resistances, supertype, flavor_text, hp = row
        
        abilities_text = parse_json_field(abilities)
        attacks_text = parse_json_field(attacks)
        weaknesses_text = parse_json_field(weaknesses)
        resistances_text = parse_json_field(resistances)
        
        combined_text = " ".join(filter(None, [abilities_text, attacks_text, weaknesses_text, resistances_text, flavor_text]))
        
        types_list = parse_list_field(types)
        subtypes_list = parse_list_field(subtypes)
        
        card_ids.append(card_id)
        card_names.append(name)
        card_features.append({
            "types": types_list,
            "subtypes": subtypes_list,
            "supertype": supertype,
            "text": combined_text
        })
        hp_values.append(hp if hp is not None else 0) 
    return card_features, hp_values, card_ids, card_names

def compute_tfidf_with_hp():
    card_features, hp_values = preprocess()
    
    combined_texts = []
    for card in card_features:
        combined_text = " ".join(card["types"] + card["subtypes"] + [card["supertype"], card["text"]])
        combined_texts.append(combined_text)
        
        # Print the combined text for each card print("Combined Text for Card:", combined_text)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    # SVD Decomposition, Inspired by SVD Demo
    docs_compressed, s, words_compressed = svds(tfidf_matrix, k = 40)
    words_compressed = words_compressed.T
    docs_compressed_norm = normalize(docs_compressed)

    scaler = MinMaxScaler()
    hp_normalized = scaler.fit_transform(np.array(hp_values).reshape(-1, 1))

    final_matrix = np.hstack((docs_compressed_norm, hp_normalized))

    return final_matrix, vectorizer.get_feature_names_out()

def svd_search(query, k=20):
    card_features, hp_values, card_ids, card_name = preprocess()

    combined_texts = [
        " ".join(card["types"] + card["subtypes"] + [card["supertype"], card["text"]])
        for card in card_features
    ]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_texts)
    docs_compressed, s, words_compressed = svds(tfidf_matrix, k=40)
    docs_compressed = normalize(docs_compressed)

    scaler = MinMaxScaler()
    hp_normalized = scaler.fit_transform(np.array(hp_values).reshape(-1, 1))
    final_matrix = np.hstack((docs_compressed, hp_normalized))

    query_vec = vectorizer.transform([query])
    query_compressed = normalize(query_vec @ words_compressed.T)

    query_combined = np.hstack((query_compressed, [[0.5]]))

    sims = final_matrix @ query_combined.T
    sims = sims.flatten()

    sorted_indices = np.argsort(sims)[::-1][:k]

    results = []
    for idx in sorted_indices:
        card = card_features[idx]
        results.append({
            "id": card_ids[idx],
            "title": card_name[idx],
            "descr": card["text"]
        })

    return json.dumps(results)

@app.route("/get_top_decks")
def get_top_decks():
     try:
         decks = get_top_limitless_decks()
         return json.dumps({"success": True, "decks": decks})
     except Exception as e:
         return json.dumps({"success": False, "message": str(e)})

@app.route("/episodes")
def episodes_search():
    text = request.args.get("title", "")
    return svd_search(text)
  
def calculate_ranked_deck_similarity(card_id):
    print(f"\nCalculating similarity for card {card_id}")
    
    query_sql = f"""
    SELECT types, abilities, attacks, subtypes, weaknesses, resistances, supertype, flavorText, hp, name 
    FROM allcards 
    WHERE id = '{card_id}'
    """
    print(f"Card query: {query_sql}")
    card_data = list(mysql_engine.query_selector(query_sql))
    
    if not card_data:
        print("No card data found")
        return 0.0
        
    row = card_data[0]
    types, abilities, attacks, subtypes, weaknesses, resistances, supertype, flavor_text, hp, name = row
    print(f"Card name: {name}, Types: {types}, HP: {hp}")
    
    abilities_text = parse_json_field(abilities)
    attacks_text = parse_json_field(attacks)
    weaknesses_text = parse_json_field(weaknesses)
    resistances_text = parse_json_field(resistances)
    combined_text = " ".join(filter(None, [abilities_text, attacks_text, weaknesses_text, resistances_text, flavor_text]))
    types_list = parse_list_field(types)
    subtypes_list = parse_list_field(subtypes)
    print(f"Parsed types: {types_list}, Parsed subtypes: {subtypes_list}")
    
    scores = {}
    
    type_score = 0
    if types_list:
        common_types = {'Fire', 'Water', 'Grass', 'Lightning', 'Psychic', 'Fighting', 'Darkness', 'Metal', 'Fairy', 'Dragon'}
        type_score = len(set(types_list) & common_types) / len(types_list)
    scores['type'] = type_score * 0.3
    print(f"Type score: {type_score} -> {scores['type']}")
    
    hp_score = 0
    try:
        hp_value = int(hp) if hp else 0
        if hp_value >= 300:
            hp_score = 1.0
        elif hp_value >= 250:
            hp_score = 0.8
        elif hp_value >= 200:
            hp_score = 0.6
        elif hp_value >= 150:
            hp_score = 0.4
        elif hp_value >= 100:
            hp_score = 0.2
    except ValueError:
        pass
    scores['hp'] = hp_score * 0.2
    print(f"HP score: {hp_score} -> {scores['hp']}")
    
    ability_score = 0
    if abilities_text or attacks_text:
        powerful_keywords = {
            'draw', 'search', 'damage', 'heal', 'energy', 'evolution', 'retreat', 'switch',
            'prevent', 'protect', 'discard', 'shuffle', 'bench', 'prize', 'knock out'
        }
        text_words = set(combined_text.lower().split())
        matching_keywords = len(text_words & powerful_keywords)
        ability_score = min(matching_keywords / 5, 1.0)
    scores['ability'] = ability_score * 0.3
    print(f"Ability score: {ability_score} -> {scores['ability']}")
    
    subtype_score = 0
    if subtypes_list:
        valuable_subtypes = {'VMAX', 'VSTAR', 'ex', 'GX', 'V'}
        subtype_score = len(set(subtypes_list) & valuable_subtypes) / len(subtypes_list)
    scores['subtype'] = subtype_score * 0.2
    print(f"Subtype score: {subtype_score} -> {scores['subtype']}")
    
    base_score = sum(scores.values())
    print(f"Base score: {base_score}")
    
    check_sql = f"""
    SELECT COUNT(*) FROM ranked_decks WHERE id_card = '{card_id}'
    """
    count_result = list(mysql_engine.query_selector(check_sql))
    tournament_count = count_result[0][0]
    print(f"Number of times this card appears in ranked_decks: {tournament_count}")
    
    if tournament_count > 0:
        query_sql = f"""
        SELECT DISTINCT id_tournament, name_tournament, category_tournament
        FROM ranked_decks 
        WHERE id_card = '{card_id}'
        """
        deck_data = list(mysql_engine.query_selector(query_sql))
        
        if deck_data:
            tournament_bonus = min(tournament_count / 10, 0.3)
            base_score *= (1 + tournament_bonus)
            print(f"Applied tournament bonus: {tournament_bonus}")
            
            championship_count = sum(1 for deck in deck_data if deck[2] and "championship" in deck[2].lower())
            if championship_count > 0:
                championship_bonus = min(championship_count / 5, 0.2)
                base_score *= (1 + championship_bonus)
                print(f"Applied championship bonus: {championship_bonus}")
    
    final_score = base_score * 100
    if final_score < 20:
        final_score = 20
    print(f"Final score: {final_score}%")
    
    return min(final_score, 100)

if 'DB_NAME' not in os.environ:
     app.run(debug=True,host="0.0.0.0",port=5000)