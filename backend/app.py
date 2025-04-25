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

def svd_search(query, card_type='pokemon', k=20):
    card_features, hp_values, card_ids, card_names = preprocess()
    
    if card_type.lower() == 'pokemon':
        filtered_indices = [i for i, card in enumerate(card_features) if card['supertype'].lower() in ['pokemon', 'pokémon']]
    else: 
        filtered_indices = [i for i, card in enumerate(card_features) if card['supertype'].lower() == 'trainer']
    
    filtered_features = [card_features[i] for i in filtered_indices]
    filtered_ids = [card_ids[i] for i in filtered_indices]
    filtered_names = [card_names[i] for i in filtered_indices]
    
    query = query.lower().strip()
    query_terms = set(query.split())
    
    query_variations = set()
    for term in query_terms:
        query_variations.add(term)
        query_variations.add(f"{term}-type")
        query_variations.add(f"{term} type")
        query_variations.add(f"{term} card")
        if card_type.lower() == 'trainer':
            query_variations.add(f"{term} trainer")
    
    combined_texts = [
        " ".join(card["types"] + card["subtypes"] + [card["supertype"], card["text"]])
        for card in filtered_features
    ]

    vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r'(?u)\b[a-zA-Z]{3,}\b')
    tfidf_matrix = vectorizer.fit_transform(combined_texts)
    docs_compressed, s, words_compressed = svds(tfidf_matrix, k=40)
    docs_compressed = normalize(docs_compressed)
    
    if card_type.lower() == 'pokemon':
        scaler = MinMaxScaler()
        filtered_hp_values = [hp_values[i] for i in filtered_indices]
        hp_normalized = scaler.fit_transform(np.array(filtered_hp_values).reshape(-1, 1))
        final_matrix = np.hstack((docs_compressed, hp_normalized))
    else:
        final_matrix = docs_compressed
    
    query_vec = vectorizer.transform([query])
    query_compressed = normalize(query_vec @ words_compressed.T)
    
    if card_type.lower() == 'pokemon':
        query_combined = np.hstack((query_compressed, [[0.5]]))
        sims = final_matrix @ query_combined.T
    else:
        sims = final_matrix @ query_compressed.T
    
    sims = sims.flatten()
    sorted_indices = np.argsort(sims)[::-1][:k]
    
    feature_names = vectorizer.get_feature_names_out()
    results = []
    
    for idx in sorted_indices:
        if sims[idx] > 0:
            card = filtered_features[idx]
            
            doc_vector = docs_compressed[idx]
            top_dim_indices = np.argsort(np.abs(doc_vector))[-5:][::-1]
            
            dimension_tags = []
            for dim_idx in top_dim_indices:
                dim_weights = words_compressed[dim_idx]
                top_term_indices = np.argsort(np.abs(dim_weights))[-5:][::-1]
                top_terms = [feature_names[i] for i in top_term_indices]
                dimension_tags.extend(top_terms)
            
            matching_tags = [tag for tag in dimension_tags if any(q in tag.lower() for q in query_variations)]
            
            if not matching_tags:
                matching_tags = list(set(dimension_tags))[:10]
            
            results.append({
                "id": filtered_ids[idx],
                "title": filtered_names[idx],
                "descr": f"Relevance: {sims[idx]:.2f}",
                "tags": matching_tags
            })
    
    return json.dumps(results)

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
    return svd_search(query, 'trainer')

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
    print(f"Executing query: {query_sql}") 
    data = list(mysql_engine.query_selector(query_sql))
    if data:
        row = data[0]
        print(f"Full row data: {row}") 
        
        title = row[7] if row[7] else "N/A"
        
        hp = row[13] if row[13] else "N/A"
        
        types_str = row[9] if row[9] else "[]"
        types = types_str.strip('[]').replace("'", "").split(',')
        types = [t.strip() for t in types if t.strip()]
        
        similarity_score = calculate_ranked_deck_similarity(card_id)
        
        card_features, hp_values, card_ids, card_names = preprocess()
        card_index = card_ids.index(card_id) if card_id in card_ids else -1
        
        tags = []
        if card_index != -1:
            
            combined_texts = [
                " ".join(card["types"] + card["subtypes"] + [card["supertype"], card["text"]])
                for card in card_features
            ]
            
            vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r'(?u)\b[a-zA-Z]{3,}\b')
            tfidf_matrix = vectorizer.fit_transform(combined_texts)
            docs_compressed, s, words_compressed = svds(tfidf_matrix, k=40)
            docs_compressed = normalize(docs_compressed)
            
            doc_vector = docs_compressed[card_index]
            feature_names = vectorizer.get_feature_names_out()
            
            top_dims = np.argsort(np.abs(doc_vector))[-10:][::-1]
            term_scores = {}

            for dim in top_dims:
                weights = words_compressed[dim]
                top_term_indices = np.argsort(np.abs(weights))[-10:]
                for idx in top_term_indices:
                    term = feature_names[idx]
                    term_scores[term] = term_scores.get(term, 0) + abs(weights[idx])
            
            sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
            tags = [term for term, _ in sorted_terms[:10]]
        
        response = {
            "title": title,
            "hp": hp,
            "types": types if types else ["N/A"],
            "similarity_score": f"{similarity_score:.1f}%",
            "tags": tags
        }
        print(f"Final response: {response}") 
        return json.dumps(response)
    return json.dumps({"error": "Card not found"}), 404

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
    
    name = name if name else "Unknown"
    types = types if types else "[]"
    abilities = abilities if abilities else "[]"
    attacks = attacks if attacks else "[]"
    weaknesses = weaknesses if weaknesses else "[]"
    resistances = resistances if resistances else "[]"
    flavor_text = flavor_text if flavor_text else ""
    
    abilities_text = parse_json_field(abilities)
    attacks_text = parse_json_field(attacks)
    weaknesses_text = parse_json_field(weaknesses)
    resistances_text = parse_json_field(resistances)
    
    if isinstance(types, str):
        types = types.strip('[]').replace("'", "").split(',')
        types = [t.strip() for t in types if t.strip()]
    
    combined_text = " ".join(filter(None, [name] * 3 + types + [abilities_text, attacks_text, weaknesses_text, resistances_text, flavor_text]))
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
        
        combined_text = " ".join(filter(None, [name] * 3 + types + [abilities_text, attacks_text, weaknesses_text, resistances_text, flavor_text]))
        
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
    if isinstance(list_str, list):
        return list_str
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
        
        types_list = parse_list_field(types)
        subtypes_list = parse_list_field(subtypes)
        
        card_ids.append(card_id)
        card_names.append(name)
        hp_category = 'high hp' if hp and hp > 200 else 'low hp'
        combined_text = " ".join(filter(None, [name] * 3 + types_list + subtypes_list + [abilities_text, attacks_text, weaknesses_text, resistances_text, flavor_text, hp_category]))
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

    vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r'(?u)\b[a-zA-Z]{3,}\b')
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    # SVD Decomposition, Inspired by SVD Demo
    docs_compressed, s, words_compressed = svds(tfidf_matrix, k = 40)
    words_compressed = words_compressed.T
    docs_compressed_norm = normalize(docs_compressed)

    scaler = MinMaxScaler()
    hp_normalized = scaler.fit_transform(np.array(hp_values).reshape(-1, 1))

    final_matrix = np.hstack((docs_compressed_norm, hp_normalized))

    return final_matrix, vectorizer.get_feature_names_out()

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
    return svd_search(text, 'pokemon')
  
@app.route("/generate_deck", methods=['POST'])
def generate_deck():
    try:
        data = request.get_json()
        selected_types = data.get('types', [])
        strategy_value = data.get('strategy_value', 50)
        risk_value = data.get('risk_value', 50)
        
        print(f"Received request with types: {selected_types}, strategy: {strategy_value}, risk: {risk_value}")
        
        if not selected_types:
            print("No types selected")
            return json.dumps({"success": False, "message": "No types selected"}), 400
            
        result = generate_deck_with_strategy(selected_types, strategy_value)
        print(f"Generated deck with {len(result['deck'])} cards")
        print(f"Pokemon: {result['statistics']['pokemon_count']}")
        print(f"Trainer: {result['statistics']['trainer_count']}")
        print(f"Energy: {result['statistics']['energy_count']}")
        
        if len(result['deck']) == 0:
            print("Warning: Generated deck is empty")
            
        return json.dumps({"success": True, "data": result})
    except Exception as e:
        print(f"Error in generate_deck: {str(e)}")
        return json.dumps({"success": False, "message": str(e)}), 500

def generate_deck_with_strategy(selected_types, strategy_value):
    DECK_SIZE = 60
    
    if strategy_value <= 40:
        pokemon_percentage = 0.25
        trainer_percentage = 0.45
        energy_percentage = 0.30
    elif strategy_value >= 60:
        pokemon_percentage = 0.45
        trainer_percentage = 0.35
        energy_percentage = 0.20
    else:
        pokemon_percentage = 0.33
        trainer_percentage = 0.40
        energy_percentage = 0.27

    num_pokemon = int(DECK_SIZE * pokemon_percentage)
    num_trainers = int(DECK_SIZE * trainer_percentage)
    num_energy = DECK_SIZE - num_pokemon - num_trainers

    num_types = len(selected_types)
    pokemon_per_type = num_pokemon // num_types if num_types > 0 else num_pokemon
    remaining_pokemon = num_pokemon - (pokemon_per_type * num_types)

    pokemon_cards = []
    for type_name in selected_types:
        safe_type = type_name.replace("'", "''")
        type_pattern = f"%['{safe_type}']%"
        
        type_query = text(f"""
        SELECT * FROM allcards 
        WHERE supertype = 'Pokémon' 
        AND types LIKE :type_pattern
        AND (
            subtypes LIKE '%Basic%' 
            OR subtypes LIKE '%Stage 1%'
            OR subtypes LIKE '%Stage 2%'
            OR subtypes LIKE '%V%'
            OR subtypes LIKE '%VMAX%'
        )
        ORDER BY RAND()
        LIMIT :limit
        """)
        
        type_pokemon = list(mysql_engine.query_selector(
            type_query.bindparams(
                type_pattern=type_pattern,
                limit=pokemon_per_type + remaining_pokemon
            )
        ))
        pokemon_cards.extend(type_pokemon)
        remaining_pokemon = 0

    type_conditions = []
    for type_name in selected_types:
        safe_type = type_name.replace("'", "''")
        type_conditions.append(f"rules LIKE '%{safe_type}%'")
    type_conditions = " OR ".join(type_conditions)
    
    if strategy_value <= 40:
        trainer_query = text("""
        SELECT * FROM allcards 
        WHERE supertype = 'Trainer' 
        AND (subtypes LIKE '%Item%' OR subtypes LIKE '%Supporter%')
        AND (
            """ + type_conditions + """
            OR rules LIKE '%search%'
            OR rules LIKE '%draw%'
            OR rules LIKE '%evolve%'
        )
        ORDER BY RAND()
        LIMIT :limit
        """)
    else:
        trainer_query = text("""
        SELECT * FROM allcards 
        WHERE supertype = 'Trainer'
        AND (
            """ + type_conditions + """
            OR rules LIKE '%search%'
            OR rules LIKE '%draw%'
        )
        ORDER BY RAND()
        LIMIT :limit
        """)

    trainer_cards = list(mysql_engine.query_selector(trainer_query.bindparams(limit=num_trainers)))

    energy_per_type = num_energy // num_types
    remaining_energy = num_energy - (energy_per_type * num_types)
    
    energy_cards = []
    for type_name in selected_types:
        safe_type = type_name.replace("'", "''")
        type_pattern = f"%['{safe_type}']%"
        
        energy_query = text(f"""
        SELECT * FROM allcards 
        WHERE supertype = 'Energy' 
        AND (
            types LIKE :type_pattern
            OR (name LIKE '%Special%' AND rules LIKE :rules_pattern)
        )
        ORDER BY RAND()
        LIMIT :limit
        """)
        
        type_energy = list(mysql_engine.query_selector(
            energy_query.bindparams(
                type_pattern=type_pattern,
                rules_pattern=f"%{safe_type}%",
                limit=energy_per_type + remaining_energy
            )
        ))
        energy_cards.extend(type_energy)
        remaining_energy = 0

    if len(trainer_cards) < num_trainers:
        generic_trainer_query = text("""
        SELECT * FROM allcards 
        WHERE supertype = 'Trainer'
        AND id NOT IN :existing_ids
        ORDER BY RAND()
        LIMIT :limit
        """)
        
        existing_ids = tuple([card[0] for card in trainer_cards]) or ('',)
        additional_trainers = list(mysql_engine.query_selector(
            generic_trainer_query.bindparams(
                existing_ids=existing_ids,
                limit=num_trainers - len(trainer_cards)
            )
        ))
        trainer_cards.extend(additional_trainers)

    if len(energy_cards) < num_energy:
        generic_energy_query = text("""
        SELECT * FROM allcards 
        WHERE supertype = 'Energy'
        AND id NOT IN :existing_ids
        ORDER BY RAND()
        LIMIT :limit
        """)
        
        existing_ids = tuple([card[0] for card in energy_cards]) or ('',)
        additional_energy = list(mysql_engine.query_selector(
            generic_energy_query.bindparams(
                existing_ids=existing_ids,
                limit=num_energy - len(energy_cards)
            )
        ))
        energy_cards.extend(additional_energy)

    deck = []
    
    for card in pokemon_cards:
        deck.append({
            "id": card[0],
            "name": card[7],
            "supertype": "Pokémon",
            "types": parse_list_field(card[9]) if card[9] else [],
            "hp": card[13] if card[13] else "N/A",
            "subtypes": parse_list_field(card[8]) if card[8] else []
        })
    
    for card in trainer_cards:
        deck.append({
            "id": card[0],
            "name": card[7],
            "supertype": "Trainer",
            "subtype": card[8] if card[8] else "N/A",
            "rules": card[26] if card[26] else ""
        })
    
    for card in energy_cards:
        deck.append({
            "id": card[0],
            "name": card[7],
            "supertype": "Energy",
            "rules": card[26] if card[26] else ""
        })

    return {
        "deck": deck,
        "statistics": {
            "pokemon_count": len(pokemon_cards),
            "trainer_count": len(trainer_cards),
            "energy_count": len(energy_cards),
            "strategy_value": strategy_value,
            "selected_types": selected_types,
            "pokemon_type_distribution": {
                type_name: len([card for card in pokemon_cards 
                              if card[9] and type_name in str(card[9])]) 
                for type_name in selected_types
            },
            "energy_type_distribution": {
                type_name: len([card for card in energy_cards 
                              if type_name in str(card[7]) or (card[26] and type_name in str(card[26]))]) 
                for type_name in selected_types
            }
        }
    }

if 'DB_NAME' not in os.environ:
     app.run(debug=True,host="0.0.0.0",port=5000)