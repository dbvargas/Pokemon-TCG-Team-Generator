# 4300-Pokemon-TCG-Generator-Project

## Summary

Our project is an intelligent deck-building system for the Pokemon Trading Card Game. We will use the Pokemon TCG card database and build competitive team comps based on strategy types as our queries. Unlike existing tools, which primarily function as card search engines, our system will generate competitive and customized decks based on user-defined strategies, playstyles, and goals. 

## Running our project locally 

For the initial setup each member would have to complete the following


### Step 0: Cloning the repo

- In your desired directory, CLONE this repository.

### Step 1: Setting up virtual environments

- python3.10 -m venv myenv
- source myenv/bin/activate
- cd backend
- pip install -r requirements.txt


### Step 2: Installing and starting MySQL
- brew install mysql
- mysql_secure_installation
    - press [ENTER] for password if they ask
    - or put in any password you would like
- mysql -u root -p  

### Step 3: Setting up .env
- in root directory create a .env file
- in the .env add:
  - DB_PASSWORD=<Put your SQL password from running ```mysql_secure_installation```>

  
### Step 4: Set up MySQL
Make sure **mysql -u root -p** is running SQL:
- CREATE DATABASE pokemon_database;
- USE pokemon_database;
- SOURCE <COPY PATH OF allcards.sql>


### Step 5: Running the project 

Make sure your MySQL server is running, then in app.py, change the SQL credentials to match your local MySQL credentials.

```flask run --host=0.0.0.0 --port=5000```