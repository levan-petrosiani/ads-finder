import mysql.connector
from mysql.connector import errorcode

config = {
    'user': 'root',
    'password': '232018', # <-- IMPORTANT: Use your password here
    'host': '127.0.0.1',         # Use 127.0.0.1 instead of localhost
    'database': 'dejavu',
    'port': 3306
}

try:
    cnx = mysql.connector.connect(**config)
    print("✅ Connection successful!")
    cursor = cnx.cursor()
    cursor.execute("SELECT DATABASE();")
    db_name = cursor.fetchone()
    print(f"Connected to database: {db_name[0]}")
    cursor.close()
    cnx.close()
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("❌ Something is wrong with your user name or password")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("❌ Database does not exist")
    else:
        print(f"❌ An error occurred: {err}")