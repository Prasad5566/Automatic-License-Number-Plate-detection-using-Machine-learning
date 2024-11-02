import sqlite3

# Function to drop the table
def drop_table():
    # Connect to the database
    conn = sqlite3.connect('rto_info.db')
    cursor = conn.cursor()

    # Drop the rto_info table
    cursor.execute('DROP TABLE IF EXISTS rto_info')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

# Call the function to drop the table
drop_table()

print("Table 'rto_info' dropped successfully.")
