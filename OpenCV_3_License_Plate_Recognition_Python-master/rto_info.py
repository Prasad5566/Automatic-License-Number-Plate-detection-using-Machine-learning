import sqlite3

# Function to set up the database
def setup_database():
    # Connect to the database (creates the file if it doesn't exist)
    conn = sqlite3.connect('rto_info.db')
    cursor = conn.cursor()

    # Create the rto_info table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rto_info (
            license_plate TEXT PRIMARY KEY,
            owner_name TEXT,
            vehicle_model TEXT,
            reg_date TEXT
        )
    ''')

    # Insert sample data
    sample_data = [
        ('MCLRNF1', 'Abhishek', 'Maruti Suzuki Alto', '2021-01-01'),
        ('RIPLS1', 'Ravitej', 'Hyundai i20', '2020-05-15'),
        ('NVSBLE', 'Ullas', 'Toyota Innova', '2019-07-30'),
        ('ANBYOND', 'Ullas', 'Toyota Innova', '2019-07-30'),
        ('NYSJ', 'Arjun', 'Honda City', '2018-03-21'),
        ('ZOOMN65', 'Nina', 'Ford Mustang', '2022-11-12'),
        ('L0LWATT', 'Rohan', 'BMW 5 Series', '2020-02-14'),
        ('GAY247', 'Pooja', 'Audi A6', '2019-09-09'),
        ('DA69YCL', 'Karan', 'Nissan GTR', '2021-12-25'),
        ('DA59YCL', 'Isha', 'Mercedes C-Class', '2017-05-05'),
        ('Z00MN65', 'Nina', 'Ford Mustang', '2022-11-12'),
        ('U8NTBAD', 'Prasad', 'Ford Mustang', '2023-11-12'),
        ('NICESKY', 'Shreya', 'Ford Mustang', '2023-11-12'),



    ]

    cursor.executemany('INSERT OR IGNORE INTO rto_info VALUES (?, ?, ?, ?)', sample_data)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

# Call the function to set up the database
setup_database()

print("Database and table created successfully with sample data.")
