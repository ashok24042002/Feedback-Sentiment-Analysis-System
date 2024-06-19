# import mysql.connector

# mydb = mysql.connector.connect(
#   host="localhost",
#   user="root",
#   password="",
#   database="dump"
# )

# mycursor = mydb.cursor()

# sql = "INSERT INTO customer (name, number) VALUES (%s, %s)"
# val = ("John",1)
# mycursor.execute(sql, val)

# mydb.commit()

# print(mycursor.rowcount, "record inserted.")

import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="forum"
)

mycursor = mydb.cursor()

mycursor.execute("SELECT id FROM discussion")

myresult = mycursor.fetchall()

for x in myresult:
    print(type(x))