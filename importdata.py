# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 14:27:54 2018

@author: 科
"""

import sqlite3

def convert(value):
    if value.startswith('~'):
        return value.strip('~')
    if not value:
        value='0'
    return float(value)

conn=sqlite3.connect('food.db')
curs=conn.cursor()

curs.execute('''
             CREATE TABLE food(
             id          TEXT   PRIMARY KEY,
             desc        TEXT,
             water       FLOAT,
             kcal        FLOAT,
             protein     FLOAT,
             fat         FLOAT,
             ash         FLAOT,
             carbs       FLOAT,
             fiber       FLOAT,
             suger       FLOAT
)
''')
             
query='INSERT INTO food VALUES(?,?,?,?,?,?,?,?,?,?)'

for line in open(r'F:\sr28abbr\ABBREV.txt'):
    fields=line.split('^')
    vals=[convert(f) for f in fields[:field_count]]
    curs.execute(query,vals)
    
conn.commit()
conn.close()