import os

file = 'dis_saves/test_1/testing_1.csv'
os.makedirs(os.path.dirname(file), exist_ok=True)
f = open(file, 'w')