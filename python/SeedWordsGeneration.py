import csv
import json

label_file = "/Users/hochen/Documents/Projects/prd-review/737684276_Gadgets_topic_word_cluster 3.csv"


seedwords = {}
with open(label_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:

        if row['Topics'] is "":
            continue

        if row['Topics'] not in seedwords:
            seedwords[row['Topics']] = []

        word_list = row['Words'].split(";")
        for item in word_list:
            if len(item.strip()) == 0:
                continue
            print("item:",item)
            word, dist = item.split(":")

            seedwords[row['Topics']].append(word)

f = open("knowledge/topics2.json", "w")
jsonstr = json.dump(seedwords, f,  indent=4)
f.close()


