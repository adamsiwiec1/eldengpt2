import csv
import logging

logging.basicConfig(filename='error.log', level=logging.ERROR)

with open('elden_ring_steam_reviews.csv', 'r', encoding='utf-8') as src_file:
    reader = csv.reader(src_file)
    with open('elden_llm.txt', 'w', encoding='utf-8') as dst_file:
        try:
            for row in reader:
                dst_file.write(row[2].replace('\n', ' ') + '\n')
        except Exception as e:
            logging.exception("An error occurred: %s", str(e))
