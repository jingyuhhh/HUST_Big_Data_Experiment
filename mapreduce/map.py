import os
import re
import threading

class MapNode(threading.Thread):
    def __init__(self, node_id, data_path):
        threading.Thread.__init__(self)
        self.node_id = node_id
        self.data_path = data_path
        self.target_words = set()
        self.word_counts = {}

    def run(self):
        self.read_target_words()
        self.process_files()
        self.write_results()

    def read_target_words(self):
        print(f'map节点 {self.node_id}: 正在读取目标单词.')
        with open("../source_data/words.txt", 'r') as file:
            words = file.read().splitlines()
            self.target_words = {word.lower() for word in words}

    def process_files(self):
        print(f'map节点 {self.node_id}: 正在处理文件.')
        for filename in os.listdir(self.data_path):
            if filename.endswith(".txt"):
                self.process_file(filename)

    def process_file(self, filename):
        with open(f'{self.data_path}/{filename}', 'r', encoding='utf-8') as file:
            words_in_file = re.findall(r'\b[a-zA-Z]+\b', file.read())
            words_in_file = [word.lower() for word in words_in_file]
            self.count_words(filename, words_in_file)

    def count_words(self, filename, words_in_file):
        for word in words_in_file:
            word = word.lower()
            if word in self.target_words:
                if (filename, word) in self.word_counts:
                    self.word_counts[(filename, word)] += 1
                else:
                    self.word_counts[(filename, word)] = 1

    def write_results(self):
        print(f'map节点 {self.node_id}: 正在写入结果.')
        with open(f'./map_result/map_result_{self.node_id}.txt', 'w') as file:
            for pair, count in self.word_counts.items():
                file.write(f'{pair[0]} {pair[1]} {count}\n')

if __name__ == '__main__':
    map_nodes = [MapNode(i, f'../source_data/folder_{i}') for i in range(1, 10)]
    for node in map_nodes:
        node.start()
    for node in map_nodes:
        node.join()
    print('map完成.')
