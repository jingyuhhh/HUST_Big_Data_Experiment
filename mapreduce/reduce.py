import os
import threading
import time


class Combiner:
    def __init__(self, data):
        self.data = data
        self.combined_data = {}

    def combine(self):
        # 合并数据，将相同的词汇的频率累加
        print('Combiner: 合并数据.')
        for pair in self.data.items():
            if pair[0] in self.combined_data.keys():
                self.combined_data[pair[0]]['count'] += pair[1]['count']
                self.combined_data[pair[0]]['file'].extend(pair[1]['file'])
            else:
                self.combined_data[pair[0]] = {'count': pair[1]['count'], 'file': pair[1]['file']}
        return self.combined_data

class Reducer(threading.Thread):
    def __init__(self, node_id, allocated_files):
        threading.Thread.__init__(self)
        self.node_id = node_id
        self.allocated_files = allocated_files
        self.word_counts = {}

    def run(self):
        # 开始运行 Reduce 节点，读取分配的文件，计算词汇的总频数，排序，输出结果
        start_time = time.time()
        self.read_and_count_words()
        combiner = Combiner(self.word_counts)
        self.word_counts = combiner.combine()
        self.sort_word_counts()
        self.output_results()
        print(f'{self.node_id}: 完成. 用时 {time.time() - start_time:.2f} 秒.')

    def read_and_count_words(self):
        # 读取分配的文件，计算词汇的频数
        print(f'{self.node_id}: 读取数据.')
        for file in self.allocated_files:
            with open(f'{file}', 'r') as f:
                data = f.read()
                self.count_words(data)

    def count_words(self, data):
        # 计算词汇的频数
        data = data.splitlines()
        for line in data:
            filename, word, count = line.split()
            if word in self.word_counts.keys():
                self.word_counts[word]['count'] += int(count)
                self.word_counts[word]['file'].append(filename.split(".")[0])
            else:
                self.word_counts[word] = {'count': int(count), 'file': [filename.split(".")[0]]}

    def sort_word_counts(self):
        # 对词汇的频数进行排序
        self.word_counts = dict(sorted(self.word_counts.items(), key=lambda x: x[1]['count'], reverse=True))

    def output_results(self):
        # 输出结果到文件
        print(f'{self.node_id}: 写入结果.')
        with open(f'reduce_result/result_{self.node_id}.txt', 'w') as f:
            for word in self.word_counts:
                f.write(f'{word} {self.word_counts[word]["count"]} {self.word_counts[word]["file"]}\n')

    def get_result(self):
        # 获取结果
        return self.word_counts

class FinalReducer(threading.Thread):
    def __init__(self, node_id, data):
        threading.Thread.__init__(self)
        self.node_id = node_id
        self.data = data
        self.res = {}

    def run(self):
        start_time = time.time()
        self.merge_data()
        self.output()
        print(f'合并节点 {self.node_id}: 完成. 用时 {time.time() - start_time:.2f} 秒.')

    def merge_data(self):
        # 合并数据，将相同的词汇的频率累加
        print(f'节点 {self.node_id}: 合并数据.')
        for data in self.data:
            for pair in data.items():
                if pair[0] in self.res:
                    self.res[pair[0]]['count'] += pair[1]['count']
                    self.res[pair[0]]['file'].extend(pair[1]['file'])
                else:
                    self.res[pair[0]] = {'count': pair[1]['count'], 'file': pair[1]['file']}
    def output(self):
        # 输出到文件
        print(f'节点 {self.node_id}: 写入结果.')
        with open(f'reduce_result/result.txt', 'w') as f:
            for word in self.res:
                f.write(f'{word} {self.res[word]["count"]} {self.res[word]["file"]} \n')

    def get_result(self):
        # 获取结果
        return self.res


class FileShuffler:
    def __init__(self, intermediate_path):
        self.intermediate_path = intermediate_path
        self.intermediate_files = os.listdir(intermediate_path)

    def shuffle(self, num_reducer):
        # 开始 Shuffle 阶段，计算文件大小，分配文件给 Reduce 节点，输出分配结果，运行 Reduce 节点，运行 Top1000Reducer 节点
        file_sizes = self.calculate_file_sizes()
        reducer_files = self.allocate_files(file_sizes, num_reducer)
        self.output_allocation(reducer_files, file_sizes)
        print('shuffle 完成.')
        results = self.run_reducers(reducer_files)
        print('reduce 完成.')
        final_result = self.run_final_reducer(results)
        print('合并完成.')

    def calculate_file_sizes(self):
        # 计算文件的大小
        file_sizes = {}
        for file in self.intermediate_files:
            file_sizes[file] = os.path.getsize(f'{self.intermediate_path}/{file}')
        return file_sizes

    def allocate_files(self, file_sizes, num_reducer):
        # 根据文件的大小，将文件分配给 Reduce 节点
        file_size_sum = sum(file_sizes.values())
        file_size_per_reducer = file_size_sum // num_reducer
        reducer_files = [[] for _ in range(num_reducer)]
        current_size = 0
        current_reducer = 0
        for file in self.intermediate_files:
            if current_reducer < num_reducer - 1 and current_size + file_sizes[file] > file_size_per_reducer * 1.1:
                current_reducer += 1
                current_size = 0
            reducer_files[current_reducer].append(file)
            current_size += file_sizes[file]
        return reducer_files

    def output_allocation(self, reducer_files, file_sizes):
        # 输出文件的分配结果
        for i, files in enumerate(reducer_files):
            print(f'{i}:')
            for file in files:
                print(f'    {file} ({file_sizes[file]} 字节)')

    def run_reducers(self, reducer_files):
        # 运行 Reduce 节点
        reducer_nodes = [Reducer(i, [f'{self.intermediate_path}/{file}' for file in reducer_files[i]]) for i in range(len(reducer_files))]
        for node in reducer_nodes:
            node.start()
        results = []
        for node in reducer_nodes:
            node.join()
            results.append(node.get_result())
        return results

    def run_final_reducer(self, results):
        reducer = FinalReducer(0, results)
        reducer.start()
        reducer.join()
        return reducer.get_result()

if __name__ == '__main__':
    shuffler = FileShuffler("map_result")
    shuffler.shuffle(3)
    with open('./reduce_result/result.txt', 'r') as f:
        result = f.read()
        top_1000 = result.splitlines()[:1000]
    with open('./reduce_result/top_1000.txt', 'w') as f:
        for line in top_1000:
            f.write(f'{line}\n')

