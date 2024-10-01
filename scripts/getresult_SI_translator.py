import numpy as np
import pickle
import os, sys
import csv
import pandas as pd


def main(csv_file, save_to):
    if os.path.exists(csv_file) == False:
        print('[WARNING] csv file not found:{}'.format(csv_file))
        return
    dict_list = []

    # open CSV file and read
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            dict_list.append(row)


    tmp = dict()
    for item in dict_list:
        algorithm = item['algorithm']
        dataset = item['dataset']
        benchmark = item['benchmark']
        seed = item['seed']
        num = item['num']
        low = item['low']
        high = item['high']
        offlineBest = item['offlineBest']
        algorithmBest = item['algorithmBest']
        SI = item['SI']
        name = f'{dataset}/{benchmark}/{seed}/{low}/{high}'
        tmp[name] = offlineBest

    dct = dict()
    for item in dict_list:
        algorithm = item['algorithm']
        dataset = item['dataset']
        benchmark = item['benchmark']
        seed = item['seed']
        num = item['num']
        low = item['low']
        high = item['high']
        offlineBest = item['offlineBest']
        algorithmBest = item['algorithmBest']
        SI = item['SI']
        name = f'{algorithm}/{dataset}/{benchmark}/{seed}/{low}/{high}'
        dct[name] = (offlineBest, algorithmBest, SI)

    seedlist = list(range(8))

    algolist = [x['algorithm'] for x in dict_list]
    algolist = list(set(algolist))

    datasetlist = [x['dataset'] for x in dict_list]
    datasetlist = list(set(datasetlist))

    def get_benchmark_list(algorithm,dataset):
        algorithm = str(algorithm)
        dataset = str(dataset)
        benchmark_list = []
        for item in dct.keys():
            args = item.split('/')
            if args[0] == algorithm and args[1] == dataset:
                benchmark_list.append(args[2])
                
        benchmark_list = list(set(benchmark_list))
        
        return benchmark_list

    def get_low_high_list(algorithm, dataset, benchmark):
        algorithm = str(algorithm)
        dataset = str(dataset)
        benchmark = str(benchmark)
        low_high_list = []
        for item in dct.keys():
            args = item.split('/')
            if args[0] == algorithm and args[1] == dataset and args[2] == benchmark:
                low_high_list.append((args[4], args[5]))
        low_high_list = list(set(low_high_list))
        return low_high_list




    after = []
    for Dataset in datasetlist:
        for Algorithm in algolist:
            for Benchmark in sorted(get_benchmark_list(Algorithm,Dataset)):
                print(f'[INFO] found : Algorithm:{Algorithm}, dataset:{Dataset}, benchmarkid:{Benchmark}')
                item = dict()
                item['dataset'] = dataset
                item['benchmark'] = Benchmark
                item['algorithm'] = Algorithm
                item['low_high'] = f"{Dataset.rstrip('_data')} {Benchmark}"
                item['flag'] = ''
                for Seed in seedlist:
                    algorithm = Algorithm
                    dataset = Dataset
                    benchmark = Benchmark
                    seed = Seed
                    name = f'{algorithm}/{dataset}/{benchmark}/{seed}/{low}/{high}'
                    offlineBest, algorithmBest, SI = '','',''
                    item[f'seed{Seed}'] =  ''
                after.append(item)
                
                for low,high in sorted(get_low_high_list(Algorithm,Dataset,Benchmark)):
                    item = dict()
                    item['dataset'] = dataset
                    item['benchmark'] = Benchmark
                    item['algorithm'] = Algorithm
                    item['low_high'] = f''
                    
                    item['flag'] = ''
                    for Seed in seedlist:
                        algorithm = Algorithm
                        dataset = Dataset
                        benchmark = Benchmark
                        seed = Seed
                        name = f'{algorithm}/{dataset}/{benchmark}/{seed}/{low}/{high}'
                        offlineBest, algorithmBest, SI = '','',''
                        item[f'seed{Seed}'] =  Seed
                    after.append(item)
                    
                    item = dict()
                    item['dataset'] = dataset
                    item['benchmark'] = Benchmark
                    item['algorithm'] = Algorithm
                    item['low_high'] = f'{low}-{high}'
                    
                    item['flag'] = 'Offline Best'
                    for Seed in seedlist:
                        algorithm = Algorithm
                        dataset = Dataset
                        benchmark = Benchmark
                        seed = Seed
                        name = f'{algorithm}/{dataset}/{benchmark}/{seed}/{low}/{high}'
                        try:
                            offlineBest, algorithmBest, SI = dct[name]
                        except:
                            offlineBest, algorithmBest, SI = '','',''
                        item[f'seed{Seed}'] =  offlineBest
                    after.append(item)
                    
                    item = dict()
                    item['dataset'] = dataset
                    item['benchmark'] = Benchmark
                    item['algorithm'] = Algorithm
                    item['low_high'] = f'{low}-{high}'
                    
                    item['flag'] = 'Optimization Result'
                    for Seed in seedlist:
                        algorithm = Algorithm
                        dataset = Dataset
                        benchmark = Benchmark
                        seed = Seed
                        name = f'{algorithm}/{dataset}/{benchmark}/{seed}/{low}/{high}'
                        try:
                            offlineBest, algorithmBest, SI = dct[name]
                        except:
                            offlineBest, algorithmBest, SI = '','',''
                        item[f'seed{Seed}'] =  algorithmBest
                    after.append(item)
                    
                    
                    item = dict()
                    item['dataset'] = dataset
                    item['benchmark'] = Benchmark
                    item['algorithm'] = Algorithm
                    item['low_high'] = f'{low}-{high}'
                    
                    item['flag'] = 'SI'
                    for Seed in seedlist:
                        algorithm = Algorithm
                        dataset = Dataset
                        benchmark = Benchmark
                        seed = Seed
                        name = f'{algorithm}/{dataset}/{benchmark}/{seed}/{low}/{high}'
                        try:
                            offlineBest, algorithmBest, SI = dct[name]
                        except:
                            offlineBest, algorithmBest, SI = '','',''
                        item[f'seed{Seed}'] =  SI
                    after.append(item)


    savedata = after
    with open(save_to, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=savedata[0].keys())
        writer.writeheader()
        for row in savedata:
            writer.writerow(row)
    print(f'[DONE] result saved to {save_to}')


csv_file = os.path.abspath(os.path.join(__file__, '..','..','results','summary', 'results_unconstraint.csv'))
save_to = os.path.abspath(os.path.join(__file__, '..','..','results','summary', 'results_unconstraint_translate.csv'))
main(csv_file, save_to)

csv_file = os.path.abspath(os.path.join(__file__, '..','..','results','summary', 'results_constraint.csv'))
save_to = os.path.abspath(os.path.join(__file__, '..','..','results','summary', 'results_constraint_translate.csv'))
main(csv_file, save_to)