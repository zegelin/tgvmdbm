import argparse
import os
import shutil
import simplejson
import sys
from pathlib import Path
from contextlib import ExitStack
import itertools
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

IO_RESULT_KEYS = {'io_bytes', 'bw', 'bw_min', 'bw_max', 'bw_mean', 'iops', 'iops_min', 'iops_max', 'iops_mean',
                  'slat_ns', 'clat_ns', 'lat_ns'}

def process_benchmark(benchmark_path):
    try:
        with ExitStack() as stack:
            metadata = simplejson.load(stack.enter_context(open(benchmark_path / 'metadata.json')))

            all(map(lambda step: step.pop('fn', None), metadata['steps']))

            results = map(lambda path: simplejson.load(stack.enter_context(open(path))), benchmark_path.glob('fio-*-results.json'))
            results = map(lambda result: result['jobs'][0], results)

            def pick_iotype(result):
                if result['read']['runtime'] > 0:
                    return result['read']

                else:
                    return result['write']

            def iotype_results_filter(result):
                return {k: v for k, v in result.items() if k in IO_RESULT_KEYS}

            results = map(lambda result: (result['jobname'], iotype_results_filter(pick_iotype(result))), results)

            # def result_filter(k, v):
            #     if k in ('read', 'write'):
            #         return v['runtime'] > 0
            #
            #     return k in ('jobname', 'latency_us')
            #
            # results = map(lambda result: {k: v for k, v in result.items() if result_filter(k, v)}, results)

            metadata['results'] = dict(results)

            return metadata

    except:
        print('Failed to process benchmark results for', benchmark_path, file=sys.stderr)
        return None

def benchmark_name(benchmark):
    def step_name(step):
        def params_repr(parameters):
            if len(parameters) == 0:
                return ''

            return '[' + ','.join("{!s}={!r}".format(*p) if isinstance(p, list) else p for p in parameters) + ']'

        return step['name'] + params_repr(step['parameters'])

    return '+'.join(map(step_name, benchmark['steps']))

def plot(benchmarks):
    benchmarks = list(reversed(list(benchmarks)))

    plt.rcdefaults()
    plt.rcParams.update({'figure.autolayout': True})

    fig, ax = plt.subplots()

    positions = np.arange(len(benchmarks))
    names = list(map(benchmark_name, benchmarks))

    iotypes = ['SEQ_READ', 'SEQ_WRITE', 'RAND_READ', 'RAND_WRITE']

    height = 1 / (len(iotypes) + 5)

    for i, iotype in enumerate(iotypes):
        data = list(map(lambda b: b['results'][iotype]['bw'], benchmarks))

        ax.barh(positions + ((height * 1.5) - (i * height)), data, height=height, label=iotype)

    data = list(map(lambda b: sum(r['bw'] for r in b['results'].values()) / len(b['results']), benchmarks))
    ax.vlines(data, positions - (height * 2), positions + (height * 2), color='black', label="mean")

    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontfamily='monospace')

    ax.legend()

    fig.set_size_inches(18.27, .2 * len(benchmarks))

    fig.savefig('plot.pdf')


def main():
    parser = argparse.ArgumentParser(description='The Great Virtual Machine Disk Benchmark')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    parser.add_argument('input_directory', type=Path, metavar='INPUT-DIRECTORY', help='directory containing benchmarks.')
    parser.add_argument('output', type=Path, metavar='INPUT-DIRECTORY', help='directory containing benchmarks.')

    args = parser.parse_args()

    # print(args)

    def vbprint(*values):
        if (args.verbose):
            print(*values, file=sys.stderr)

    vbprint('Input directory: {}'.format(args.input_directory))
    vbprint('Input directory: {}'.format(args.input_directory))
    # if os.path.isdir(args.output_directory):
    #     vbprint('Output directory exists. Removing.')
    #     shutil.rmtree(args.output_directory)

    # os.makedirs(args.output_directory)

    benchmark_paths = map(lambda p: p.parent, args.input_directory.glob('**/metadata.json'))

    benchmark_paths_iter = tqdm(benchmark_paths, desc='Benchmarks', unit='benchmark')

    benchmarks = filter(None, map(process_benchmark, benchmark_paths_iter))
    #benchmarks = itertools.islice(benchmarks, 10)

    # benchmarks = sorted(benchmarks, key=lambda r: r['results']['SEQ_WRITE']['bw'], reverse=True)



    # list(print(benchmark_name(benchmark), benchmark['results']['SEQ_WRITE']['bw']) for benchmark in benchmarks)

    #plot(benchmarks)


    simplejson.dump(benchmarks, open('results.json', 'w'), iterable_as_array=True)

    return








if __name__ == '__main__':
    main()
