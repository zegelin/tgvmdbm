#!/bin/python

#  _____ ______     ____  __ ____  ____  __  __
# |_   _/ ___\ \   / /  \/  |  _ \| __ )|  \/  |
#   | || |  _ \ \ / /| |\/| | | | |  _ \| |\/| |
#   | || |_| | \ V / | |  | | |_| | |_) | |  | |
#   |_| \____|  \_/  |_|  |_|____/|____/|_|  |_|
# 
#  The great virtual machine disk benchmark
#
from datetime import datetime
import importlib.util
import os
import shlex
import shutil
import socket
import tempfile
import traceback
from enum import Enum, auto
import inspect
import sys
from collections import namedtuple, OrderedDict
from contextlib import contextmanager, ExitStack
from functools import wraps
import itertools
from typing import List
from pprint import pprint
import json
from tqdm import tqdm
import time
import argparse
import re
import simplejson



NVME_DEVICE = '/dev/nvme0n1'

Context = namedtuple('Context', ['target', 'working_directory', 'log_file', 'subprocess', 'set_status'], defaults=[None, None, None, None, None])


class BenchmarkStep(namedtuple('BenchmarkStep', ['name', 'parameters', 'fn'])):
    __slots__ = ()

    def __str__(self):
        def params_repr():
            if len(self.parameters) == 0:
                return ''

            return '[' + ','.join("{!s}={!r}".format(*p) if isinstance(p, tuple) else str(p) for p in self.parameters) + ']'

        return self.name + params_repr()

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class Benchmark(namedtuple('Benchmark', ['steps'])):
    __slots__ = ()

    def __str__(self):
        return '+'.join(map(str, self.steps))


def benchmark_step(name, *args, **kwargs):
    def decorator(step_fn):
        return BenchmarkStep(name=str(name), parameters=args+tuple(kwargs.items()), fn=contextmanager(step_fn))

    return decorator


BlockDeviceTarget = namedtuple('BlockDeviceTarget', ['device'])
FilesystemTarget = namedtuple('FilesystemTarget', ['mountpoint'])
ZpoolTarget = namedtuple('ZpoolTarget', ['name'])
QemuImageTarget = namedtuple('QemuImageTarget', ['file', 'format'])

def check_nvme(target: BlockDeviceTarget, subprocess):
    # check that device isn't mounted/in-use
    lsblk_result = subprocess.run(['lsblk', '-J', target.device], capture_output=True, check=True)
    lsblk_result = simplejson.loads(lsblk_result.stdout)

    if any(map(lambda b: b['mountpoint'] is not None, lsblk_result['blockdevices'])):
        raise RuntimeError('NVMe device {} is mounted.'.format(target.device))

    if any(map(lambda b: b.get('children') is not None, lsblk_result['blockdevices'])):
        raise RuntimeError('NVMe device {} is formatted with a partition table.'.format(target.device))


def format_nvme():
    @benchmark_step('nvme')
    def format_nvme_step(target: BlockDeviceTarget, subprocess, set_status, **kwargs):
        assert isinstance(target, BlockDeviceTarget)

        check_nvme(target, subprocess)

        # log some SMART info
        set_status('NVMe SMART')
        subprocess.check_call(['nvme', 'smart-log', target.device])

        set_status('NVMe low-level format')
        #subprocess.check_call(['nvme', 'format', target.device])

        yield Context(target=target)

    yield format_nvme_step


def mkfs_helper(type, mkfs_args=(), mountpoint=os.path.join('/mnt', 'tgvmdbm')):
    def wrapper(target: BlockDeviceTarget, subprocess, set_status, **kwargs):
        assert isinstance(target, BlockDeviceTarget)

        set_status('Format FS')
        subprocess.check_call(['mkfs', '-t', type] + list(mkfs_args) + [target.device])

        set_status('Mount FS')
        subprocess.check_call(['mkdir', '-p', mountpoint])  # use subprocess vs os.makedirs so SSH is supported
        subprocess.check_call(['mount', target.device, mountpoint])

        try:
            yield Context(target=FilesystemTarget(mountpoint=mountpoint))

        finally:
            subprocess.check_call(('umount', mountpoint))
            subprocess.check_call(['wipefs', '--all', target.device])

    return wrapper


def mkfs_xfs(blocksizes=(512, 4096), **kwargs):
    yield benchmark_step('xfs')(mkfs_helper('xfs', **kwargs))


def mkfs_ext4(**kwargs):
    yield benchmark_step('ext4')(mkfs_helper('ext4', **kwargs))


def mkfs(filesystems=(mkfs_xfs(), mkfs_ext4())):
    return itertools.chain(*filesystems)


def format_zpool(name='tgvmdbm-pool', ashift_values=(9, 12, 13), compression_values=('off', 'lz4')):
    for ashift in ashift_values:
        for compression in compression_values:
            @benchmark_step('zpool', ashift=ashift, c=compression)
            def format_zpool_step(target: BlockDeviceTarget, subprocess,
                                  ashift=ashift, compression=compression,
                                  **kwargs):
                assert isinstance(target, BlockDeviceTarget)

                subprocess.check_call(['zpool', 'create',
                                       '-o', 'ashift={}'.format(ashift), '-m', 'none',
                                       '-O', 'compression={}'.format(compression),
                                       name, target.device])

                try:
                    yield Context(target=ZpoolTarget(name=name))

                finally:
                    subprocess.check_call(['zpool', 'destroy', '-f', name])
                    subprocess.check_call(['wipefs', '--all', target.device]) # zpool create makes a GPT table -- wipe it

            yield format_zpool_step


def create_zfs_dataset(name='dataset', mountpoint='/mnt/tgvmdbm', record_sizes=(512, 4096, 8192, '128K')):
    for recordsize in record_sizes:
        @benchmark_step('zfs', rs=recordsize)
        def create_zfs_dataset_step(target: ZpoolTarget, subprocess, recordsize=recordsize, **kwargs):
            assert isinstance(target, ZpoolTarget)

            subprocess.check_call(['zfs', 'create',
                                   '-o', 'recordsize={}'.format(recordsize),
                                   '-o', 'mountpoint={}'.format(mountpoint),
                                   '{}/{}'.format(target.name, name)])

            yield Context(target=FilesystemTarget(mountpoint=mountpoint))

        yield create_zfs_dataset_step


def create_zvol(name='zvol', blocksizes=(512, 4096, 8192)):
    for volblocksize in blocksizes:
        @benchmark_step('zvol', vbs=volblocksize)
        def create_zvol_step(target: ZpoolTarget, subprocess, volblocksize=volblocksize, **kwargs):
            assert isinstance(target, ZpoolTarget)

            subprocess.check_call(['zfs', 'create',
                                   '-o', 'volblocksize={}'.format(volblocksize),
                                   '-V', '50G', '{}/{}'.format(target.name, name)])

            yield Context(target=BlockDeviceTarget(device=os.path.join('/dev', 'zvol', target.name, name)))

        yield create_zvol_step


def qemu_img_helper(format, options={}, size='50G', filename=None):
    if filename is None:
        filename = 'image.{}'.format(format)

    def wrapper(target: FilesystemTarget, subprocess, set_status, **kwargs):
        assert isinstance(target, FilesystemTarget)

        args = ['qemu-img', 'create', '-f', format]

        if len(options) > 0:
            args.extend(['-o', ','.join("{!s}={!r}".format(k,v) for (k,v) in options.items())])

        file = os.path.join(target.mountpoint, filename)

        args.extend([file, size])

        set_status('Creating QEMU image')
        subprocess.check_call(args)

        yield Context(target=QemuImageTarget(file=file, format=format))

    return wrapper


def qemu_img_qcow2(cluster_sizes=(512, )):
    for cluster_size in cluster_sizes:
        yield benchmark_step('img', 'qcow2', cs=cluster_size)(qemu_img_helper('qcow2', options={'cluster_size': cluster_size}))


def qemu_img_raw():
    yield benchmark_step('img', 'raw')(qemu_img_helper('raw'))


def qemu_img(formats=(qemu_img_qcow2(), qemu_img_raw())):
    return itertools.chain(*formats)


class QemuStorageControllerDevice(Enum):
    VIRTIO_BLK_PCI = 'virtio-blk-pci'
    VIRTIO_SCSI_PCI = 'virtio-scsi-pci'

    def __str__(self):
        return self.value


def qemu(storage_controllers=QemuStorageControllerDevice,
         write_cache_values=(True, False),
         direct_io_values=(True, False),
         iothreads_values=(True, False),
         aio_backends=('threads', 'native'),
         vm_resources_dir='/home/adam/busybox-vm/buildroot-2018.05.2/output/images'):

    def qemu_bool(b: bool):
        return 'on' if b is True else 'off'

    for storage_controller in storage_controllers:
        for write_cache in write_cache_values:
            for direct_io in direct_io_values:
                for iothreads in iothreads_values:
                    for aio_backend in aio_backends:

                        if aio_backend == 'native' and not direct_io:
                            continue

                        @benchmark_step('qemu', storage_controller, wc=write_cache, dio=direct_io, iot=iothreads, aio=aio_backend)
                        def qemu_step(target, subprocess, set_status, log_file, working_directory,
                                      storage_controller=storage_controller,
                                      write_cache=write_cache, direct_io=direct_io,
                                      iothreads=iothreads, aio_backend=aio_backend,
                                      **kwargs):
                            # symlink the working directory to /tmp
                            # I spent to long trying to correctly quote the pathname for the smb=XYZ attribute of -netdev...
                            working_directory_link = tempfile.mktemp()
                            os.symlink(os.path.abspath(working_directory), working_directory_link)

                            args = ['qemu-system-x86_64',
                                    '-nodefconfig', '-no-user-config', '-nodefaults', '-nographic',
                                    '-machine', 'q35,accel=kvm', '-enable-kvm', '-cpu', 'host',
                                    '-smp', '4,cores=4,threads=1,sockets=1',
                                    '-m', '512',

                                    '-kernel', os.path.join(vm_resources_dir, 'bzImage'),
                                    '-append', 'quiet console=ttyS0,115200n8',
                                    '-initrd', os.path.join(vm_resources_dir, 'rootfs.cpio'),

                                    '-serial', 'stdio',  # write VM console message to logs

                                    '-netdev', 'user,id=netdev0,smb={},hostfwd=tcp::8022-:22'.format(working_directory_link),
                                    '-device', 'virtio-net,netdev=netdev0,mac=52:5a:01:15:34:57'
                                    ]

                            # block device
                            blockdev_cache_param = ',cache.direct={}'.format(qemu_bool(direct_io))

                            if isinstance(target, BlockDeviceTarget):
                                args.extend([
                                    '-blockdev', 'node-name=block0,driver=file,filename={},aio={}'.format(target.device, aio_backend) +
                                                 blockdev_cache_param,
                                    '-blockdev', 'node-name=block1,driver=raw,file=block0' +
                                                 blockdev_cache_param
                                ])

                                blockdev = 'block1'

                            elif isinstance(target, QemuImageTarget):
                                args.extend([
                                    '-blockdev', 'node-name=block0,driver=file,filename={},aio={}'.format(target.file, aio_backend) +
                                                 blockdev_cache_param,
                                    '-blockdev', 'node-name=block1,driver={},file=block0'.format(target.format) +
                                                 blockdev_cache_param
                                ])

                                blockdev = 'block1'

                            else:
                                raise NotImplementedError

                            # storage controller
                            device_cache_param = ',write-cache={}'.format(qemu_bool(write_cache))

                            if iothreads is True:
                                args.extend(['-object', 'iothread,id=iothread0'])
                                iothread_param = ',iothread=iothread0'

                            else:
                                iothread_param = ''

                            if storage_controller == QemuStorageControllerDevice.VIRTIO_BLK_PCI:
                                args.extend([
                                    '-device', 'virtio-blk-pci,drive={}'.format(blockdev) +
                                               device_cache_param + iothread_param
                                ])

                                new_target = BlockDeviceTarget(device="/dev/vda")

                            elif storage_controller == QemuStorageControllerDevice.VIRTIO_SCSI_PCI:
                                args.extend([
                                    '-device', 'virtio-scsi-pci,id=scsi0' + iothread_param,
                                    '-device', 'scsi-hd,drive={},bus=scsi0.0'.format(blockdev) +
                                               device_cache_param
                                ])

                                new_target = BlockDeviceTarget(device="/dev/sda")

                            else:
                                raise NotImplementedError


                            # boot VM
                            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as doorbell_socket:
                                doorbell_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                                doorbell_socket.bind(('localhost', 34234))
                                doorbell_socket.settimeout(5)
                                doorbell_socket.listen(1)

                                qemu_process = subprocess.Popen(args)

                                try:
                                    set_status('Waiting for QEMU VM to start')

                                    # wait for the doorbell
                                    try:
                                        (client_socket, address) = doorbell_socket.accept()
                                        client_socket.close()
                                    except socket.timeout:
                                        if qemu_process.poll() is None:
                                            raise Exception("qemu VM failed to ring the doorbell.")

                                        else:
                                            raise Exception("qemu exited with code {}.".format(qemu_process.returncode))

                                    # by this point, SSH should work...

                                    ssh_args = ['ssh',
                                                '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null',
                                                '-p', '8022', 'root@127.0.0.1', '--']

                                    yield Context(target=new_target,
                                                  subprocess=patch_subprocess(log_file=log_file, args_prefix=ssh_args),
                                                  working_directory='/root')

                                finally:
                                    qemu_process.terminate()
                                    qemu_process.wait()

                        yield qemu_step


class FioIOType(Enum):
    SEQ_READ = ['-rw=read']
    SEQ_WRITE = ['-rw=write']
    RAND_READ = ['-rw=randread']
    RAND_WRITE = ['-rw=randwrite']

    def __init__(self, args):
        self.args = args


def fio_benchmark(blocksizes=(512, 4096, 8192), io_types=FioIOType, directio=True, runtime=10):
    for blocksize in blocksizes:
        @benchmark_step('fio', 'direct' if directio else 'buffered', bs=blocksize)
        def fio_benchmark_step(target, subprocess, set_status, working_directory, blocksize=blocksize, **kwargs):
            common_args = ['fio']

            if isinstance(target, BlockDeviceTarget):
                common_args.append('--filename={}'.format(target.device))

            elif isinstance(target, FilesystemTarget):
                common_args.extend([
                    '--filename={}/fio'.format(target.mountpoint),
                    '--size=1G'
                ])

            else:
                raise NotImplementedError

            common_args.extend([
                '--ioengine=libaio',
                '--iodepth=32',
                '--runtime={}'.format(runtime),
                '--direct={}'.format(int(directio)),
                '--blocksize=%s' % blocksize,
                '--output-format=json',
                # '--parse-only'
            ])

            for io_type in tqdm(io_types, desc='fio'):
                args = list(common_args)
                args.extend([
                    '--output={}'.format(os.path.join(working_directory, 'fio-{}-results.json'.format(io_type.name))),
                    '--name={}'.format(io_type.name)
                ])
                args.extend(io_type.args)

                set_status(io_type.name)
                subprocess.check_call(args)

            yield

        yield fio_benchmark_step


# benchmarks = [
#     [fio_benchmark()],
#
#     [mkfs(), fio_benchmark()],
#     [format_zpool(), create_zfs_dataset(), fio_benchmark(directio=False)],
#
#     [format_zpool(), create_zvol(), fio_benchmark()],
#
#     [qemu(), fio_benchmark()],  # "passthrough"
#
#     [mkfs(), qemu_img(), qemu(), fio_benchmark()],
#     [format_zpool(), create_zfs_dataset(), qemu_img(), qemu(direct_io_values=(False, )), fio_benchmark()],
#
#     [format_zpool(), create_zvol(), qemu(), fio_benchmark()],
# ]

benchmarks = [
    [fio_benchmark()],

    #[mkfs(), fio_benchmark()],
    #[format_zpool(), create_zfs_dataset(), fio_benchmark(directio=False)],

    #[format_zpool(), create_zvol(), fio_benchmark()],

    #[qemu(), fio_benchmark()],  # "passthrough"

    #[mkfs(), qemu_img(), qemu(), fio_benchmark()],
    #[format_zpool(), create_zfs_dataset(), qemu_img(), qemu(direct_io_values=(False, )), fio_benchmark()],

    [format_zpool(), create_zvol(blocksizes=(8192, 16384)), qemu(storage_controllers=(QemuStorageControllerDevice.VIRTIO_SCSI_PCI, ), write_cache_values=(True, ), direct_io_values=(True, ), iothreads_values=(True, )), fio_benchmark(blocksizes=(8192, ))],
]

# every group needs to format the NVMe device
benchmarks = tuple(map(lambda steps: [format_nvme()] + steps, benchmarks))


def materialize_benchmark(group):
    sequences = tuple(itertools.product(*group))

    return (Benchmark(steps=steps) for steps in sequences)


benchmarks = tuple(itertools.chain(*map(materialize_benchmark, benchmarks)))


def run_benchmarks(benchmarks: List[Benchmark], output_directory):
    print('{} benchmarks to run.'.format(len(benchmarks)))
    print('On your marks... Get set... Go!')

    postfix = OrderedDict(Failed=0)

    benchmarks_iter = tqdm(benchmarks, desc='Benchmarks', unit='benchmark', postfix=postfix)

    with open(os.path.join(output_directory, 'log.txt'), 'w') as log_file:
        for (i, benchmark) in enumerate(benchmarks_iter):
            benchmark_dir = os.path.join(output_directory, '{}-{}'.format(i, benchmark))
            os.makedirs(benchmark_dir, exist_ok=True)

            try:
                print('[{}] Benchmark {} \'{}\' started.'.format(datetime.now(), i, benchmark), file=log_file)
                run_benchmark(benchmark, benchmark_dir)
                print('[{}] Benchmark {} \'{}\' complete.'.format(datetime.now(), i, benchmark), file=log_file)

            except KeyboardInterrupt:
                return

            except:
                print('[{}] Benchmark {} \'{}\' failed.'.format(datetime.now(), i, benchmark), file=log_file)

                postfix["Failed"] += 1
                benchmarks_iter.set_postfix(ordered_dict=postfix)


def load_subprocess_module():
    subprocess_spec = importlib.util.find_spec('subprocess')
    subprocess = importlib.util.module_from_spec(subprocess_spec)
    subprocess_spec.loader.exec_module(subprocess)

    return subprocess


def patch_subprocess(log_file, args_prefix=()):
    subprocess = load_subprocess_module()
    _popen_communicate = subprocess.Popen.communicate

    def make_popen_init(_popen_init):
        def popen_init(self, args, *argv, **kwargs):
            args = list(args_prefix) + list(args)
            print('\n>', ' '.join(map(shlex.quote, args)), file=log_file, flush=True)

            # self.returncode = 0
            # self.stdout, self.stderr, self.stdin = None, None, None

            if 'stdout' in kwargs:
                print('<stdout captured>', file=log_file)

            if 'stderr' in kwargs:
                print('<stderr captured>', file=log_file)

            kwargs.setdefault('stdout', log_file)
            kwargs.setdefault('stderr', log_file)

            _popen_init(self, args, *argv, **kwargs)

        return popen_init

    def popen_communicate(*args, **kwargs):
        (stdout_data, stderr_data) = _popen_communicate(*args, **kwargs)

        print('\nstdout:', file=log_file, flush=True)
        print('<no data>', file=log_file, flush=True) if len(stdout_data) == 0 else log_file.write(
            str(stdout_data, encoding='utf-8'))

        print('\nstderr:', file=log_file, flush=True)
        print('<no data>', file=log_file, flush=True) if len(stderr_data) == 0 else log_file.write(
            str(stderr_data, encoding='utf-8'))

        return (stdout_data, stderr_data)

    subprocess.Popen.__init__ = make_popen_init(subprocess.Popen.__init__)
    subprocess.Popen.communicate = popen_communicate

    return subprocess


def run_benchmark(benchmark: Benchmark, working_directory):
    # save benchmark metadata
    with open(os.path.join(working_directory, 'metadata.json'), 'w') as metadata_file:
        def default(o):
            if callable(o):
                return repr(o)

            if isinstance(o, Enum):
                return o.value

        simplejson.dump(benchmark, metadata_file, default=default, indent=4)

    # run benchmark
    with open(os.path.join(working_directory, 'log.txt'), 'w') as log_file:
        context_stack = []

        try:
            # use an ExitStack so that each step of the benchmark may perform cleanup
            with ExitStack() as exit_stack:
                steps_iter = tqdm(benchmark.steps, file=sys.stdout, desc=str(benchmark), unit='step')

                # patch subprocess module to print commands on Popen
                subprocess = patch_subprocess(log_file)

                # build the initial Context/stack for the steps
                context = Context(
                    target=BlockDeviceTarget(device=NVME_DEVICE),
                    subprocess=subprocess,
                    working_directory=working_directory,
                    log_file=log_file
                )

                # run the steps
                for step in steps_iter:
                    steps_iter.set_postfix_str('')

                    context_stack.append(context)

                    context = context._replace(set_status=lambda status: steps_iter.set_postfix_str(
                        '{}: {}'.format(step.name, status)))


                    # run the benchmark step
                    new_context = exit_stack.enter_context(step(**context._asdict()))
                    new_context = Context() if new_context is None else new_context

                    # merge contexts: pick new context value if not None, else use old value (may be None)
                    context = Context(*map(
                        lambda values: next((item for item in values if item is not None), None),
                        zip(new_context, context)
                    ))

        except KeyboardInterrupt:
            raise

        except:
            print('\n\n-----\nBenchmark {} failed.'.format(benchmark), file=log_file)
            traceback.print_exc(file=log_file)

            print('\nContext stack:', file=log_file)
            pprint(context_stack, stream=log_file)

            raise





""""0-3,4,5" -> {0, 1, 2, 3, 4, 5}"""
SELECTOR_SPEC_RE = re.compile('(\d+)(?:-(\d+))?$')


def selector(range):
    def fn(spec):
        try:
            if spec == '*':
                return list(range)

            sel = map(lambda s: SELECTOR_SPEC_RE.match(s).groups(), spec.split(','))
            sel = map(lambda m: list(map(int, filter(None, m))), sel)
            sel = map(lambda r: range(r[0], r[1] + 1) if len(r) == 2 else r, sel)

            # set to remove duplicates, sorted to run in-order
            sel = list(sorted(set(itertools.chain(*sel))))

            if sel[0] < range[0] or sel[-1] > range[-1]:
                raise argparse.ArgumentTypeError('selector ({}) out of range [{}, {}]'
                                                 .format(spec, range.start, range.stop - 1))

            return sel

        except AttributeError:
            raise argparse.ArgumentTypeError('invalid selector ({})'.format(spec))

    return fn


def list_benchmarks(benchmarks: List[Benchmark]):
    for (group, items) in itertools.groupby(enumerate(benchmarks), key=lambda x: x[1].group):
        print('{}'.format(group.name), file=sys.stderr)

        for (index, benchmark) in items:
            print('\t{:>3}\t{}'.format(index, benchmark.name), file=sys.stderr)


class ListBenchmarksAction(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
        super(ListBenchmarksAction, self).__init__(
            option_strings=option_strings, dest=dest, default=default, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        list_benchmarks(benchmarks)

        exit(0)


def default_output_directory():
    return os.path.join(os.getcwd(), 'benchmark-{}'.format(int(time.time())))


def main():
    parser = argparse.ArgumentParser(description='The Great Virtual Machine Disk Benchmark')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    parser.add_argument('-l', '--list', action=ListBenchmarksAction, help='list benchmarks and exit')
    # parser.add_argument('-b', '--benchmark', type=selector(range(0, len(benchmarks))), default='*',
    #                     help='select benchmarks')
    parser.add_argument('output_directory', metavar='OUTPUT-DIRECTORY', nargs='?', default=default_output_directory(),
                        help='directory to output benchmarks (default: "%(default)s"). '
                             'this path will be created if necessary')

    args = parser.parse_args()

    # print(args)

    def vbprint(*values):
        if (args.verbose):
            print(*values, file=sys.stderr)

    vbprint('Output directory: {}'.format(args.output_directory))
    if os.path.isdir(args.output_directory):
        vbprint('Output directory exists. Removing.')
        shutil.rmtree(args.output_directory)

    os.makedirs(args.output_directory)

    check_nvme(target=BlockDeviceTarget(device=NVME_DEVICE), subprocess=load_subprocess_module())

    # benchmarks = OrderedDict(map(benchmarks.items().))

    # x = OrderedDict(itertools.chain(*(benchmark.items() for benchmark in benchmarks.values())))
    run_benchmarks(benchmarks, args.output_directory)

    print('Done!\n')


if __name__ == '__main__':
    main()