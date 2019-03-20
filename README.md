TGVMDBM
=======

The Great Virtual Machine Disk Benchmark

---


This was a project I was working on in my spare time for a while.
The idea was to determine the "best" disk configuration for a qemu-backed VM.

Essentially it comprises of:
- a Python script that runs every benchmark config permutation (of which there are 1000's)
- a buildroot external config that builds a tiny and fast-booting VM image with the required packages to run the benchmarks.
- the initial work of creating a webapp to visualise the results.

Maybe one day I'll have more time to work on this.


