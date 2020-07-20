from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
import pprint
import glob, os
from random import seed
from random import randint
import time
import tracemalloc


class Aho:
    states_a = []
    states_b = []
    max_states = 0
    used_states = 0
    indexed = 0

    inverted = {'A': 0, 'T': 1, 'C': 2, 'G': 3}

    def __init__(self, max):
        n = 4
        self.max_states = max
        self.states_a = [[0 for i in range(n)] for j in range(max)]
        self.states_b = [[0 for i in range(n)] for j in range(max)]

    def print(self):
        for i in range(0, self.max_states):
            print("%i:\t\t" % i, end='')
            for j in range(0, 4):
                print ("(%i , %i)\t\t" % (self.states_a[i][j], self.states_b[i][j]), end='')
            print("")

    def insert(self, word):
        current_state = 0
        previous_state = 0
        choice = 0
        for character in word:
            previous_state = current_state
            choice = self.inverted[character]
            if self.states_a[current_state][choice] != 0:
                current_state = self.states_a[current_state][choice]
            else:
                self.used_states = self.used_states + 1
                self.states_a[current_state][choice] = self.used_states
                current_state = self.used_states
        self.indexed = self.indexed + 1
        self.used_states = self.used_states - 1
        self.states_b[previous_state][choice] = self.indexed

    def search(self, word):
        current_state = 0
        previous_state = 0
        choice = 0
        for character in word:
            previous_state = current_state
            choice = self.inverted[character]
            if self.states_a[current_state][choice] != 0:
                current_state = self.states_a[current_state][choice]
            else:
                return False
        if self.states_b[previous_state][choice] != 0:
            return True
        return False

    def openclSearch(self, dataset):
        tracemalloc.start()

        a_np = np.array(self.states_a)
        b_np = np.array(self.states_b)
        t_np = np.array(dataset)
        k_np = np.random.rand(t_np.size)

        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        mf = cl.mem_flags
        a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
        b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
        t_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t_np)

        prg = cl.Program(ctx, """
            
            __kernel void search(
                const unsigned int size, __global const float *a_g, __global const float *b_g, __global const char *t_g,
                 __global float *res_g) {
                    int current_state = 0;
                    int previous_state = 0;
                    int choice = 0;
                    
                    int i = get_global_id(0);
                    
                    for (int k = 0; k < size; k++){
                        previous_state = current_state;
                        switch (t_g[k + size * i]) {
                            case 'A': choice = 0;
                            break;
                            case 'T': choice = 1;
                            break;
                            case 'C': choice = 2;
                            break;
                            case 'G': choice = 3;
                            break;
                        }
                        
                        if(a_g[choice + 4 * current_state] != 0){
                            current_state = a_g[choice + 4 * current_state];
                        }else{
                            res_g[i] = 0;
                            break;
                        }
                    }
                    
                    if(b_g[choice + 4 * current_state] != 0){
                        res_g[i] = 1;
                    }else{
                        res_g[i] = 0;
                    }
            }
        """).build()

        start_time = time.time()
        res_g = cl.Buffer(ctx, mf.WRITE_ONLY, k_np.nbytes)
        prg.search(queue, t_np.shape, None, np.int32(6), a_g, b_g, t_g, res_g)

        res_np = np.empty_like(k_np)
        cl.enqueue_copy(queue, res_np, res_g)
        end_time = time.time()
        print("---Ejecutado en %s seconds ---" % (end_time - start_time))
        current, peak = tracemalloc.get_traced_memory()
        print(f"Uso de memoria actual {current / 10**6}MB; El pico fue {peak / 10**6}MB")
        tracemalloc.stop()


def randomstring(size=32):
    di = ['A', 'C', 'G', 'T']
    word = ''
    for i in range(0,32):
        word = word + di[randint(0, 3)]
    return word

def randomdataset(datasize=10, size=32):
    dataset = []
    for i in range(0, datasize):
        dataset.append(randomstring(size))
    return dataset

def runTests(t):
    z = 16
    dataset_test = randomdataset(t, z)
    dataset_pattern = randomdataset(t, z)
    with open('./datasets/pattern-%i-%i.txt' % (t, z), 'w') as f:
        for item in dataset_pattern:
            f.write("%s\n" % item)
    with open('./datasets/test-%i-%i.txt' % (t, z), 'w') as f:
        for item in dataset_test:
            f.write("%s\n" % item)

    stat = t*z*4
    print ("Creando aho para %i con % estados" % (t, stat))
    aho = Aho(t*z*4)
    for word in dataset_pattern:
        aho.insert(word)
    print("busqueda")
    aho.openclSearch(dataset_test)



def test():
    sizes = [150000]

    for test in sizes:
        runTests(test)


test()
# aho.print()
# aho.insert("AAATCG")
# aho.insert("TACGCC")
# aho.insert("AAATTG")
# aho.print()
# print(aho.search("AAATCG"))
# print(aho.search("AATCGC"))
# print(aho.search("AAATCC"))
# print(aho.search("GAATCG"))
# print(aho.search("TACGCC"))
# print(aho.search("AAATTG"))
# aho.openclSearch()