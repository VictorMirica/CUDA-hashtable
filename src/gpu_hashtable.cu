#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctime>
#include <sstream>
#include <string>
#include "test_map.hpp"
#include "gpu_hashtable.hpp"

#define LOAD_FACTOR 0.9
#define RESIZE_FACTOR 2

using namespace std;

/*
Allocate CUDA memory only through glbGpuAllocator
cudaMalloc -> glbGpuAllocator->_cudaMalloc
cudaMallocManaged -> glbGpuAllocator->_cudaMallocManaged
cudaFree -> glbGpuAllocator->_cudaFree
*/

/**
 * Function constructor GpuHashTable
 * Performs init
 * Example on using wrapper allocators _cudaMalloc and _cudaFree
 */

// Function that handles cuda errors
static void HandleError(cudaError_t err) {
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

// Function for calculating the hash of a key
__device__ int calculateHash(int key, int size) {
	return key % size;
}

__host__ void getBlocksAndThreads(int numKeys, int* blocks, int* threads) {
	cudaDeviceProp prop;
	HandleError(cudaGetDeviceProperties(&prop, 0));

	*threads = prop.maxThreadsPerBlock;
	*blocks =  numKeys / *threads + (numKeys % *threads == 0 ? 0 : 1);
}

// Kernel for inser ting a batch of keys and values
__global__ void insertBatchKernel(entry* table, int* keys, int* values, int numKeys, int size, int* currSize) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < numKeys) {
		int key = keys[index];
		int value = values[index];
		int hash = calculateHash(key, size);
		entry* current = table + hash;
		while (true) {
			int currKey = atomicCAS(&(current->key), 0, key);
			if (currKey == 0 || currKey == key) {
				atomicExch(&(current->value), value);
				if (currKey == 0) {
					atomicAdd(currSize, 1);
				}
				break;
			} else {
				hash = calculateHash(hash + 1, size);
				current = table + hash;
			}
		}
	}
}

// Kernel for getting a batch of keys and values
__global__ void getBatchKernel(entry* table, int* keys, int* values, int numKeys, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < numKeys) {
		int key = keys[index];
		int hash = calculateHash(key, size);
		entry* current = table + hash;
		while (true) {
			int currKey = current->key;
			if (currKey == key) {
				atomicExch(&(values[index]), current->value);
				break;
			} else if (currKey == 0) {
				break;
			} else {
				hash = calculateHash(hash + 1, size);
				current = table + hash;
			}
		}
	}
}

__global__ void reshapeTableKernel(entry* oldTable, entry* newTable, int oldSize, int newSize) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < oldSize) {
		entry* current = oldTable + index;
		if (current->key != 0) {
			int hash = calculateHash(current->key, newSize);
			entry* newCurrent = newTable + hash;
			while (true) {
				int newCurrKey = newCurrent->key;
				if (newCurrKey == 0) {
					int currKey = current->key;
					int currValue = current->value;
					if (atomicCAS(&(newCurrent->key), 0, currKey) == 0) {
						atomicExch(&(newCurrent->value), currValue);
						break;
					}
				} else {
					hash = calculateHash(hash + 1, newSize);
					newCurrent = newTable + hash;
				}
			}
		}
	}
}

GpuHashTable::GpuHashTable(int size) {
	int tableSize = size * sizeof(entry);
	entry* table;

	// Allocate memory for the table
	HandleError(glbGpuAllocator->_cudaMallocManaged((void**)&table, tableSize));

	// Initialize the table
	HandleError(cudaMemset((void*)table, 0, tableSize));

	this->table = table;
	maxSize = size;
	currSize = 0;
}

/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
	HandleError(glbGpuAllocator->_cudaFree((void*)table));
}

/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int tableSize = numBucketsReshape * sizeof(entry);
	entry* newTable;

	// Allocate memory for the new table
	HandleError(glbGpuAllocator->_cudaMallocManaged((void**)&newTable, tableSize));

	// Initialize the new table
	HandleError(cudaMemset((void*)newTable, 0, tableSize));

	// Write the old table to the new table
	int numBlocks, numThreads;
	getBlocksAndThreads(maxSize, &numBlocks, &numThreads);
	reshapeTableKernel<<<numBlocks, numThreads>>>(table, newTable, maxSize, numBucketsReshape);
	HandleError(cudaDeviceSynchronize());

	// Free the old table
	HandleError(glbGpuAllocator->_cudaFree((void*)table));

	table = newTable;
	maxSize = numBucketsReshape;
}

/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	// Allocate memory for the keys and values
	int keysSize = numKeys * sizeof(int);
	int valuesSize = numKeys * sizeof(int);
	int* d_keys = 0;
	int* d_values = 0;
	
	HandleError(glbGpuAllocator->_cudaMalloc((void**)&d_keys, keysSize));
	HandleError(glbGpuAllocator->_cudaMalloc((void**)&d_values, valuesSize));

	// Copy the keys and values to the device
	HandleError(cudaMemcpy(d_keys, keys, keysSize, cudaMemcpyHostToDevice));
	HandleError(cudaMemcpy(d_values, values, valuesSize, cudaMemcpyHostToDevice));

	// Check if the table needs to be resized
	if (currSize + numKeys > maxSize * LOAD_FACTOR) {
		reshape(maxSize * RESIZE_FACTOR);
	}

	// Copy the current size to the device
	int* d_currSize = 0;
	HandleError(glbGpuAllocator->_cudaMalloc((void**)&d_currSize, sizeof(int)));
	HandleError(cudaMemcpy(d_currSize, &currSize, sizeof(int), cudaMemcpyHostToDevice));

	// Insert the keys and values
	int numBlocks, numThreads;
	getBlocksAndThreads(numKeys, &numBlocks, &numThreads);
	insertBatchKernel<<<numBlocks, numThreads>>>(table, d_keys, d_values, numKeys, maxSize, d_currSize);
	HandleError(cudaDeviceSynchronize());

	// Copy the current size back to the host
	HandleError(cudaMemcpy(&currSize, d_currSize, sizeof(int), cudaMemcpyDeviceToHost));

	// Free the memory
	HandleError(glbGpuAllocator->_cudaFree(d_keys));
	HandleError(glbGpuAllocator->_cudaFree(d_values));
	HandleError(glbGpuAllocator->_cudaFree(d_currSize));

	return true;
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	// Allocate memory for the keys and values
	int keysSize = numKeys * sizeof(int);
	int* d_keys;
	int* d_values;
	HandleError(glbGpuAllocator->_cudaMalloc((void**)&d_keys, keysSize));
	HandleError(glbGpuAllocator->_cudaMalloc((void**)&d_values, keysSize));

	// Copy the keys to the device
	HandleError(cudaMemcpy(d_keys, keys, keysSize, cudaMemcpyHostToDevice));

	// Get the values
	int numBlocks;
	int numThreads;
	getBlocksAndThreads(numKeys, &numBlocks, &numThreads);
	getBatchKernel<<<numBlocks, numThreads>>>(table, d_keys, d_values, numKeys, maxSize);
	HandleError(cudaDeviceSynchronize());

	// Copy the values to the host
	int* values = (int*)malloc(keysSize);
	HandleError(cudaMemcpy(values, d_values, keysSize, cudaMemcpyDeviceToHost));

	// Free the memory
	HandleError(glbGpuAllocator->_cudaFree(d_keys));
	HandleError(glbGpuAllocator->_cudaFree(d_values));

	return values;
}
