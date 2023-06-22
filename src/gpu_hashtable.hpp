#ifndef _HASHCPU_
#define _HASHCPU_

/**
 * Class GpuHashTable to implement functions
 */
struct entry {
	int key;
	int value;

	entry() {
		key = 0;
		value = 0;
	}
	entry(int k, int val) {
		key = k;
		value = val;
	}
};

class GpuHashTable
{
	private:
		entry *table;
		int maxSize;
		int currSize;

	public:
		GpuHashTable(int size);
		void reshape(int sizeReshape);

		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);

		~GpuHashTable();
};

#endif
