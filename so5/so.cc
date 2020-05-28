#include "so.h"
#include <algorithm>
#include <iostream>

using namespace std;

void swap(data_t* a, int i, int j) {
    data_t t = a[i]; a[i] = a[j]; a[j] = t;
}

int median(data_t* a, int i1, int i2, int i3) {
    if (a[i1] <= a[i2]) {
        if (a[i2] <= a[i3])
            return i2;
        else if (a[i3] <= a[i1])
            return i1;
        else return i3;
    } else {
        if (a[i1] <= a[i3])
            return i1;
        else if (a[i3] <= a[i2])
            return i2;
        else
            return i3;
    }
}

int partition(data_t* a, int lo, int hi) {
    int pivotIndex = median(a, lo, (lo+hi)/2, hi);
    data_t pivot = a[pivotIndex];
    swap(a, pivotIndex, hi);
    int i = lo - 1;
    int j = lo;
    while (j < hi) {
        if(a[j] <= pivot) {
            i++;
            swap(a, i, j);
        }
        j++;
    }
    swap(a, i + 1, hi);
    return i + 1;
}

void quickSort(data_t* a, int lo, int hi) {
    int threshold = 32;
    if (hi - lo < threshold)
        sort(a + lo, a + hi + 1);
    else {
        int j = partition(a, lo, hi);
        if (lo < j - 1)
            #pragma omp task
            quickSort(a, lo, j - 1);
        if (j + 1 < hi)
            #pragma omp task
            quickSort(a, j + 1, hi);
    }
}

void psort(int n, data_t* data) {
    if(n >= 2)
        #pragma omp parallel
        #pragma omp single
        quickSort(data, 0, n - 1);
}
