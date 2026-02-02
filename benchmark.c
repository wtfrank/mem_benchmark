#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <pthread.h>
#include <unistd.h>

#define MEM_SIZE (32ULL * 1024 * 1024 * 1024) // 16GB
#define PAGE_SIZE 4096

typedef struct {
    void *src;
    void *dest;
    size_t size;
} thread_args_t;

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void* threaded_copy(void* args) {
    thread_args_t *t = (thread_args_t*)args;
    memcpy(t->dest, t->src, t->size);
    return NULL;
}

void run_test(const char* name, void* src, void* dest, size_t size, int threads) {
    double start = get_time();
    
    if (threads == 1) {
        memcpy(dest, src, size);
    } else {
        pthread_t t_ids[threads];
        thread_args_t t_args[threads];
        size_t chunk = size / threads;
        for (int i = 0; i < threads; i++) {
            t_args[i].src = (char*)src + (i * chunk);
            t_args[i].dest = (char*)dest + (i * chunk);
            t_args[i].size = chunk;
            pthread_create(&t_ids[i], NULL, threaded_copy, &t_args[i]);
        }
        for (int i = 0; i < threads; i++) pthread_join(t_ids[i], NULL);
    }

    double end = get_time();
    double duration = end - start;
    double gb_per_sec = (size / (1024.0 * 1024.0 * 1024.0)) / duration;
    printf("%-35s: %.4f sec (%.2f GB/s)\n", name, duration, gb_per_sec);
}

// #define FD_MAPPED 1
#define SHM_SEG_NAME "bench_shm"
int main() {
    // 1. Setup Shared Memory (Source)
    
    #ifdef FD_MAPPED
    printf("fd backed shared memory.\n");
    int fd = shm_open(SHM_SEG_NAME, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (0 > ftruncate(fd, MEM_SIZE)) {
      printf("ftruncate failed:%s\n", strerror(errno));
    }
    void* shm_src = mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    #else
    printf("anonymous shared memory.\n");
    void* shm_src = mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    #endif
    printf("Ensuring shared memory pages are mapped in\n");
    memset(shm_src, 0xAB, MEM_SIZE);

    printf("Benchmarking %lldGB Copy...\n\n", MEM_SIZE/1024/1024/1024);

    // Test 1: Naive memcpy (Small Pages, Cold)
    void* dest1 = mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    run_test("Naive memcpy (Small/Cold)", shm_src, dest1, MEM_SIZE, 1);
    munmap(dest1, MEM_SIZE);

    // Test 2: Huge Pages
    // Note: Requires transparent_hugepage/enabled = always OR mmap with MAP_HUGETLB
    void* dest2 = mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (dest2 == MAP_FAILED) {
        printf("Huge Pages: Not supported (check /proc/meminfo)\n");
    } else {
        run_test("Huge Pages (Cold)", shm_src, dest2, MEM_SIZE, 1);
        munmap(dest2, MEM_SIZE);
    }

    // Test 2b: Huge Pages warmed
    void* dest2b = mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (dest2b == MAP_FAILED) {
        printf("Huge Pages: Not supported (check /proc/meminfo)\n");
    } else {
        for (size_t i = 0; i < MEM_SIZE; i += PAGE_SIZE) ((char*)dest2b)[i] = 0; 
        run_test("Huge Pages (Warm)", shm_src, dest2b, MEM_SIZE, 1);
        munmap(dest2b, MEM_SIZE);
    }

    // Test 2c: Huge Pages warmed 4 threads
    void* dest2c = mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (dest2c == MAP_FAILED) {
        printf("Huge Pages: Not supported (check /proc/meminfo)\n");
    } else {
        for (size_t i = 0; i < MEM_SIZE; i += PAGE_SIZE) ((char*)dest2c)[i] = 0; 
        run_test("Huge Pages (Warm 4 threads)", shm_src, dest2c, MEM_SIZE, 4);
        munmap(dest2c, MEM_SIZE);
    }



    // Test 3: Explicit Touch (Warmup)
    void* dest3 = mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    for (size_t i = 0; i < MEM_SIZE; i += PAGE_SIZE) ((char*)dest3)[i] = 0; 
    run_test("Explicit Touch (Warmed)", shm_src, dest3, MEM_SIZE, 1);
    munmap(dest3, MEM_SIZE);

    // Test 4: MADV_WILLNEED
    void* dest4 = mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(dest4, MEM_SIZE, MADV_WILLNEED);
    run_test("MADV_WILLNEED", shm_src, dest4, MEM_SIZE, 1);
    munmap(dest4, MEM_SIZE);

    // Test 5: MADV_SEQUENTIAL
    void* dest5 = mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    madvise(dest5, MEM_SIZE, MADV_SEQUENTIAL);
    run_test("MADV_SEQUENTIAL", shm_src, dest5, MEM_SIZE, 1);
    munmap(dest5, MEM_SIZE);

    // Test 6: Multi-threaded (2 Threads)
    void* dest6 = mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    run_test("Multi-threaded (2 threads)", shm_src, dest6, MEM_SIZE, 2);
    munmap(dest6, MEM_SIZE);

    // Test 6b: Multi-threaded (2 Threads warmed)
    void* dest6b = mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    for (size_t i = 0; i < MEM_SIZE; i += PAGE_SIZE) ((char*)dest6b)[i] = 0; 
    run_test("Multi-threaded (2 threads warmed)", shm_src, dest6b, MEM_SIZE, 2);
    munmap(dest6b, MEM_SIZE);

    // Test 6c: Multi-threaded (3 Threads warmed)
    void* dest6c = mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    for (size_t i = 0; i < MEM_SIZE; i += PAGE_SIZE) ((char*)dest6c)[i] = 0; 
    run_test("Multi-threaded (3 threads warmed)", shm_src, dest6c, MEM_SIZE, 3);
    munmap(dest6c, MEM_SIZE);

    // Test 7: Multi-threaded (4 Threads)
    void* dest7 = mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    run_test("Multi-threaded (4 threads)", shm_src, dest7, MEM_SIZE, 4);
    munmap(dest7, MEM_SIZE);

    // Test 7: Multi-threaded (4 Threads warmed)
    void* dest7b = mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    for (size_t i = 0; i < MEM_SIZE; i += PAGE_SIZE) ((char*)dest7b)[i] = 0; 
    run_test("Multi-threaded (4 threads warmed)", shm_src, dest7b, MEM_SIZE, 4);
    munmap(dest7b, MEM_SIZE);

    // Test 8: Multi-threaded (8 Threads)
    void* dest8 = mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    run_test("Multi-threaded (8 threads)", shm_src, dest8, MEM_SIZE, 8);
    munmap(dest8, MEM_SIZE);

    // Test 8: Multi-threaded (8 Threads warmed)
    void* dest8b = mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    for (size_t i = 0; i < MEM_SIZE; i += PAGE_SIZE) ((char*)dest8b)[i] = 0; 
    run_test("Multi-threaded (8 threads warmed)", shm_src, dest8b, MEM_SIZE, 8);
    munmap(dest8b, MEM_SIZE);

    // Test 9: Multi-threaded (16 Threads)
    void* dest9 = mmap(NULL, MEM_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    run_test("Multi-threaded (16 threads)", shm_src, dest9, MEM_SIZE, 16);
    munmap(dest9, MEM_SIZE);



    munmap(shm_src, MEM_SIZE);

#ifdef FD_MAPPED
    shm_unlink(SHM_SEG_NAME);
#endif
    return 0;
}
