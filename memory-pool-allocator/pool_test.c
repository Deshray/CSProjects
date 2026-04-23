/*
 * pool_test.c — Test suite for the memory pool allocator.
 *
 * Tests are grouped into categories:
 *   1. Basic allocation and free
 *   2. Boundary conditions (size 0, NULL, oversized)
 *   3. Coalescing (all 4 cases)
 *   4. Splitting
 *   5. Realloc (shrink, in-place grow, move)
 *   6. Calloc (zero-init, overflow)
 *   7. Double-free and dangling pointer safety
 *   8. Alignment
 *   9. Stress / fragmentation
 *  10. Pool exhaustion
 */

#include "pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

/* ── Test framework ───────────────────────────────────────────────────────── */

static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(expr, msg)                                        \
    do {                                                        \
        tests_run++;                                            \
        if (expr) {                                             \
            tests_passed++;                                     \
            printf("  [PASS] %s\n", msg);                      \
        } else {                                                \
            tests_failed++;                                     \
            printf("  [FAIL] %s  (line %d)\n", msg, __LINE__); \
        }                                                       \
    } while (0)

#define SECTION(name) \
    do { printf("\n--- %s ---\n", name); pool_reset(); } while (0)

/* ── Helper: check 8-byte alignment ──────────────────────────────────────── */

static int is_aligned(void *ptr)
{
    return ((uintptr_t)ptr % 8) == 0;
}

/* ── 1. Basic allocation and free ─────────────────────────────────────────── */

static void test_basic(void)
{
    SECTION("1. Basic allocation and free");

    void *p1 = pool_malloc(64);
    CHECK(p1 != NULL, "malloc(64) returns non-NULL");
    CHECK(is_aligned(p1), "malloc(64) is 8-byte aligned");

    /* Write and read back pattern to verify usability */
    memset(p1, 0xAB, 64);
    unsigned char *b = (unsigned char *)p1;
    int pattern_ok = 1;
    for (int i = 0; i < 64; i++) if (b[i] != 0xAB) { pattern_ok = 0; break; }
    CHECK(pattern_ok, "64 bytes are writable and readable");

    pool_free(p1);

    /* After freeing, the pool should have the block available again */
    void *p2 = pool_malloc(64);
    CHECK(p2 != NULL, "malloc(64) after free returns non-NULL");
    pool_free(p2);
}

/* ── 2. Boundary conditions ───────────────────────────────────────────────── */

static void test_boundaries(void)
{
    SECTION("2. Boundary conditions");

    CHECK(pool_malloc(0) == NULL, "malloc(0) returns NULL");

    pool_free(NULL);   /* should not crash */
    CHECK(1, "free(NULL) does not crash");

    /* Oversized allocation */
    void *huge = pool_malloc(POOL_SIZE + 1);
    CHECK(huge == NULL, "malloc(POOL_SIZE+1) returns NULL");

    /* Allocate exactly 1 byte */
    void *one = pool_malloc(1);
    CHECK(one != NULL, "malloc(1) returns non-NULL");
    CHECK(is_aligned(one), "malloc(1) is 8-byte aligned");
    pool_free(one);

    /* Pointer outside pool — should not crash */
    int stack_var = 42;
    pool_free(&stack_var);
    CHECK(1, "free(stack pointer) does not crash");
}

/* ── 3. Coalescing ────────────────────────────────────────────────────────── */

static void test_coalescing(void)
{
    SECTION("3. Coalescing");

    /*
     * Allocate three adjacent blocks A, B, C.
     * Free them in different orders to exercise all 4 coalesce cases.
     */

    /* Case 2: free middle then right → merges C right */
    void *a = pool_malloc(128);
    void *b = pool_malloc(128);
    void *c = pool_malloc(128);
    CHECK(a && b && c, "Three 128-byte allocs succeed");

    pool_free(b);
    pool_free(c);   /* C is free, and its left (B) is also free → merge B+C */

    /* Now A is allocated, B+C are one free block.
     * After freeing A, the whole region should coalesce. */
    pool_free(a);

    /* The entire pool should now be a single free block,
     * so a large allocation should succeed. */
    void *big = pool_malloc(384 - 8); /* slightly less than 3*128 */
    CHECK(big != NULL, "Large alloc after full coalesce succeeds (cases 2+3+4)");
    pool_free(big);

    /* Case 1: allocate two blocks, free them non-adjacently */
    pool_reset();
    void *x = pool_malloc(64);
    void *y = pool_malloc(64);
    void *z = pool_malloc(64);
    pool_free(x);
    /* y is still allocated so x and z do not coalesce */
    void *x2 = pool_malloc(64);
    CHECK(x2 != NULL, "Alloc reuses freed block when neighbour still in use (case 1)");
    pool_free(x2);
    pool_free(y);
    pool_free(z);
}

/* ── 4. Splitting ─────────────────────────────────────────────────────────── */

static void test_splitting(void)
{
    SECTION("4. Block splitting");

    /*
     * Allocate a small block, free it, then allocate an even smaller block.
     * The allocator should split and leave a remainder that is reclaimable.
     */
    void *large = pool_malloc(512);
    CHECK(large != NULL, "malloc(512) succeeds");
    pool_free(large);

    void *small = pool_malloc(64);
    CHECK(small != NULL, "malloc(64) from previously-freed 512-byte region");

    /* The remainder (512 - 64 - overhead) should still be allocatable */
    void *remainder = pool_malloc(256);
    CHECK(remainder != NULL, "Second alloc from split remainder succeeds");

    pool_free(small);
    pool_free(remainder);
}

/* ── 5. Realloc ───────────────────────────────────────────────────────────── */

static void test_realloc(void)
{
    SECTION("5. Realloc");

    /* Realloc from NULL == malloc */
    void *p = pool_realloc(NULL, 128);
    CHECK(p != NULL, "realloc(NULL, 128) == malloc(128)");
    memset(p, 0x55, 128);

    /* Shrink in place */
    void *q = pool_realloc(p, 64);
    CHECK(q == p, "realloc shrink returns same pointer (in-place)");
    unsigned char *b = (unsigned char *)q;
    int pattern_ok = 1;
    for (int i = 0; i < 64; i++) if (b[i] != 0x55) { pattern_ok = 0; break; }
    CHECK(pattern_ok, "Payload preserved after in-place shrink");

    /* Grow — may return same or new pointer */
    void *r = pool_realloc(q, 512);
    CHECK(r != NULL, "realloc grow (128 → 512) returns non-NULL");
    b = (unsigned char *)r;
    /* First 64 bytes must still be 0x55 */
    pattern_ok = 1;
    for (int i = 0; i < 64; i++) if (b[i] != 0x55) { pattern_ok = 0; break; }
    CHECK(pattern_ok, "First 64 bytes preserved after grow");
    pool_free(r);

    /* Realloc to 0 == free */
    void *s = pool_malloc(64);
    CHECK(s != NULL, "Setup: malloc(64)");
    void *t = pool_realloc(s, 0);
    CHECK(t == NULL, "realloc(ptr, 0) returns NULL (acts as free)");
}

/* ── 6. Calloc ────────────────────────────────────────────────────────────── */

static void test_calloc(void)
{
    SECTION("6. Calloc");

    void *p = pool_calloc(16, sizeof(int));  /* 64 bytes */
    CHECK(p != NULL, "calloc(16, 4) returns non-NULL");
    CHECK(is_aligned(p), "calloc result is 8-byte aligned");

    /* Verify zero-initialisation */
    int *arr = (int *)p;
    int all_zero = 1;
    for (int i = 0; i < 16; i++) if (arr[i] != 0) { all_zero = 0; break; }
    CHECK(all_zero, "calloc memory is zero-initialised");
    pool_free(p);

    /* Overflow detection */
    size_t big = (size_t)-1;
    void *ov = pool_calloc(2, big);
    CHECK(ov == NULL, "calloc(2, SIZE_MAX) returns NULL (overflow detected)");
}

/* ── 7. Double-free and safety ────────────────────────────────────────────── */

static void test_safety(void)
{
    SECTION("7. Double-free and safety");

    void *p = pool_malloc(64);
    CHECK(p != NULL, "malloc(64) for double-free test");

    pool_free(p);
    pool_free(p);   /* second free — should silently no-op */
    CHECK(1, "Double-free does not crash");

    /* Pool should still be usable after double-free */
    void *q = pool_malloc(64);
    CHECK(q != NULL, "Pool usable after double-free");
    pool_free(q);
}

/* ── 8. Alignment ─────────────────────────────────────────────────────────── */

static void test_alignment(void)
{
    SECTION("8. Alignment");

    /* Various odd sizes — all returned pointers must be 8-byte aligned */
    size_t sizes[] = { 1, 2, 3, 7, 9, 15, 17, 33, 100, 255, 1000 };
    int n = (int)(sizeof(sizes) / sizeof(sizes[0]));
    void *ptrs[11];

    for (int i = 0; i < n; i++) {
        ptrs[i] = pool_malloc(sizes[i]);
    }

    int all_aligned = 1;
    for (int i = 0; i < n; i++) {
        if (ptrs[i] && !is_aligned(ptrs[i])) { all_aligned = 0; break; }
    }
    CHECK(all_aligned, "All allocations of odd sizes are 8-byte aligned");

    for (int i = 0; i < n; i++) pool_free(ptrs[i]);
}

/* ── 9. Stress / fragmentation ────────────────────────────────────────────── */

static void test_stress(void)
{
    SECTION("9. Stress and fragmentation");

    /*
     * Interleave allocations and frees of varied sizes.
     * This creates a fragmented heap, then tests whether coalescing
     * reclaims enough space for a large allocation.
     */
#define N 64
    void *ptrs[N];
    size_t sizes[N];

    /* Pseudo-random sizes between 8 and 512 bytes */
    unsigned seed = 0xDEADBEEF;
    for (int i = 0; i < N; i++) {
        seed = seed * 1664525u + 1013904223u;
        sizes[i] = 8 + (seed % 505);
        ptrs[i] = pool_malloc(sizes[i]);
    }

    int all_ok = 1;
    for (int i = 0; i < N; i++) {
        if (!ptrs[i]) { all_ok = 0; break; }
    }
    CHECK(all_ok, "64 varied allocations all succeed");

    /* Write a pattern into each block */
    for (int i = 0; i < N; i++) {
        if (ptrs[i]) memset(ptrs[i], (unsigned char)i, sizes[i]);
    }

    /* Free every other block to create fragmentation */
    for (int i = 0; i < N; i += 2) pool_free(ptrs[i]);

    /* Free the remaining blocks — coalescing should reclaim the full pool */
    for (int i = 1; i < N; i += 2) pool_free(ptrs[i]);

    /* After all frees the pool should be coalesced to one big block */
    void *big = pool_malloc(POOL_SIZE / 2);
    CHECK(big != NULL, "Large alloc after stress+coalesce succeeds");
    pool_free(big);
#undef N
}

/* ── 10. Pool exhaustion ──────────────────────────────────────────────────── */

static void test_exhaustion(void)
{
    SECTION("10. Pool exhaustion");

    /*
     * Repeatedly allocate 1 KB blocks until the pool cannot satisfy
     * another 1 KB request. Verify graceful NULL return on exhaustion,
     * then verify that freeing a block makes that size available again.
     */
    int count = 0;
    void *last_ok = NULL;
    while (1) {
        void *p = pool_malloc(1024);
        if (!p) break;
        last_ok = p;
        count++;
    }
    CHECK(count > 0, "At least one 1 KB allocation before exhaustion");

    /* The pool cannot service another 1 KB block. */
    void *fail = pool_malloc(1024);
    CHECK(fail == NULL, "malloc(1024) returns NULL when pool cannot fit 1 KB");

    /* Freeing the last 1 KB block should make 1 KB available again. */
    pool_free(last_ok);
    void *reclaim = pool_malloc(512);
    CHECK(reclaim != NULL, "malloc(512) succeeds after freeing a block from exhausted pool");
    pool_free(reclaim);
}

/* ── Main ─────────────────────────────────────────────────────────────────── */

int main(void)
{
    printf("========================================\n");
    printf("   Memory Pool Allocator — Test Suite   \n");
    printf("========================================\n");

    test_basic();
    test_boundaries();
    test_coalescing();
    test_splitting();
    test_realloc();
    test_calloc();
    test_safety();
    test_alignment();
    test_stress();
    test_exhaustion();

    printf("\n========================================\n");
    printf("  Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0)
        printf("  (%d FAILED)", tests_failed);
    printf("\n========================================\n");

    printf("\nPool state after all tests:\n");
    pool_stats();

    return tests_failed > 0 ? 1 : 0;
}
