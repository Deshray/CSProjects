/*
 * pool.h — Memory Pool Allocator
 *
 * A fixed-size memory pool allocator implementing malloc/free/realloc/calloc
 * from scratch using boundary tags and first-fit placement with coalescing.
 *
 * Design:
 *   - 1 MB static backing buffer (no OS calls after init)
 *   - Implicit free list traversal (O(n) alloc, O(1) free with coalescing)
 *   - Boundary tags: every block has a matching header + footer
 *   - 4-case coalescing: handles prev-free, next-free, both, neither
 *   - Block splitting on allocation when remainder >= MIN_BLOCK_SIZE
 *   - In-place realloc when possible (expand into adjacent free block)
 *   - 8-byte alignment on all allocations
 */

#ifndef POOL_H
#define POOL_H

#include <stddef.h>

/* Pool size in bytes — can be overridden at compile time */
#ifndef POOL_SIZE
#define POOL_SIZE (1 * 1024 * 1024)   /* 1 MB */
#endif

/*
 * pool_init — explicitly initialise the pool.
 * Called automatically on first use; safe to call multiple times.
 */
void pool_init(void);

/*
 * pool_malloc — allocate at least `size` bytes.
 * Returns a pointer to aligned memory, or NULL if the pool is exhausted.
 * Behaviour on size == 0 is undefined (returns NULL).
 */
void *pool_malloc(size_t size);

/*
 * pool_free — return a previously allocated block to the pool.
 * Coalesces adjacent free blocks immediately.
 * No-op on NULL. Ignores pointers not belonging to the pool.
 * Detects and ignores double-free (marks are checked before freeing).
 */
void pool_free(void *ptr);

/*
 * pool_realloc — resize a previously allocated block.
 * Attempts in-place expansion into the adjacent block when possible.
 * Falls back to alloc + copy + free when in-place is not possible.
 * Equivalent to pool_malloc(new_size) when ptr == NULL.
 * Equivalent to pool_free(ptr) when new_size == 0.
 */
void *pool_realloc(void *ptr, size_t new_size);

/*
 * pool_calloc — allocate nmemb * size bytes, zero-initialised.
 * Returns NULL on overflow or pool exhaustion.
 */
void *pool_calloc(size_t nmemb, size_t size);

/*
 * pool_stats — print diagnostic information about the pool state.
 * Reports: used/free block counts, fragmentation ratio, largest free block.
 */
void pool_stats(void);

/*
 * pool_reset — wipe the pool back to a single empty block.
 * Useful for testing; not safe during normal use.
 */
void pool_reset(void);

#endif /* POOL_H */
