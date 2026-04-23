/*
 * pool.c — Memory Pool Allocator Implementation
 *
 * Block layout (implicit free list with boundary tags):
 *
 *   +------------------+--------------------+------------------+
 *   |  block_header_t  |   payload (user)   |  block_footer_t  |
 *   |  size | is_free  |   ...N bytes...    |  size | is_free  |
 *   +------------------+--------------------+------------------+
 *   ^                                                          ^
 *   hdr                                                      footer
 *
 * `size` in both header and footer is the TOTAL block size:
 *   header + payload + footer (rounded to ALIGNMENT).
 *
 * Boundary tags allow O(1) backward coalescing: to find the previous
 * block, read the footer immediately before the current header.
 *
 * Coalescing cases (P = previous, N = next, C = current):
 *   Case 1:  P=alloc, N=alloc  →  no merge
 *   Case 2:  P=alloc, N=free   →  merge C + N
 *   Case 3:  P=free,  N=alloc  →  merge P + C
 *   Case 4:  P=free,  N=free   →  merge P + C + N
 */

#include "pool.h"

#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

/* ── Alignment ────────────────────────────────────────────────────────────── */

#define ALIGNMENT       8
#define ALIGN(n)        (((n) + (ALIGNMENT - 1)) & ~(size_t)(ALIGNMENT - 1))

/* ── Block metadata structures ────────────────────────────────────────────── */

typedef struct {
    size_t size;     /* total block size (header + payload + footer) */
    int    is_free;  /* 1 = free, 0 = allocated                      */
} block_tag_t;

/* Header and footer are the same structure; keep them the same size. */
#define TAG_SIZE        ALIGN(sizeof(block_tag_t))
#define OVERHEAD        (2 * TAG_SIZE)                  /* header + footer    */
#define MIN_PAYLOAD     ALIGNMENT                       /* smallest allocation */
#define MIN_BLOCK       (OVERHEAD + MIN_PAYLOAD)        /* smallest valid block */

/* ── Pool backing buffer ──────────────────────────────────────────────────── */

static char   pool_buf[POOL_SIZE];
static int    pool_initialized = 0;

/* ── Diagnostic counters ──────────────────────────────────────────────────── */

static int    total_allocs = 0;
static int    total_frees  = 0;

/* ── Pointer helpers ──────────────────────────────────────────────────────── */

/* Cast a raw address to a block_tag_t pointer. */
static inline block_tag_t *tag_at(void *addr)
{
    return (block_tag_t *)addr;
}

/* Header of the block containing this header address. */
static inline block_tag_t *hdr(void *header_addr)
{
    return tag_at(header_addr);
}

/* Footer of the block whose header is at `h`. */
static inline block_tag_t *ftr(block_tag_t *h)
{
    return tag_at((char *)h + h->size - TAG_SIZE);
}

/* Header of the next block. */
static inline block_tag_t *next_hdr(block_tag_t *h)
{
    return tag_at((char *)h + h->size);
}

/* Header of the previous block (via its footer immediately before `h`). */
static inline block_tag_t *prev_hdr(block_tag_t *h)
{
    block_tag_t *prev_footer = tag_at((char *)h - TAG_SIZE);
    return tag_at((char *)h - prev_footer->size);
}

/* Payload pointer from a header pointer. */
static inline void *payload(block_tag_t *h)
{
    return (char *)h + TAG_SIZE;
}

/* Header pointer from a payload pointer. */
static inline block_tag_t *header_from_payload(void *ptr)
{
    return tag_at((char *)ptr - TAG_SIZE);
}

/* ── Bounds checking ──────────────────────────────────────────────────────── */

static inline int ptr_in_pool(void *p)
{
    return (char *)p >= pool_buf && (char *)p < pool_buf + POOL_SIZE;
}

/* True if the block at `h` fits entirely within the pool. */
static inline int block_in_pool(block_tag_t *h)
{
    if (!ptr_in_pool(h)) return 0;
    if (h->size < MIN_BLOCK) return 0;
    char *end = (char *)h + h->size;
    return end <= pool_buf + POOL_SIZE;
}

/* True if `h` is the very first block (no previous block exists). */
static inline int is_first_block(block_tag_t *h)
{
    return (char *)h == pool_buf;
}

/* True if next_hdr(h) is at or past the end of the pool. */
static inline int is_last_block(block_tag_t *h)
{
    return (char *)h + h->size >= pool_buf + POOL_SIZE;
}

/* ── Tag write helper ─────────────────────────────────────────────────────── */

/*
 * Write `size` and `is_free` to both the header and footer of the block
 * whose header is at `h`. This keeps boundary tags in sync.
 */
static void set_block(block_tag_t *h, size_t size, int is_free)
{
    h->size    = size;
    h->is_free = is_free;
    block_tag_t *f = ftr(h);
    f->size    = size;
    f->is_free = is_free;
}

/* ── Coalescing ───────────────────────────────────────────────────────────── */

/*
 * coalesce — merge `h` with adjacent free blocks.
 *
 * The block at `h` must already be marked free before calling this.
 * Returns a pointer to the (possibly enlarged) merged block header.
 */
static block_tag_t *coalesce(block_tag_t *h)
{
    int prev_free = (!is_first_block(h)) && prev_hdr(h)->is_free;
    int next_free = (!is_last_block(h))  && next_hdr(h)->is_free;

    if (!prev_free && !next_free) {
        /* Case 1: both neighbours allocated — nothing to merge */
        return h;
    }

    if (!prev_free && next_free) {
        /* Case 2: merge current + next */
        block_tag_t *n = next_hdr(h);
        set_block(h, h->size + n->size, 1);
        return h;
    }

    if (prev_free && !next_free) {
        /* Case 3: merge prev + current */
        block_tag_t *p = prev_hdr(h);
        set_block(p, p->size + h->size, 1);
        return p;
    }

    /* Case 4: merge prev + current + next */
    block_tag_t *p = prev_hdr(h);
    block_tag_t *n = next_hdr(h);
    set_block(p, p->size + h->size + n->size, 1);
    return p;
}

/* ── Public API ───────────────────────────────────────────────────────────── */

void pool_init(void)
{
    if (pool_initialized) return;

    /* Initialise the entire pool as a single free block. */
    block_tag_t *h = tag_at(pool_buf);
    set_block(h, POOL_SIZE, 1);

    total_allocs   = 0;
    total_frees    = 0;
    pool_initialized = 1;
}

void pool_reset(void)
{
    pool_initialized = 0;
    pool_init();
}

void *pool_malloc(size_t size)
{
    if (!pool_initialized) pool_init();
    if (size == 0) return NULL;

    /* Round up payload and compute required block size. */
    size_t payload_size = ALIGN(size);
    size_t needed       = payload_size + OVERHEAD;
    if (needed < MIN_BLOCK) needed = MIN_BLOCK;

    /* First-fit traversal across all blocks. */
    block_tag_t *h = tag_at(pool_buf);
    while (block_in_pool(h)) {
        if (h->is_free && h->size >= needed) {

            size_t leftover = h->size - needed;

            if (leftover >= MIN_BLOCK) {
                /*
                 * Split: carve `needed` bytes off the front,
                 * leave the remainder as a free block.
                 */
                set_block(h, needed, 0);
                block_tag_t *split = next_hdr(h);
                set_block(split, leftover, 1);
            } else {
                /* Use the whole block (absorb internal fragmentation). */
                set_block(h, h->size, 0);
            }

            total_allocs++;
            return payload(h);
        }
        h = next_hdr(h);
    }

    return NULL;   /* pool exhausted */
}

void pool_free(void *ptr)
{
    if (!ptr) return;

    /* Ignore pointers outside the pool. */
    if (!ptr_in_pool(ptr)) return;

    block_tag_t *h = header_from_payload(ptr);

    /* Detect double-free: if already free, bail out silently. */
    if (h->is_free) return;

    /* Validate size sanity. */
    if (!block_in_pool(h)) return;

    set_block(h, h->size, 1);
    coalesce(h);
    total_frees++;
}

void *pool_realloc(void *ptr, size_t new_size)
{
    if (!ptr)       return pool_malloc(new_size);
    if (new_size == 0) { pool_free(ptr); return NULL; }
    if (!ptr_in_pool(ptr)) return NULL;

    block_tag_t *h = header_from_payload(ptr);
    if (!block_in_pool(h) || h->is_free) return NULL;

    size_t needed = ALIGN(new_size) + OVERHEAD;
    if (needed < MIN_BLOCK) needed = MIN_BLOCK;

    /* ── In-place shrink ───────────────────────────────────────────────── */
    if (needed <= h->size) {
        size_t leftover = h->size - needed;
        if (leftover >= MIN_BLOCK) {
            set_block(h, needed, 0);
            block_tag_t *split = next_hdr(h);
            set_block(split, leftover, 1);
            coalesce(split);
        }
        /* else: absorb the small leftover into this block */
        return ptr;
    }

    /* ── In-place expand into next free block ──────────────────────────── */
    if (!is_last_block(h)) {
        block_tag_t *n = next_hdr(h);
        if (n->is_free && (h->size + n->size) >= needed) {
            size_t combined  = h->size + n->size;
            size_t leftover  = combined - needed;
            if (leftover >= MIN_BLOCK) {
                set_block(h, needed, 0);
                block_tag_t *split = next_hdr(h);
                set_block(split, leftover, 1);
            } else {
                set_block(h, combined, 0);
            }
            return ptr;
        }
    }

    /* ── Fallback: alloc new block, copy payload, free old ─────────────── */
    void *new_ptr = pool_malloc(new_size);
    if (!new_ptr) return NULL;

    size_t old_payload = h->size - OVERHEAD;
    size_t copy_bytes  = old_payload < new_size ? old_payload : new_size;
    memcpy(new_ptr, ptr, copy_bytes);
    pool_free(ptr);

    return new_ptr;
}

void *pool_calloc(size_t nmemb, size_t size)
{
    /* Check for multiplication overflow. */
    if (nmemb != 0 && size > (size_t)-1 / nmemb) return NULL;

    size_t total = nmemb * size;
    void  *ptr   = pool_malloc(total);
    if (ptr) memset(ptr, 0, total);
    return ptr;
}

void pool_stats(void)
{
    if (!pool_initialized) pool_init();

    size_t free_payload  = 0;
    size_t used_payload  = 0;
    int    free_blocks   = 0;
    int    used_blocks   = 0;
    size_t largest_free  = 0;

    block_tag_t *h = tag_at(pool_buf);
    while (block_in_pool(h)) {
        size_t p = h->size - OVERHEAD;
        if (h->is_free) {
            free_payload += p;
            free_blocks++;
            if (h->size > largest_free) largest_free = h->size;
        } else {
            used_payload += p;
            used_blocks++;
        }
        h = next_hdr(h);
    }

    size_t largest_free_payload = (largest_free >= OVERHEAD)
                                  ? largest_free - OVERHEAD : 0;

    double frag = 0.0;
    if (free_payload > 0) {
        frag = 100.0 * (1.0 - (double)largest_free_payload /
                               (double)free_payload);
    }

    printf("=== Pool Statistics ===========================\n");
    printf("  Pool capacity  : %d bytes\n",    POOL_SIZE);
    printf("  Tag overhead   : %zu bytes\n",   TAG_SIZE);
    printf("  Min block size : %zu bytes\n",   MIN_BLOCK);
    printf("-----------------------------------------------\n");
    printf("  Used blocks    : %d  (%zu bytes payload)\n",
           used_blocks, used_payload);
    printf("  Free blocks    : %d  (%zu bytes payload)\n",
           free_blocks, free_payload);
    printf("  Largest free   : %zu bytes payload\n", largest_free_payload);
    printf("  Fragmentation  : %.1f%%\n", frag);
    printf("-----------------------------------------------\n");
    printf("  Total allocs   : %d\n", total_allocs);
    printf("  Total frees    : %d\n", total_frees);
    printf("  Live allocs    : %d\n", total_allocs - total_frees);
    printf("===============================================\n");
}
