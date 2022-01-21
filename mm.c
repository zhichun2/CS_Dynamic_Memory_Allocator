/**
 * @file mm.c
 * @brief A 64-bit struct-based implicit free list memory allocator
 *
 * 15-213: Introduction to Computer Systems
 *
 *
 *************************************************************************
 *
 * FILE OVERVIEW
 * This file contains a dynamic memory allocator. A structure called block
 * is used to store the memory. Depending on the size and the allocation
 * status, the block will contain different fields.
 *
 * Allocated blocks:
 * If the memory content of a block is greater than 8 bytes, an allocated
 * block will have a minimum block size of 32 bytes: 8 bytes for header,
 * a minimum of 24 bytes for payload.
 * If the memory content of a block is less than 8 bytes, an allocated
 * mini block will have a uniformed block size of 16 bytes: 8 bytes for
 * header and 8 bytes for payload.
 *
 * Free blocks:
 * If the memory content of a block is greater than 8 bytes, a free block
 * will have a minimum block size of 32 bytes: 8 bytes for header, 8 bytes
 * for a pointer to the previous free block, 8 bytes for a pointer to the
 * next block, and 8 bytes for footer.
 * If the memory content of a block is less than 8 bytes, a free block will
 * have a uniformed block size of 16 bytes: 8 bytes for header and 8 bytes
 * for a pointer to the next free block.
 *
 * Organization of free list:
 * Segmentation list is used to implementation of free list. There are 15
 * buckets and each of the bucket contain blocks of different sizes. The
 * buckets are divided with threshold of powers of 2. The first bucket
 * is used to store mini blocks of size 16 bytes, and it is implemented using
 * singly linked list. The rest of the buckets are used to store regular
 * blocks, and it is implemented using doubly linked list.
 *
 * Allocater Manipulation:
 * When allocating a block, the allocater takes free blocks of suitable size
 * from the free list. When freeing a block, it puts the block back to the
 * its corresponding bucket.
 *
 *************************************************************************
 *
 * @author Zhichun Zhao <zhichun2@andrew.cmu.edu>
 */

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "memlib.h"
#include "mm.h"

/* Do not change the following! */

#ifdef DRIVER
/* create aliases for driver tests */
#define malloc mm_malloc
#define free mm_free
#define realloc mm_realloc
#define calloc mm_calloc
#define memset mem_memset
#define memcpy mem_memcpy
#endif /* def DRIVER */

/* You can change anything from here onward */

/*
 *****************************************************************************
 * If DEBUG is defined (such as when running mdriver-dbg), these macros      *
 * are enabled. You can use them to print debugging output and to check      *
 * contracts only in debug mode.                                             *
 *                                                                           *
 * Only debugging macros with names beginning "dbg_" are allowed.            *
 * You may not define any other macros having arguments.                     *
 *****************************************************************************
 */
#ifdef DEBUG
/* When DEBUG is defined, these form aliases to useful functions */
#define dbg_printf(...) printf(__VA_ARGS__)
#define dbg_requires(expr) assert(expr)
#define dbg_assert(expr) assert(expr)
#define dbg_ensures(expr) assert(expr)
#define dbg_printheap(...) print_heap(__VA_ARGS__)
#else
/* When DEBUG is not defined, no code gets generated for these */
/* The sizeof() hack is used to avoid "unused variable" warnings */
#define dbg_printf(...) (sizeof(__VA_ARGS__), -1)
#define dbg_requires(expr) (sizeof(expr), 1)
#define dbg_assert(expr) (sizeof(expr), 1)
#define dbg_ensures(expr) (sizeof(expr), 1)
#define dbg_printheap(...) ((void)sizeof(__VA_ARGS__))
#endif

/* Basic constants */

typedef uint64_t word_t;

/** @brief Word and header size (bytes) */
static const size_t wsize = sizeof(word_t);

/** @brief Double word size (bytes) */
static const size_t dsize = 2 * wsize;

/** @brief Minimum block size (bytes) */
static const size_t min_block_size = dsize;

/** @brief Minimum size for extending heap (bytes) */
static const size_t chunksize = (1 << 12);

/** @brief Bit mask for extracting alloc bit */
static const word_t alloc_mask = 0x1;

/** @brief Bit mask for extracting alloc bit for prev block */
static const word_t last_alloc_mask = 0x2;

/** @brief Bit mask for extracting mini bit */
static const word_t mini_mask = 0x4;

/** @brief Bit mask for extracting block size */
static const word_t size_mask = ~(word_t)0xF;

/** @brief Represents the header and payload of one block in the heap */
typedef struct block_t {
    /** @brief Header contains size + allocation flag */
    word_t header;
    union {
        struct {
            struct block_t *next;
            struct block_t *prev;
        };
        /** @brief A pointer to the block payload */
        char payload[0];
    };
} block_t;

/* Global variables */

/** @brief Pointer to first block in the heap */
static block_t *heap_start = NULL;

/** @brief Seglist bucket on the stack */
block_t *seglist[15];

/*
 *****************************************************************************
 * The functions below are short wrapper functions to perform                *
 * bit manipulation, pointer arithmetic, and other helper operations.        *
 *                                                                           *
 * We've given you the function header comments for the functions below      *
 * to help you understand how this baseline code works.                      *
 *                                                                           *
 * Note that these function header comments are short since the functions    *
 * they are describing are short as well; you will need to provide           *
 * adequate details for the functions that you write yourself!               *
 *****************************************************************************
 */

/*
 * ---------------------------------------------------------------------------
 *                        BEGIN SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/**
 * @brief Returns the maximum of two integers.
 * @param[in] x
 * @param[in] y
 * @return `x` if `x > y`, and `y` otherwise.
 */
static size_t max(size_t x, size_t y) {
    return (x > y) ? x : y;
}

/**
 * @brief Rounds `size` up to next multiple of n
 * @param[in] size
 * @param[in] n
 * @return The size after rounding up
 */
static size_t round_up(size_t size, size_t n) {
    return n * ((size + (n - 1)) / n);
}

/**
 * @brief Packs the `size` and `alloc` of a block into a word suitable for
 *        use as a packed value.
 *
 * Packed values are used for both headers and footers.
 *
 * The allocation status is packed into the lowest bit of the word.
 *
 * @param[in] size The size of the block being represented
 * @param[in] alloc True if the block is allocated
 * @return The packed value
 */
static word_t pack(size_t size, bool alloc, bool last, bool mini) {
    word_t word = size;
    if (alloc) {
        word |= alloc_mask;
    }
    if (last) {
        word |= last_alloc_mask;
    }
    if (mini) {
        word |= mini_mask;
    }
    return word;
}

/**
 * @brief Extracts the size represented in a packed word.
 *
 * This function simply clears the lowest 4 bits of the word, as the heap
 * is 16-byte aligned.
 *
 * @param[in] word
 * @return The size of the block represented by the word
 */
static size_t extract_size(word_t word) {
    return (word & size_mask);
}

/**
 * @brief Extracts the size of a block from its header.
 * @param[in] block
 * @return The size of the block
 */
static size_t get_size(block_t *block) {
    return extract_size(block->header);
}

/**
 * @brief Given a payload pointer, returns a pointer to the corresponding
 *        block.
 * @param[in] bp A pointer to a block's payload
 * @return The corresponding block
 */
static block_t *payload_to_header(void *bp) {
    return (block_t *)((char *)bp - offsetof(block_t, payload));
}

/**
 * @brief Given a block pointer, returns a pointer to the corresponding
 *        payload.
 * @param[in] block
 * @return A pointer to the block's payload
 * @pre The block must be a valid block, not a boundary tag.
 */
static void *header_to_payload(block_t *block) {
    dbg_requires(get_size(block) != 0);
    return (void *)(block->payload);
}

/**
 * @brief Given a block pointer, returns a pointer to the corresponding
 *        footer.
 * @param[in] block
 * @return A pointer to the block's footer
 * @pre The block must be a valid block, not a boundary tag.
 */
static word_t *header_to_footer(block_t *block) {
    dbg_requires(get_size(block) != 0 &&
                 "Called header_to_footer on the epilogue block");
    return (word_t *)(block->payload + get_size(block) - dsize);
}

/**
 * @brief Given a block footer, returns a pointer to the corresponding
 *        header.
 * @param[in] footer A pointer to the block's footer
 * @return A pointer to the start of the block
 * @pre The footer must be the footer of a valid block, not a boundary tag.
 */
static block_t *footer_to_header(word_t *footer) {
    size_t size = extract_size(*footer);
    dbg_assert(size != 0 && "Called footer_to_header on the prologue block");
    return (block_t *)((char *)footer + wsize - size);
}

/**
 * @brief Returns the allocation status of a given header value.
 *
 * This is based on the lowest bit of the header value.
 *
 * @param[in] word
 * @return The allocation status correpsonding to the word
 */
static bool extract_alloc(word_t word) {
    return (bool)(word & alloc_mask);
}

/**
 * @brief Returns the allocation status of a block, based on its header.
 * @param[in] block
 * @return The allocation status of the block
 */
static bool get_alloc(block_t *block) {
    return extract_alloc(block->header);
}

/**
 * @brief Returns the payload size of a given block.
 *
 * The payload size is equal to the entire block size minus the sizes of the
 * block's header and footer.
 *
 * @param[in] block
 * @return The size of the block's payload
 */
static size_t get_payload_size(block_t *block) {
    size_t asize = get_size(block);
    return asize - wsize;
}

/**
 * @brief Returns the allocation status of the previous block
 * of a given header value.
 *
 * This is based on the second lowest bit of the header value.
 *
 * @param[in] word
 * @return The allocation status of the previous block correpsonding to the word
 */
static bool extract_last_alloc(word_t word) {
    return (bool)((word & last_alloc_mask) >> 1);
}

/**
 * @brief Returns the allocation status the previous block of a block,
 * based on its header.
 * @param[in] block
 * @return The allocation status of the previous block of the block
 */
static bool get_last_alloc(block_t *block) {
    return extract_last_alloc(block->header);
}

/**
 * @brief Returns the mini status of the previous block
 * of a given header value.
 *
 * This is based on the second lowest bit of the header value.
 *
 * @param[in] word
 * @return The mini status of the previous block correpsonding to the word
 */
static bool extract_last_mini(word_t word) {
    return (bool)((word & mini_mask) >> 2);
}

/**
 * @brief Returns the mini status the previous block of a block,
 * based on its header.
 * @param[in] block
 * @return The mini status of the previous block of the block
 */
static bool get_last_mini(block_t *block) {
    return extract_last_mini(block->header);
}

/**
 * @brief writes the header and footer of a block
 *
 * This function is used to update the last allocation bit and last mini bit
 * in the header and footer in a given block, corresponding to the last block.
 *
 * @param[in] block
 * @param[in] allocation status of the last block
 * @param[in] mini status of the last block
 */
static void write_hf(block_t *block, bool last, bool mini) {
    size_t size = get_size(block);
    bool alloc = get_alloc(block);
    block->header = pack(size, alloc, last, mini);
    if (alloc == false && size > 16) {
        word_t *footerp = header_to_footer(block);
        *footerp = pack(size, alloc, last, mini);
    }
}

/**
 * @brief Writes an epilogue header at the given address.
 *
 * The epilogue header has size 0, and is marked as allocated.
 *
 * @param[out] block The location to write the epilogue header
 */
static void write_epilogue(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires((char *)block == mem_heap_hi() - 7);
    block->header = pack(0, true, false, false);
}

/**
 * @brief Finds the next consecutive block on the heap.
 *
 * This function accesses the next block in the "implicit list" of the heap
 * by adding the size of the block.
 *
 * @param[in] block A block in the heap
 * @return The next consecutive block on the heap
 * @pre The block is not the epilogue
 */
static block_t *find_next(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires(get_size(block) != 0 &&
                 "Called find_next on the last block in the heap");
    return (block_t *)((char *)block + get_size(block));
}

/**
 * @brief Writes a block starting at the given address.
 *
 * This function writes both a header and footer, where the location of the
 * footer is computed in relation to the header. The header and footer will
 * contain size of block, last mini bit, last alloc bit, and alloc bit.
 *
 * @param[out] block The location to begin writing the block header
 * @param[in] size The size of the new block
 * @param[in] alloc The allocation status of the new block
 * @pre block to write must not be NULL
 * @pre block size must be greater than one
 */
static void write_block(block_t *block, size_t size, bool alloc, bool last,
                        bool mini) {
    dbg_requires(block != NULL);
    dbg_requires(size > 0);
    bool cur_mini;
    if (size == 16) {
        cur_mini = true;
    } else {
        cur_mini = false;
    }
    block->header = pack(size, alloc, last, mini);
    if (!alloc && cur_mini == false) {
        word_t *footerp = header_to_footer(block);
        *footerp = pack(size, alloc, last, mini);
    }
    block_t *next = find_next(block);
    write_hf(next, alloc, cur_mini);
}

/**
 * @brief Finds the footer of the previous block on the heap.
 * @param[in] block A block in the heap
 * @return The location of the previous block's footer
 */
static word_t *find_prev_footer(block_t *block) {
    return &(block->header) - 1;
}

/**
 * @brief Finds the previous consecutive block on the heap.
 *
 * This is the previous block in the "implicit list" of the heap.
 *
 * If the function is called on the first block in the heap, NULL will be
 * returned, since the first block in the heap has no previous block!
 *
 * The position of the previous block is found by reading the previous
 * block's footer to determine its size, then calculating the start of the
 * previous block based on its size.
 *
 * @param[in] block A block in the heap
 * @return The previous consecutive block in the heap.
 */
static block_t *find_prev(block_t *block) {
    dbg_requires(block != NULL);
    if (get_last_mini(block) == true) {
        return (block_t *)((char *)block - 16);
    }
    word_t *footerp = find_prev_footer(block);

    // Return NULL if called on first block in the heap
    if (extract_size(*footerp) == 0) {
        return NULL;
    }

    return footer_to_header(footerp);
}

/*
 * ---------------------------------------------------------------------------
 *                        END SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/******** The remaining content below are helper and debug routines ********/

/** @brief print the blocks in the seglists */
void print_free() {
    int i = 0;
    while (i < 15) {
        dbg_printf("~~~start of seglist[%d]~~~\n", i);
        block_t *temp = seglist[i];
        bool flag = true;
        while (temp != NULL && flag) {

            dbg_printf("Size : %lu, Allocated : %d, Mini : %d \n",
                       get_size(temp), get_alloc(temp), get_last_mini(temp));
            dbg_printf("block address: %lu\n", (size_t)temp);
            dbg_printf("block prev: %lu\n", (size_t)temp->prev);
            dbg_printf("block next: %lu\n", (size_t)temp->next);
            temp = temp->next;
            if (temp != NULL && temp == temp->next) {
                dbg_printf("cycle: temp == temp->next\n");
                flag = false;
            }
            dbg_printf("________________________\n");
        }
        dbg_printf("~~~end of seglist[%d]~~~\n", i);
        i++;
    }
}

/** @brief prints the blocks on heap */
void print_heap() {
    block_t *temp = heap_start;
    while (temp != payload_to_header(mem_sbrk(0))) {
        block_t *temp_prev = find_prev(temp);
        block_t *temp_next = find_next(temp);
        dbg_printf("Size : %lu, Allocated : %d, Mini : %d \n", get_size(temp),
                   get_alloc(temp), get_last_mini(temp));
        dbg_printf("block address: %lu\n", (size_t)temp);
        dbg_printf("block prev: %lu\n", (size_t)temp_prev);
        dbg_printf("block next: %lu\n", (size_t)temp_next);
        temp = temp_next;
    }
    dbg_printf("=================================\n");
}

/**
 * @brief find the index to the correct bucket in seglist
 * @param[in] block size
 * @return index to the correct bucket
 * @pre size must be greater than 0
 */
int find_class(size_t size) {
    dbg_requires(size > 0);
    if (size <= 16) {
        return 0;
    } else if (size > 16 && size <= 32) {
        return 1;
    } else if (size > 32 && size <= 64) {
        return 2;
    } else if (size > 64 && size <= 128) {
        return 3;
    } else if (size > 128 && size <= 256) {
        return 4;
    } else if (size > 256 && size <= 512) {
        return 5;
    } else if (size > 512 && size <= 1024) {
        return 6;
    } else if (size > 1024 && size <= 2048) {
        return 7;
    } else if (size > 2048 && size <= 4096) {
        return 8;
    } else if (size > 4096 && size <= 8192) {
        return 9;
    } else if (size > 8192 && size <= 16384) {
        return 10;
    } else if (size > 16384 && size <= 32768) {
        return 11;
    } else if (size > 32768 && size <= 65536) {
        return 12;
    } else if (size > 65536 && size <= 131072) {
        return 13;
    } else {
        return 14;
    }
}

/**
 * @brief check if a block is in the seglist
 * @param[in] pointer to the start of a seglist
 * @param[in] block
 * @return if a block is in the seglist
 */
bool is_in(block_t *list_start, block_t *block) {
    dbg_requires(block != NULL);
    if (list_start == NULL) {
        return false;
    }
    block_t *temp = list_start;
    while (temp != NULL) {
        if (temp == block) {
            return true;
        }
        temp = temp->next;
    }
    return false;
}

/**
 * @brief insert a free block to the seglist
 * @param[in] block
 * @pre block must not be NULL
 */
void insert(block_t *block) {
    dbg_requires(block != NULL);
    int i = find_class(get_size(block));
    dbg_requires(!is_in(seglist[i], block));
    if (get_size(block) > 16) {
        if (seglist[i] == NULL) {
            seglist[i] = block;
            block->prev = NULL;
            block->next = NULL;
        } else {
            seglist[i]->prev = block;
            block->next = seglist[i];
            block->prev = NULL;
            seglist[i] = block;
        }
    } else {
        if (seglist[i] == NULL) {
            seglist[i] = block;
            block->next = NULL;
        } else {
            block->next = seglist[i];
            seglist[i] = block;
        }
    }
}

/**
 * @brief delete a block from a seglist
 * @param[in] block
 * @pre block must not be NULL
 */
void delete (block_t *block) {
    dbg_requires(block != NULL);
    int i = find_class(get_size(block));
    dbg_requires(seglist[i] != NULL);
    dbg_requires(is_in(seglist[i], block));
    // delete first block
    if (get_size(block) > 16) {
        if (block == seglist[i]) {
            // only block
            if (block->next == NULL) {
                seglist[i] = NULL;
            } else {
                seglist[i] = block->next;
                block->next->prev = NULL;
                block->prev = NULL;
                block->next = NULL;
            }
        }
        // delete last block
        else if (block->next == NULL) {
            block->prev->next = NULL;
            block->prev = NULL;
        }
        // delete middle block
        else {
            block->prev->next = block->next;
            block->next->prev = block->prev;
            block->next = NULL;
            block->prev = NULL;
        }
    }
    // deleting mini blocks
    else {
        block_t *temp = seglist[i];
        if (temp == block) {
            // only block
            if (temp->next == NULL) {
                block->next = NULL;
                seglist[i] = NULL;
            }
            // first block
            else {
                seglist[i] = block->next;
                block->next = NULL;
            }
            return;
        }
        while (temp->next != block) {
            temp = temp->next;
        }
        temp->next = block->next;
        block->next = NULL;
    }
}

/**
 * @brief coalesce a free block with adjacent free blocks on the heap
 *
 * @param[in] block
 * @return coalesced block
 * @pre block size must be greater than 0
 */
static block_t *coalesce_block(block_t *block) {
    dbg_requires(get_size(block) != 0);
    block_t *prev_block;
    block_t *next_block = find_next(block);
    size_t block_size = get_size(block);
    size_t prev_size = 0;
    size_t next_size = get_size(next_block);
    bool last_alloc = get_last_alloc(block);
    bool prev_last_alloc;
    bool last_mini = get_last_mini(block);
    bool prev_last_mini;
    if (last_alloc == false) {
        prev_block = find_prev(block);
        prev_size = get_size(prev_block);
    }
    // both prev and next are free (case 4)
    // un-tie the prev and next blocks, return the new block
    if (last_alloc == 0 && get_alloc(next_block) == 0) {
        prev_last_alloc = get_last_alloc(prev_block);
        prev_last_mini = get_last_mini(prev_block);
        delete (prev_block);
        delete (next_block);
        size_t size = prev_size + block_size + next_size;
        write_block(prev_block, size, false, prev_last_alloc, prev_last_mini);
        return prev_block;
    }
    // only next is free (case 2)
    else if (last_alloc == 1 && get_alloc(next_block) == 0) {
        delete (next_block);
        size_t size = block_size + next_size;
        write_block(block, size, false, last_alloc, last_mini);
        return block;
    }
    // only prev is free (case 3)
    else if (last_alloc == 0 && get_alloc(next_block) == 1) {
        prev_last_alloc = get_last_alloc(prev_block);
        prev_last_mini = get_last_mini(prev_block);
        delete (prev_block);
        size_t size = block_size + prev_size;
        write_block(prev_block, size, false, prev_last_alloc, prev_last_mini);
        return prev_block;
    }
    // both not free (case 1)
    else {
        write_block(block, block_size, false, last_alloc, last_mini);
        return block;
    }
}

/**
 * @brief request more space on heap
 *
 * create a free block of the size requested or chunksize,
 * coalesce it with the last block on heap if that block is free.
 *
 * @param[in] size
 * @return a free block
 */
static block_t *extend_heap(size_t size) {
    void *bp;
    bool last = get_last_alloc(payload_to_header(mem_sbrk(0)));
    bool mini = get_last_mini(payload_to_header(mem_sbrk(0)));
    // Allocate an even number of words to maintain alignment
    size = round_up(size, dsize);
    if ((bp = mem_sbrk(size)) == (void *)-1) {
        return NULL;
    }

    // find address to the new space and epilogue
    block_t *block = payload_to_header(bp);
    block_t *block_next = (block_t *)((char *)block + size);

    write_epilogue(block_next);
    write_block(block, size, false, last, mini);

    // Coalesce in case the previous block was free
    block = coalesce_block(block);

    // add the new block to free
    insert(block);

    return block;
}

/**
 * @brief split a large block into an allocated block and a free block
 *
 * check if the excess space in a block is larger than min_block_size
 * if it is, split the block into two: the first half is of the size
 * requested, and the second block stores the extra space. A pointer
 * to the second block is returned
 *
 * @param[in] block
 * @param[in] asize
 * @return pointer to a free block or NULL
 * @pre block must be allocated
 * @pre block must be larger than min_block_size
 */
static block_t *split_block(block_t *block, size_t asize) {
    dbg_requires(get_alloc(block));
    dbg_requires(asize >= min_block_size);

    size_t block_size = get_size(block);
    bool last = get_last_alloc(block);
    bool last_mini = get_last_mini(block);
    bool cur_mini = false;
    if (asize == 16) {
        cur_mini = true;
    }

    // split the block into two if it is large enough
    if ((block_size - asize) >= min_block_size) {
        block_t *block_next = (block_t *)((char *)block + asize);
        write_block(block_next, block_size - asize, false, true, cur_mini);
        write_block(block, asize, true, last, last_mini);

        block_next = find_next(block);

        return block_next;
    }
    dbg_ensures(get_alloc(block));

    // if the block is too small to be split into two, return NULL
    return NULL;
}

/**
 * @brief
 * Find a block that is large enough to contain a given size
 * and small enough to reduce fragmentation. Search will find
 * the best fit within a controlled number of loops
 * @param[in] asize
 * @return block
 */
static block_t *find_fit(size_t asize) {
    int i = find_class(asize);
    if (i == 0 && seglist[i] != NULL) {
        return seglist[i];
    }
    while (i < 15) {
        block_t *block = seglist[i];
        block_t *best = NULL;
        int limit = 3;
        // control the loop times
        while (block != NULL && limit > 0) {
            if ((asize <= get_size(block))) {
                if (best == NULL || get_size(best) > get_size(block)) {
                    best = block;
                }
                limit--;
            }
            block = block->next;
        }
        if (best == NULL) {
            i++;
        } else {
            return best;
        }
    }
    // no fit found
    return NULL;
}

/**
 * @brief check if the seglist is valid
 *
 * looping though the buckets of the seglists by blocks and check for
 * the boundry, doubly linked list / node number consistency
 *
 * @param[in] line number
 * @return if the seglist is valid
 */
bool checkFree(int line) {
    block_t *cur = seglist[0];
    int i = 1;

    // checking seglist for mini blocks
    while (cur != NULL) {
        if ((size_t)cur > ((size_t)mem_heap_hi()) ||
            (size_t)cur < ((size_t)mem_heap_lo())) {
            dbg_printf("boundry failed\n");
            return false;
        }
        cur = cur->next;
    }

    // checking seglist for regular blocks
    while (i < 15) {
        block_t *temp = seglist[i];
        block_t *end = NULL;
        size_t count = 0;
        if (seglist[i] == NULL) {
        }
        while (temp != NULL) {
            block_t *next = temp->next;
            // consistency
            if (next != NULL && temp != next->prev) {
                dbg_printf("consistency failed\n");
                return false;
            }
            // boundry
            if ((size_t)temp > ((size_t)mem_heap_hi()) ||
                (size_t)temp < ((size_t)mem_heap_lo())) {
                dbg_printf("boundry failed\n");
                return false;
            }
            if (next == NULL) {
                end = temp;
            }
            count++;
            temp = temp->next;
        }
        // looping backwards to check if # of blocks match
        while (end != NULL) {
            count--;
            end = end->prev;
        }
        if (count != 0) {
            dbg_printf("# of nodes failed\n");
            return false;
        }
        i++;
    }
    return true;
}

/**
 * @brief check if a heap is valid
 *
 * looping though the blocks on a heap and check for prologue/epilogue,
 * boundry, allignment, last alloc/mini bit/header/footer consistency, size,
 * coalesce and seglist
 *
 * @param[in] line
 * @return if a heap is valid
 */
bool mm_checkheap(int line) {
    block_t *prologue = (block_t *)((word_t *)heap_start - 1);
    block_t *epilogue = payload_to_header(mem_sbrk(0));
    block_t *temp = heap_start;

    // check prologue
    if ((size_t)prologue < (size_t)mem_heap_lo() || get_size(prologue) != 0 ||
        get_alloc(prologue) == false) {
        dbg_printf("prologue returns false\n");
        return false;
    }

    // check epilogue
    if ((size_t)epilogue < (size_t)mem_heap_hi() - 7 ||
        get_size(epilogue) != 0 || get_alloc(epilogue) == false) {
        dbg_printf("epilogue returns false\n");
        return false;
    }

    // looping through the heap block by block
    while (get_size(temp) != 0) {
        word_t *temp_payload = header_to_payload(temp);
        word_t *temp_footer = header_to_footer(temp);
        block_t *temp_prev;
        block_t *temp_next = find_next(temp);
        bool next_alloc = get_alloc(temp);
        if (temp == heap_start) {
            temp_prev = prologue;
        } else {
            temp_prev = find_prev(temp);
        }

        // last alloc bit consistency
        if (get_last_alloc(temp_next) != next_alloc) {
            dbg_printf("last alloc consistency failed\n");
            return false;
        }
        // mini check
        if ((get_last_mini(temp_next) == true && get_size(temp) != 16) ||
            (get_last_mini(temp_next) == false && get_size(temp) == 16)) {
            dbg_printf("mini check returns false\n");
            return false;
        }
        // boundry
        if ((size_t)temp > ((size_t)mem_heap_hi() - 7 - min_block_size) ||
            (size_t)temp < (size_t)heap_start) {
            dbg_printf("boundry returns false\n");
            return false;
        }
        // alignment
        if (((size_t)temp & 0x7) != 0 || ((size_t)temp_payload & 0xF) != 0) {
            dbg_printf("alignment returns false\n");
            return false;
        }
        // header/footer check
        if (get_alloc(temp) == 0 && get_size(temp) != 16) {
            if (get_size(temp) != extract_size(*temp_footer) ||
                get_alloc(temp) != extract_alloc(*temp_footer)) {
                dbg_printf("header/footer returns false\n");
                return false;
            }
        }
        // check coalescing
        if (get_alloc(temp) == 0) {
            // first block / only block case
            if (temp_prev == NULL) {
                if (get_alloc(temp_next) == 0) {
                    dbg_printf("coalesce first returns false\n");
                    return false;
                }
            }
            // last block case
            if (temp_next == epilogue) {
                if (get_last_alloc(temp) == 0) {
                    dbg_printf("coalesce last returns false\n");
                    return false;
                }
            }
            // middle block case
            else if (get_last_alloc(temp) == 0 || get_alloc(temp_next) == 0) {
                dbg_printf("coalesce middle returns false\n");
                return false;
            }
        }

        temp = temp_next;
    }
    // check seglists
    if (!checkFree(line)) {
        dbg_printf("checkFree returns false\n");

        return false;
    }
    return true;
}

/**
 * @brief initialize the heap
 *
 * request space on an empty heap, add the block containing the space
 * requested into the corresponding seglist, write the prologue and epilogue,
 * and initialize the seglist buckets.
 *
 * @return if init was successful
 */
bool mm_init(void) {
    // Create the initial empty heap
    word_t *start = (word_t *)(mem_sbrk(2 * wsize));

    if (start == (void *)-1) {
        return false;
    }

    start[0] = pack(0, true, false, false); // Heap prologue (block footer)
    start[1] = pack(0, true, true, false);  // Heap epilogue (block header)

    // Heap starts with first "block header", currently the epilogue
    heap_start = (block_t *)&(start[1]);
    for (int i = 0; i < 15; i++) {
        seglist[i] = NULL;
    }

    // Extend the empty heap with a free block of chunksize bytes
    if (extend_heap(chunksize) == NULL) {
        return false;
    }
    return true;
}

/**
 * @brief allocate space of a given size
 *
 * Initialize the heap for the first request. Search for the free list for
 * a block of good fit. If a free block is found, split the free block and
 * add the extra space into seglist. If no fit is found, request for more
 * space. It updates the alloc/last alloc/mini bits, and returns a pointer
 * to the payload of the block allocated.
 *
 * @param[in] size
 * @return pointer to the payload of a block
 * @pre the heap and seglists must be valid
 */
void *malloc(size_t size) {
    dbg_requires(mm_checkheap(__LINE__));
    dbg_printf("CALLING MALLOC\n");
    size_t asize;      // Adjusted block size
    size_t extendsize; // Amount to extend heap if no fit is found
    block_t *block;
    void *bp = NULL;

    // Initialize heap if it isn't initialized
    if (heap_start == NULL) {
        dbg_printf("MALLOC CALLING MM_INIT()\n");
        mm_init();
    }

    // Ignore spurious request
    if (size == 0) {
        dbg_ensures(mm_checkheap(__LINE__));
        return bp;
    }

    // Adjust block size to include overhead and to meet alignment requirements
    if (size <= 8) {
        asize = 16;
    } else {
        asize = round_up(size + wsize, dsize);
    }

    // Search the free list for a fit
    block = find_fit(asize);

    // If no fit is found, request more memory, and then and place the block
    if (block == NULL) {
        // Always request at least chunksize
        extendsize = max(asize, chunksize);
        block = extend_heap(extendsize);

        // extend_heap returns an error
        if (block == NULL) {
            return bp;
        }
    }

    // The block should be marked as free
    dbg_assert(!get_alloc(block));

    // Mark block as allocated
    size_t block_size = get_size(block);
    bool last = get_last_alloc(block);
    bool mini = get_last_mini(block);
    write_block(block, block_size, true, last, mini);
    // at this point, allocated block is still in free list

    // Try to split the block if too large, upload the free list
    delete (block);
    block_t *excess = split_block(block, asize);
    dbg_assert(get_alloc(block));

    // add the extra free space to its correponding seglist bucket
    if (excess != NULL) {
        dbg_assert(!get_alloc(excess));
        insert(excess);
    }

    bp = header_to_payload(block);

    dbg_ensures(mm_checkheap(__LINE__));
    return bp;
}

/**
 * @brief free a block containing payload where the pointer points to
 *
 * This function update the header/footer of the block being freed and
 * the block next to it on heap. It then coalesce the block with adjacent
 * free block and update the seglist.
 *
 * @param[in] bp
 */
void free(void *bp) {
    dbg_requires(mm_checkheap(__LINE__));
    dbg_printf("CALLING FREE\n");
    if (bp == NULL) {
        return;
    }

    block_t *block = payload_to_header(bp);
    size_t size = get_size(block);

    // The block should be marked as allocated
    dbg_assert(get_alloc(block));

    // Mark the block as free
    bool last = get_last_alloc(block);
    bool mini = get_last_mini(block);
    write_block(block, size, false, last, mini);

    // Try to coalesce the block with its neighbors
    block = coalesce_block(block);
    insert(block);

    dbg_ensures(mm_checkheap(__LINE__));
}

/**
 * @brief resize allocated memory
 *
 * This function is used to resize the memory block
 * which is allocated bny malloc or calloc before,
 * and returns a pointer to the new, resized memory.
 *
 * @param[in] ptr
 * @param[in] size
 * @return a pointer to the resized memory
 */
void *realloc(void *ptr, size_t size) {
    block_t *block = payload_to_header(ptr);
    size_t copysize;
    void *newptr;

    // If size == 0, then free block and return NULL
    if (size == 0) {
        free(ptr);
        return NULL;
    }

    // If ptr is NULL, then equivalent to malloc
    if (ptr == NULL) {
        return malloc(size);
    }

    // Otherwise, proceed with reallocation
    newptr = malloc(size);

    // If malloc fails, the original block is left untouched
    if (newptr == NULL) {
        return NULL;
    }

    // Copy the old data
    copysize = get_payload_size(block); // gets size of old payload
    if (size < copysize) {
        copysize = size;
    }
    memcpy(newptr, ptr, copysize);

    // Free the old block
    free(ptr);

    return newptr;
}

/**
 * @brief allocate memory and set its content to 0
 *
 * This function allocate the requested memory, initialize its content to 0,
 * and returns a pointer to the block payload.
 *
 * @param[in] elements
 * @param[in] size
 * @return pointer to the payload of the block allocated
 */
void *calloc(size_t elements, size_t size) {
    void *bp;
    size_t asize = elements * size;

    if (elements == 0) {
        return NULL;
    }
    if (asize / elements != size) {
        // Multiplication overflowed
        return NULL;
    }

    bp = malloc(asize);
    if (bp == NULL) {
        return NULL;
    }

    // Initialize all bits to 0
    memset(bp, 0, asize);

    return bp;
}

/*
 *****************************************************************************
 * Do not delete the following super-secret(tm) lines!                       *
 *                                                                           *
 * 53 6f 20 79 6f 75 27 72 65 20 74 72 79 69 6e 67 20 74 6f 20               *
 *                                                                           *
 * 66 69 67 75 72 65 20 6f 75 74 20 77 68 61 74 20 74 68 65 20               *
 * 68 65 78 61 64 65 63 69 6d 61 6c 20 64 69 67 69 74 73 20 64               *
 * 6f 2e 2e 2e 20 68 61 68 61 68 61 21 20 41 53 43 49 49 20 69               *
 *                                                                           *
 * 73 6e 27 74 20 74 68 65 20 72 69 67 68 74 20 65 6e 63 6f 64               *
 * 69 6e 67 21 20 4e 69 63 65 20 74 72 79 2c 20 74 68 6f 75 67               *
 * 68 21 20 2d 44 72 2e 20 45 76 69 6c 0a c5 7c fc 80 6e 57 0a               *
 *                                                                           *
 *****************************************************************************
 */
