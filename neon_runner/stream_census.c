/* Exact, bounded-workspace record census for large LANU/LANP streams.
 * Non-cryptographic fingerprints only accelerate ordering. Equal fingerprints
 * are resolved by full metadata and payload comparison: collisions never imply
 * equal glyphs. No inference, no pruning and no timing/throughput claims.
 * POSIX implementation; compiled separately from the inference ABI with -lz.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <zlib.h>

typedef struct {
    uint64_t offset, count;
    uint32_t width, kind;
} CensusProjection;

typedef struct {
    uint64_t records, unique_records, payload_bytes, unique_payload_bytes;
    uint64_t zlib_bytes, workspace_bytes, largest_fingerprint_group;
    uint64_t unique_by_kind[8];
} CensusResult;

typedef struct { uint64_t fingerprint, location; } Record;
#define LOCATION_MASK UINT64_C(0x0fffffffffffffff)
static const unsigned widths[4] = {36, 48, 92, 132};

static int payload_compare(const Record *a, const Record *b, const uint8_t *stream) {
    unsigned ka = (unsigned)(a->location >> 60), kb = (unsigned)(b->location >> 60);
    if (ka != kb) return ka < kb ? -1 : 1;
    return memcmp(stream + (a->location & LOCATION_MASK),
                  stream + (b->location & LOCATION_MASK), widths[ka & 3u]);
}

static void sift(Record *a, size_t size, size_t root, const uint8_t *stream) {
    while (root < size / 2) {
        size_t child = root * 2 + 1;
        if (child + 1 < size && payload_compare(&a[child], &a[child + 1], stream) < 0) child++;
        if (payload_compare(&a[root], &a[child], stream) >= 0) break;
        Record t = a[root]; a[root] = a[child]; a[child] = t;
        root = child;
    }
}

static void sort_collision_group(Record *a, size_t size, const uint8_t *stream) {
    for (size_t i = size / 2; i > 0; --i) sift(a, size, i - 1, stream);
    for (size_t i = size; i > 1; --i) {
        Record t = a[0]; a[0] = a[i - 1]; a[i - 1] = t;
        sift(a, i - 1, 0, stream);
    }
}

static void radix_sort(Record *a, Record *scratch, size_t count) {
    Record *src = a, *dst = scratch;
    for (unsigned pass = 0; pass < 8; ++pass) {
        size_t histogram[256] = {0}, cursor[256];
        unsigned shift = pass * 8;
        for (size_t i = 0; i < count; ++i) histogram[(src[i].fingerprint >> shift) & 255u]++;
        size_t running = 0;
        for (unsigned b = 0; b < 256; ++b) { cursor[b] = running; running += histogram[b]; }
        for (size_t i = 0; i < count; ++i) dst[cursor[(src[i].fingerprint >> shift) & 255u]++] = src[i];
        Record *tmp = src; src = dst; dst = tmp;
    }
    /* Eight passes is even, so the final sorted result is in a. */
}

static uint64_t fingerprint(const uint8_t *p, unsigned width, unsigned kind, unsigned bits) {
    uint64_t h = UINT64_C(14695981039346656037) ^ (uint64_t)kind;
    for (unsigned j = 0; j < width; ++j) { h ^= p[j]; h *= UINT64_C(1099511628211); }
    /* Small bit counts are exposed only to prove collision handling in tests. */
    return bits == 64 ? h : h & ((UINT64_C(1) << bits) - 1);
}

static int zfeed(z_stream *z, const uint8_t *p, size_t count, int finish, uint64_t *output_bytes) {
    uint8_t output[65536];
    do {
        unsigned take = count > 8u * 1024u * 1024u ? 8u * 1024u * 1024u : (unsigned)count;
        z->next_in = (Bytef *)p; z->avail_in = take;
        int flush = finish && take == count ? Z_FINISH : Z_NO_FLUSH;
        int result;
        do {
            z->next_out = output; z->avail_out = sizeof(output);
            result = deflate(z, flush);
            if (result != Z_OK && result != Z_STREAM_END) return -1;
            *output_bytes += sizeof(output) - z->avail_out;
        } while (z->avail_in || z->avail_out == 0);
        p += take; count -= take;
        if (result == Z_STREAM_END) return count == 0 ? 0 : -1;
    } while (count);
    return 0;
}

/* Return 0 on success. The Python wrapper validates header and layout first.
 * Workspace includes two 16-byte arrays plus 1 MiB reserved for the zlib state,
 * radix histograms and stack buffers. File-backed mmap pages are not included
 * in that workspace number; it is not a total RSS/residency measurement.
 */
int census_glyphs_c(const char *path, const CensusProjection *projections, size_t projection_count,
                   uint64_t max_workspace, unsigned hash_bits, CensusResult *result,
                   char *error, size_t error_size) {
    int code = -1, fd = -1, z_ready = 0;
    struct stat before, after;
    uint8_t *stream = MAP_FAILED;
    Record *records = NULL, *scratch = NULL;
    CensusResult report = {0};
    z_stream z = {0};
#define FAIL(message) do { snprintf(error, error_size, "%s", message); goto cleanup; } while (0)
    if (!path || !projections || !projection_count || !result || !error || !error_size)
        return -1;
    if (hash_bits == 0 || hash_bits > 64) FAIL("hash_bits must be in 1..64");
    fd = open(path, O_RDONLY);
    if (fd < 0 || fstat(fd, &before) || before.st_size < 16) FAIL("Cannot open/stat stream");
    uint64_t cursor = 16;
    for (size_t i = 0; i < projection_count; ++i) {
        const CensusProjection *p = &projections[i];
        if (p->kind >= 8 || widths[p->kind & 3u] != p->width || p->offset != cursor || !p->count)
            FAIL("Invalid projection kind/width/count/contiguity");
        if (p->count > (UINT64_MAX - cursor) / p->width) FAIL("Projection byte count overflow");
        cursor += p->count * p->width;
        if (report.records > UINT64_MAX - p->count) FAIL("Glyph count overflow");
        report.records += p->count;
    }
    if (cursor != (uint64_t)before.st_size || cursor > LOCATION_MASK) FAIL("Layout/file size mismatch");
    if (report.records > (SIZE_MAX - 1048576u) / (2u * sizeof(Record))) FAIL("Workspace size overflow");
    report.workspace_bytes = report.records * 2u * sizeof(Record) + 1048576u;
    if (report.workspace_bytes > max_workspace) FAIL("Exact census exceeds the workspace budget");
    records = malloc((size_t)report.records * sizeof(Record));
    scratch = malloc((size_t)report.records * sizeof(Record));
    if (!records || !scratch) FAIL("Cannot allocate census arrays");
    stream = mmap(NULL, (size_t)before.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (stream == MAP_FAILED) FAIL("Cannot mmap stream");
    if (deflateInit(&z, 6) != Z_OK) FAIL("Cannot initialize zlib");
    z_ready = 1;
    size_t index = 0;
    for (size_t i = 0; i < projection_count; ++i) {
        const CensusProjection *p = &projections[i];
        uint64_t bytes = p->count * p->width;
        for (uint64_t j = 0; j < p->count; ++j) {
            uint64_t offset = p->offset + j * p->width;
            records[index].fingerprint = fingerprint(stream + offset, p->width, p->kind, hash_bits);
            records[index++].location = offset | ((uint64_t)p->kind << 60);
        }
        if (zfeed(&z, stream + p->offset, (size_t)bytes, 0, &report.zlib_bytes)) FAIL("zlib compression failed");
        report.payload_bytes += bytes;
        if ((i + 1) % 40 == 0) fprintf(stderr, "Census read %zu/%zu projections\n", i + 1, projection_count);
    }
    if (zfeed(&z, stream, 0, 1, &report.zlib_bytes)) FAIL("zlib finalization failed");
    fprintf(stderr, "Census ordering %llu records\n", (unsigned long long)report.records);
    radix_sort(records, scratch, (size_t)report.records);
    for (size_t begin = 0; begin < (size_t)report.records;) {
        size_t end = begin + 1;
        while (end < (size_t)report.records && records[end].fingerprint == records[begin].fingerprint) end++;
        size_t count = end - begin;
        if (count > report.largest_fingerprint_group) report.largest_fingerprint_group = count;
        int all_equal = 1;
        for (size_t j = begin + 1; j < end; ++j) {
            if (payload_compare(&records[begin], &records[j], stream) != 0) { all_equal = 0; break; }
        }
        if (!all_equal) sort_collision_group(records + begin, count, stream);
        for (size_t j = begin; j < end; ++j) {
            if (j == begin || (!all_equal && payload_compare(&records[j-1], &records[j], stream) != 0)) {
                unsigned kind = (unsigned)(records[j].location >> 60);
                report.unique_records++; report.unique_by_kind[kind]++;
                report.unique_payload_bytes += widths[kind & 3u];
            }
        }
        begin = end;
    }
    if (fstat(fd, &after) || before.st_size != after.st_size ||
        before.st_mtim.tv_sec != after.st_mtim.tv_sec || before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
        before.st_ctim.tv_sec != after.st_ctim.tv_sec || before.st_ctim.tv_nsec != after.st_ctim.tv_nsec)
        FAIL("Source changed during census");
    *result = report; code = 0;
cleanup:
    if (z_ready) deflateEnd(&z);
    if (stream != MAP_FAILED) munmap(stream, (size_t)before.st_size);
    if (fd >= 0) close(fd);
    free(records); free(scratch);
    return code;
#undef FAIL
}
