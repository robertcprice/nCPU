/* ncpu.h — C ABI for the native nCPU comprehension engine.
 *
 * Every result is decided by a synthesized, verified Mog program. Link against
 * the cdylib/staticlib built from the `mog_synth` crate (src/ffi.rs).
 */
#ifndef NCPU_H
#define NCPU_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque engine handle. */
typedef struct NcpuEngine NcpuEngine;

/* Build the engine (synthesizes the lexicon + rules once; takes a few seconds).
 * Returns an owned handle the caller must release with ncpu_engine_free. */
NcpuEngine *ncpu_engine_new(void);

/* Release an engine. Passing NULL is a no-op. */
void ncpu_engine_free(NcpuEngine *engine);

/* 1 if the action is semantically licensed (animate subject, inanimate object),
 * 0 if blocked, -1 on invalid input. */
int ncpu_comprehend_roles(const NcpuEngine *engine, const char *sentence);

/* 1 if the sentence is grammatical in subject-verb agreement, 0 if not,
 * -1 on invalid input. */
int ncpu_check_agreement(const NcpuEngine *engine, const char *sentence);

/* 1 if the word is an animate noun (a "person"), 0 if not, -1 on invalid input. */
int ncpu_is_person(const NcpuEngine *engine, const char *word);

/* Judge a conditional argument: 1 valid, 0 invalid, -1 unparseable/invalid. */
int ncpu_judge_argument(const NcpuEngine *engine, const char *sentence);

#ifdef __cplusplus
}
#endif

#endif /* NCPU_H */
