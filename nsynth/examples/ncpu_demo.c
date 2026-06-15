/* ncpu_demo.c — drive the native nCPU comprehension engine from C.
 *
 * Build (from the nsynth crate root):
 *   cargo build --release --lib
 *   cc examples/ncpu_demo.c -I include -L target/release -lmog_synth \
 *      -o target/ncpu_demo                       # link the cdylib
 *   DYLD_LIBRARY_PATH=target/release target/ncpu_demo   # (LD_LIBRARY_PATH on Linux)
 */
#include <stdio.h>
#include "ncpu.h"

static const char *verdict(int v) {
    return v == 1 ? "yes" : (v == 0 ? "no " : "??");
}

int main(void) {
    fprintf(stderr, "[C] building engine (synthesizing verified programs)...\n");
    NcpuEngine *e = ncpu_engine_new();
    if (!e) {
        fprintf(stderr, "engine init failed\n");
        return 1;
    }

    printf("nCPU comprehension engine, driven from C:\n\n");

    printf("  selectional restriction (1=licensed, 0=blocked):\n");
    printf("    %s  the teacher writes the report\n",
           verdict(ncpu_comprehend_roles(e, "the teacher writes the report")));
    printf("    %s  the report writes the teacher\n",
           verdict(ncpu_comprehend_roles(e, "the report writes the teacher")));
    printf("    %s  the teacher helps the student   (animate object -> blocked)\n",
           verdict(ncpu_comprehend_roles(e, "the teacher helps the student")));

    printf("\n  subject-verb agreement (1=grammatical, 0=not):\n");
    printf("    %s  the captains watch the report\n",
           verdict(ncpu_check_agreement(e, "the captains watch the report")));
    printf("    %s  the captains watches the report  (the old bug)\n",
           verdict(ncpu_check_agreement(e, "the captains watches the report")));

    printf("\n  is-a-person (1=person, 0=thing):\n");
    printf("    %s  teacher\n", verdict(ncpu_is_person(e, "teacher")));
    printf("    %s  report\n", verdict(ncpu_is_person(e, "report")));

    printf("\n  logical validity (1=valid, 0=invalid):\n");
    printf("    %s  modus ponens\n", verdict(ncpu_judge_argument(e,
        "If the alarm rings, then the guard wakes. The alarm rings. Thus, the guard wakes.")));
    printf("    %s  affirming the consequent\n", verdict(ncpu_judge_argument(e,
        "If the alarm rings, then the guard wakes. The guard wakes. Thus, the alarm rings.")));

    ncpu_engine_free(e);
    printf("\nevery answer above was decided by a synthesized Mog program, called from C.\n");
    return 0;
}
