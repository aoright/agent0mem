# Invalid evaluation scripts (quarantined)

The scripts in this directory are NOT valid evaluations and must not be used to gate
official smoke or full submissions.

They query the exact questions (or entities) from previously seen official AML smoke
runs. Earlier versions of `app/database.py` contained hand-written answers keyed to those
same questions, so these scripts reported near-100% while the real benchmark score stayed
around 45-55%. The test was circular.

They are kept only for historical reference.

Use the held-out evaluator instead:

    AGENT0MEM_DATA_DIR=<isolated dir> python3 scripts/heldout/locomo_eval.py ingest ...
    AGENT0MEM_DATA_DIR=<isolated dir> python3 scripts/heldout/locomo_eval.py run --variant clean_wide ...

The held-out evaluator excludes every conversation that appeared in an official smoke run
and uses the official AML answer template and judge prompt verbatim.
