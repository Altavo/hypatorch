# Working in this repository

This package ships **agent documentation inside the wheel**, at `hypatorch/agent_docs/`.
Consumers copy that bundle out of the installed distribution, so what is written
there is what their agents read for the version they pinned — and a page that
describes a surface this package no longer has is worse than no page.

- A change to what a config can name — a field, a target, a parameter — updates
  the matching page in `hypatorch/agent_docs/` **in the same commit**. `test/surface.json`
  pins that surface; when the test fails, fix the page and re-pin with
  `UPDATE_SURFACE=1 pytest test/test_agent_docs.py`. The snapshot catches renamed,
  added and removed names, not changed meaning — a behaviour change under a
  stable name is yours to notice.
- Explanations of **why** the design is what it is, what was rejected, and what
  changed go in `docs/`, which does not ship.
- Pages in the bundle never link outside it: it is copied without the rest of
  this repository.
- Name another package only if this distribution depends on it. For a consumer
  or a sibling, name the role instead — a reference pointing up or across the
  dependency graph cannot be checked here and will go stale.
- Every page opens with a `Use this when ...` line, and `OVERVIEW.md` is the
  entry point a consumer links to.
