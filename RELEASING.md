# Releasing hypatorch

Releases are published to [PyPI](https://pypi.org/project/hypatorch/) automatically
by GitHub Actions whenever a **GitHub Release** is published. Authentication uses
PyPI **Trusted Publishing** (OIDC) — there are no API tokens or secrets to manage,
and every maintainer who can publish a release can publish a package.

## How to cut a release

1. Make sure `main` is green (the **Tests** workflow passes).
2. Pick the next version `X.Y.Z` ([semantic versioning](https://semver.org/)).
   You do **not** edit any version string by hand — the version is derived from
   the git tag by [setuptools-scm](https://setuptools-scm.readthedocs.io/).
3. Create the release (UI or CLI):

   **GitHub UI:** Releases → *Draft a new release* → *Choose a tag* → type
   `vX.Y.Z` → *Create new tag on publish* → target the commit you want
   (usually latest `main`) → add notes → **Publish release**.

   **CLI:**
   ```bash
   gh release create vX.Y.Z --target main --generate-notes
   ```

4. The **Publish** workflow runs automatically: full test matrix → build
   sdist+wheel → upload to PyPI. Watch it under the repo's *Actions* tab.
5. Confirm the new version appears at https://pypi.org/project/hypatorch/.

The tag **must** be `vX.Y.Z`. The workflow fails the build if the release tag
doesn't match the built version (e.g. the tag isn't on a clean commit), so a
mistagged release never reaches PyPI.

> **First release:** a `release` event runs the workflow from the *tagged
> commit*, so the first published release must be a **new tag created after
> these changes land on `main`** (e.g. `v1.0.3`). The pre-existing `v1.0.2` tag
> points at an older commit that has no workflow, so releasing it would not
> trigger a publish. (`1.0.1` is the latest on PyPI today.) If you specifically
> want `1.0.2` on PyPI, delete and recreate the `v1.0.2` tag on the new `main`
> HEAD first; otherwise just release `v1.0.3`.

## One-time setup (already done by a maintainer)

These two steps are what make trusted publishing work; they only need to be done
once for the project.

1. **PyPI** → project `hypatorch` → *Manage* → *Publishing* → add a **GitHub**
   trusted publisher:
   - Owner: `Altavo`
   - Repository: `hypatorch`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
2. **GitHub** → repo *Settings* → *Environments* → create an environment named
   `pypi`. Leave *required reviewers* **off** so releases publish without a
   manual gate (add reviewers only if you later want an approval step).

## Versioning notes

- The version comes from the latest `vX.Y.Z` git tag via setuptools-scm and is
  written to `hypatorch/_version.py` at build time (this file is git-ignored).
- Building from an untagged or dirty tree yields a dev version like
  `1.0.3.dev2+g<hash>`; that's expected for local/dev builds and is rejected by
  the release workflow's version check.
