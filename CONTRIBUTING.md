# Contributing to TimeToAlign!

The full **Developers' Reference** lives on the documentation site at
`docs/page/contributing.qmd`. It covers:

- Repository layout and the seven-subpackage architecture (with the
  layered import-direction rules).
- Environment setup, `pip install -e ".[dev]"`, pre-commit registration.
- `timetoshare`, the TimeToShare package (the manifest schema and its
  validator, which checks every manifest the library reads and writes), is a
  runtime dependency in `[project] dependencies`, so every install brings it.
  Until a release of TimeToShare carries the schema version the library
  writes, it is a git reference to TimeToShare's `main`
  branch; PyPI refuses a direct reference in a published package's
  requirements, so a release of the library needs the registry range in its
  place.
- The test oracle `pymeasuremap` (the Measure Map reference implementation,
  the oracle for compressed measure-map rows) lives in the `conformance`
  dependency group (PEP 735), not in an extra, as a git reference to a fixed
  commit. A dependency group is not part of the published distribution
  metadata, so the reference never reaches PyPI. `pip install -e ".[dev]"`
  does not install it; run `pip install --group conformance` (pip 25.1 or
  later) as well, or use `tox`, which installs the group in every test
  environment. From a checkout with the sibling repositories,
  `pip install -e ../TimeToShare/python -e ../pyMeasureMap` installs both
  packages from source.
- The canonical build/test/publish flow via `tox` (envs: default, `lint`,
  `build`, `clean`, `publish`).
- Coding standards (style, typing, Google-style docstrings, error
  handling, pitch spelling).
- The Quarto + quartodoc documentation pipeline, including how to add a
  new tutorial or how-to notebook (jupytext source → `notebooks.csv` →
  sidebar entry → glossary update).
- The hard rule that **jupytext and quarto versions may only ever be
  upgraded, never downgraded**.
- Conventional Commits and how `release-please` (running on every push to
  `main`) computes the next version. Breaking changes use both `!` *and*
  a `BREAKING CHANGE:` footer.
- IDE guidance (PyCharm preferred).

To preview the developers' reference locally:

```bash
cd docs/page
quartodoc build
quarto render
# open _site/contributing.html
```

Or read the rendered version on the public site:
<https://timetoalign.github.io/contributing.html>.
