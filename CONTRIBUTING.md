# Contributing to TimeToAlign!

The full **Developers' Reference** lives on the documentation site at
`docs/page/contributing.qmd`. It covers:

- Repository layout and the seven-subpackage architecture (with the
  layered import-direction rules).
- Environment setup, `pip install -e ".[dev]"`, pre-commit registration.
- The test oracles live in the `conformance` dependency group (PEP 735), not in
  an extra: `timeskeleton` (the TimeSkeleton schema the conformance tests
  validate against) and `pymeasuremap` (the Measure Map reference
  implementation, the oracle for compressed measure-map rows). A dependency
  group is not part of the published distribution metadata, so its git
  references never reach PyPI. `pip install -e ".[dev]"` does not install it;
  run `pip install --group conformance` (pip 25.1 or later) as well, or use
  `tox`, which installs the group in every test environment. From a checkout
  with the sibling repositories, `pip install -e ../timeskeleton/python
  -e ../pyMeasureMap` works too.
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
