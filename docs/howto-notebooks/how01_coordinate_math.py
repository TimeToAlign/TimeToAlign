# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How to Do Coordinate Math
#
# Domains, TimeUnits, NumberTypes, and type-safe Coordinate arithmetic.

# %%
from fractions import Fraction

import pandas as pd

from timetoalign import Coordinate, Domain, IdCoordinate, NumberType, TimeUnit

# %% [markdown]
# ## The Three Domains
#
# | Domain | Description | Examples |
# |--------|-------------|----------|
# | **Physical** | Real-world time | Seconds, samples |
# | **Logical** | Symbolic/musical | Beats, quarters, ticks |
# | **Graphical** | Visual/spatial | Pixels, centimetres |

# %%
list(Domain)

# %%
Domain.physical == Domain.ph == Domain("physical") == Domain("ph")

# %% [markdown]
# ## TimeUnits

# %%
unit_data = [
    {"unit": u.name, "domain": u.domain.name, "discrete": u.is_discrete}
    for u in TimeUnit
]
pd.DataFrame(unit_data).sort_values(["domain", "discrete", "unit"])

# %%
# Convenient aliases
aliases = {
    "TimeUnit.s": TimeUnit.seconds,
    "TimeUnit.ms": TimeUnit.milliseconds,
    "TimeUnit.q": TimeUnit.quarters,
    "TimeUnit.b": TimeUnit.beats,
    "TimeUnit.px": TimeUnit.pixels,
    "TimeUnit.pulses": TimeUnit.ticks,
    "TimeUnit.divs": TimeUnit.ticks,
}
pd.Series({k: v.name for k, v in aliases.items()}, name="resolves_to")

# %% [markdown]
# ## NumberType
#
# | Type | Python Type | Use Case |
# |------|-------------|----------|
# | `int` | `int` | Discrete units (samples, ticks) |
# | `float` | `float` | Physical time (seconds) |
# | `fraction` | `Fraction` | Exact rationals (beats, quarters) |

# %%
{
    "from int": NumberType.from_number(42),
    "from float": NumberType.from_number(3.14),
    "from Fraction": NumberType.from_number(Fraction(3, 4)),
}

# %% [markdown]
# ### Why Fractions Matter

# %%
float_sum = sum(0.1 for _ in range(10))
fraction_sum = sum(Fraction(1, 10) for _ in range(10))

{
    "10x float": float_sum,
    "10x float == 1": float_sum == 1,
    "10x fraction": fraction_sum,
    "10x fraction == 1": fraction_sum == 1,
}

# %% [markdown]
# ## Coordinates
#
# Immutable, hashable, type-safe value+unit pairs.

# %%
c1 = Coordinate(120, TimeUnit.ticks)
c2 = Coordinate(1.5, TimeUnit.seconds)
c3 = Coordinate(Fraction(3, 4), TimeUnit.quarters)

c1, c2, c3

# %%
{
    "value": c3.value,
    "unit": c3.unit,
    "number_type": c3.number_type,
    "domain": c3.domain,
}

# %% [markdown]
# ### Arithmetic
#
# Coordinates are *positions*, so the arithmetic is deliberately strict. The
# guiding distinction is between a position (a `Coordinate`) and a span between
# two positions (a `Duration`). The rules below fall straight out of that
# distinction.

# %% [markdown]
# **Comparisons.** Coordinates of the same unit compare as you would expect.

# %%
x = Coordinate(10, TimeUnit.seconds)
y = Coordinate(5, TimeUnit.seconds)

{"x > y": x > y, "x == y": x == y, "x <= y": x <= y}

# %% [markdown]
# **Subtracting two Coordinates gives a `Duration`.** The span between two
# positions is a duration, not another position. Reverse the operands and the
# duration is signed — `is_negative` / `is_positive` make the direction
# queryable.

# %%
forwards = x - y  # 10s - 5s
backwards = y - x  # 5s - 10s

{
    "x - y": forwards,
    "type": type(forwards).__name__,
    "is_positive": forwards.is_positive(),
    "y - x": backwards,
    "is_negative": backwards.is_negative(),
}

# %% [markdown]
# **A Coordinate plus or minus a Duration is a Coordinate** — shifting a
# position by a span lands on another position. Shifting by a bare number works
# the same way.

# %%
{
    "x - (x - y)": x - forwards,  # Coordinate - Duration -> Coordinate
    "x + 2": x + 2,  # Coordinate + number -> Coordinate
    "x - 2": x - 2,
}

# %% [markdown]
# **Adding two Coordinates is forbidden.** The sum of two positions is
# meaningless; subtract them instead to obtain the span between them.

# %%
try:
    x + y  # two positions cannot be added
except TypeError as e:
    print(f"TypeError: {e}")

# %% [markdown]
# **A Duration on the left of `+ Coordinate` is forbidden too** — write
# `coord + dur`, not `dur + coord`.

# %%
try:
    forwards + x  # Duration + Coordinate
except TypeError as e:
    print(f"TypeError: {e}")

# %% [markdown]
# **Multiplying two TimeScalars is forbidden.** Scaling a position by a number
# is fine; multiplying two positions (or two spans) is not.

# %%
try:
    x * y  # Coordinate * Coordinate
except TypeError as e:
    print(f"TypeError: {e}")

# %% [markdown]
# **Scaling by a number preserves the type.** A scaled Coordinate is a
# Coordinate; a scaled Duration is a Duration.

# %%
{
    "x * 2": x * 2,
    "x / 2": x / 2,
    "(x - y) * 2": forwards * 2,
    "(x - y) type": type(forwards * 2).__name__,
}

# %% [markdown]
# **Different units cannot be combined.** This is a separate rule from the
# two-Coordinates ban above: even subtraction — which *is* allowed between two
# Coordinates — refuses operands whose units disagree.

# %%
try:
    Coordinate(1, TimeUnit.ticks) - Coordinate(1, TimeUnit.seconds)
except TypeError as e:
    print(f"TypeError: {e}")

# %% [markdown]
# **Id-bearing coordinates preserve their timeline.** Subtracting two
# `IdCoordinate`s on the *same* timeline yields an `IdDuration` that keeps the
# `timeline_id`. Mixing two different timelines raises.

# %%
p = IdCoordinate(10, TimeUnit.quarters, timeline_id="clt_1")
q = IdCoordinate(4, TimeUnit.quarters, timeline_id="clt_1")
span = p - q

{
    "p - q": span,
    "type": type(span).__name__,
    "timeline_id": span.timeline_id,
}

# %% [markdown]
# ### Type Conversions
#
# `to_int()` takes a rounding mode, and `"round"` is the default — a bare call
# rounds to nearest, it does not truncate. Name the mode explicitly whenever
# the direction matters.

# %%
c = Coordinate(Fraction(7, 4), TimeUnit.quarters)

{
    "original": c,
    "to_float()": c.to_float(),
    "to_int() — default is 'round'": c.to_int(),
    "to_int('round')": c.to_int("round"),
    "to_int('floor')": c.to_int("floor"),
    "to_int('ceil')": c.to_int("ceil"),
    "to_int('truncate')": c.to_int("truncate"),
    "to_fraction()": c.to_fraction(),
}

# %%
original = Coordinate(100, TimeUnit.ticks)

{
    "original": original,
    "with_value(200)": original.with_value(200),
    "with_unit(samples)": original.with_unit(TimeUnit.samples),
}
