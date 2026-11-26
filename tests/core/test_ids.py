"""Tests for core/ids.py."""

import hashlib
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pytest

import timetoalign.core.ids as ids_module
from timetoalign.core import TimeUnit
from timetoalign.core.ids import (
    IdMinter,
    _mix64,
    _refuse_held_ids,
    _unmix64,
    content_minter,
    reset_content_minters,
    resolve_id,
    resolve_ids,
    seed_from,
    timeline_id_prefix,
    validate_id,
)

# region The minter

_MASK = (1 << 64) - 1
_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def _rule_mix64(value: int) -> int:
    """SplitMix64's finalizer as the id rule words it, with plain integers."""
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _MASK
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _MASK
    return value ^ (value >> 31)


def _rule_ids(prefix: str, seed: int, length: int, count: int) -> list[str]:
    """The first *count* ids of a minter, computed from the rule alone."""
    start = _rule_mix64(seed % 2**64)
    minted = []
    for k in range(1, count + 1):
        value = _rule_mix64((start + k) % 2**64) % 36**length
        suffix = ""
        for _ in range(length):
            value, digit = divmod(value, 36)
            suffix = _DIGITS[digit] + suffix
        minted.append(f"{prefix}_{suffix}")
    return minted


def _rule_seed(*encoded: bytes) -> int:
    """The seed of already encoded parts, computed from the rule alone."""
    data = b"".join(len(part).to_bytes(8, "big") + part for part in encoded)
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


class TestMix64:
    """mix64 is a bijection: its inverse undoes it."""

    _VALUES = [0, 1, 2**63, 2**64 - 1, 0x9E3779B97F4A7C15] + [
        int(value)
        for value in np.random.default_rng(20261126).integers(
            0, 2**64, size=1000, dtype=np.uint64
        )
    ]

    def test_round_trip(self) -> None:
        assert [_unmix64(_mix64(value)) for value in self._VALUES] == self._VALUES
        assert [_mix64(_unmix64(value)) for value in self._VALUES] == self._VALUES

    def test_reference_values(self) -> None:
        assert _mix64(0x9E3779B97F4A7C15) == 0xE220A8397B1DCDAF
        assert _mix64(0) == 0

    def test_the_rule(self) -> None:
        assert [_mix64(value) for value in self._VALUES] == [
            _rule_mix64(value) for value in self._VALUES
        ]


class TestSeededIds:
    """A seed gives the same ids, by the rule, in every process."""

    def test_pinned(self) -> None:
        minter = IdMinter(seed=0)
        assert [minter.mint("note"), minter.mint("claim"), minter.mint("note")] == [
            "note_1be6k8ij6dzad",
            "claim_3cccv6pc7fclm",
            "note_0gloaudxbozjk",
        ]

    @pytest.mark.parametrize("seed", [0, 7, 2**64 - 1])
    @pytest.mark.parametrize("length", [13, 7, 1])
    def test_the_rule(self, seed: int, length: int) -> None:
        minter = IdMinter(seed, length=length)
        assert [minter.mint("note") for _ in range(20)] == _rule_ids(
            "note", seed, length, 20
        )


class TestMintMany:
    """mint_many answers what as many calls of mint answer."""

    @pytest.mark.parametrize("prefix", ["note", "score:clt", "perf:Szegedi‐1966:dpt"])
    @pytest.mark.parametrize("length", [13, 10, 7, 1])
    # The seed whose start is 2^64 - 10: the counter wraps past 2^64 mid-batch.
    @pytest.mark.parametrize("seed", [20261126, _unmix64(2**64 - 10)])
    def test_equals_repeated_mint(self, prefix: str, length: int, seed: int) -> None:
        batch = IdMinter(seed, length=length).mint_many(prefix, 50)
        one_by_one = IdMinter(seed, length=length)
        assert batch.type == pa.string()
        assert batch.to_pylist() == [one_by_one.mint(prefix) for _ in range(50)]

    def test_one_counter(self) -> None:
        minter = IdMinter(seed=3)
        minted = minter.mint_many("event", 3).to_pylist() + [minter.mint("event")]
        assert minted == _rule_ids("event", 3, 13, 4)

    def test_none(self) -> None:
        minter = IdMinter(seed=3)
        none = minter.mint_many("event", 0)
        assert (none.type, len(none), minter.mint("event")) == (
            pa.string(),
            0,
            _rule_ids("event", 3, 13, 1)[0],
        )

    def test_negative(self) -> None:
        with pytest.raises(
            ValueError, match="Cannot mint a negative number of ids: -1"
        ):
            IdMinter(seed=3).mint_many("event", -1)


class TestBounds:
    """The length is 1 to 13; the seed is taken modulo 2^64."""

    @pytest.mark.parametrize("length", [0, 14, -1])
    def test_length_refused(self, length: int) -> None:
        with pytest.raises(ValueError) as refused:
            IdMinter(seed=0, length=length)
        assert str(refused.value) == (
            f"An id length is an integer from 1 to 13, not {length!r}"
        )

    @pytest.mark.parametrize("length", [1, 13])
    def test_length_accepted(self, length: int) -> None:
        minter = IdMinter(seed=0, length=length)
        suffixes = [minter.mint("n")[2:] for _ in range(10)]
        assert (minter.length, {len(suffix) for suffix in suffixes}) == (
            length,
            {length},
        )

    def test_seed_modulo(self) -> None:
        assert IdMinter(seed=-1).seed == 2**64 - 1
        assert IdMinter(seed=-1).mint("n") == IdMinter(seed=2**64 - 1).mint("n")
        assert IdMinter(seed=2**64).seed == 0
        assert IdMinter(seed=2**64).mint("n") == IdMinter(seed=0).mint("n")

    def test_seed_refused(self) -> None:
        with pytest.raises(TypeError) as refused:
            IdMinter(seed=cast(Any, "0"))
        assert str(refused.value) == "'str' object cannot be interpreted as an integer"

    def test_read_only(self) -> None:
        minter = IdMinter(seed=0)
        with pytest.raises(AttributeError):
            setattr(minter, "seed", 1)
        with pytest.raises(AttributeError):
            setattr(minter, "length", 1)


class TestSeedFrom:
    """seed_from hashes length-prefixed encoded parts."""

    @pytest.mark.parametrize(
        ("parts", "encoded", "expected"),
        [
            ((), (), 16406829232824261652),
            (("clt",), (b"clt",), 17977021633267006117),
            ((b"\x00\x01",), (b"\x00\x01",), 2031309672964614293),
            (
                ("clt", {"length": 16, "a": [1, 2]}),
                (b"clt", b'{"a":[1,2],"length":16}'),
                12473343271204303308,
            ),
        ],
    )
    def test_pinned(
        self, parts: tuple[object, ...], encoded: tuple[bytes, ...], expected: int
    ) -> None:
        assert (seed_from(*parts), _rule_seed(*encoded)) == (expected, expected)

    def test_parts_kept_apart(self) -> None:
        assert seed_from("ab", "c") != seed_from("a", "bc")

    def test_key_order_does_not_matter(self) -> None:
        assert seed_from({"a": 1, "b": 2}) == seed_from({"b": 2, "a": 1})

    def test_refused(self) -> None:
        with pytest.raises(TypeError) as refused:
            seed_from("clt", object())
        assert str(refused.value) == (
            "Cannot compute an id seed from a object: it has no JSON form"
        )


class TestContentMinter:
    """One minter per content seed in the process, until reset."""

    def test_equal_content_distinct_ids(self) -> None:
        first = content_minter("region", 0, 4)
        second = content_minter("region", 0, 4)
        assert first is second
        assert [first.mint("region"), second.mint("region")] == [
            "region_16y3ae79lr7h9",
            "region_35z1ve4l9h0nl",
        ]
        assert first.seed == seed_from("region", 0, 4)

    def test_reset_starts_again(self) -> None:
        before = [content_minter("region", 0, 4).mint("region") for _ in range(3)]
        reset_content_minters()
        after = [content_minter("region", 0, 4).mint("region") for _ in range(3)]
        assert before == after == _rule_ids("region", seed_from("region", 0, 4), 13, 3)

    def test_length_is_part_of_the_key(self) -> None:
        assert content_minter("region", length=7) is not content_minter("region")
        assert content_minter("region", length=7).length == 7


def test_fresh_minters_differ() -> None:
    first, second = IdMinter(), IdMinter()
    assert first.seed != second.seed
    assert first.mint("note") != second.mint("note")


# endregion


# region No numbered generator, one collision check


@pytest.mark.parametrize(
    "name",
    [
        "mint_id",
        "mint_ids",
        "id_column",
        "_admit_id",
        "_plan_admissions",
        "_highest_minted_number",
    ],
)
def test_the_numbered_machinery_is_gone(name: str) -> None:
    assert not hasattr(ids_module, name)


class TestRefuseHeldIds:
    """One check: an id held in the scope or twice among newcomers refuses."""

    def test_free_ids_pass_and_change_nothing(self) -> None:
        taken = {"score": "timeline 'score'"}
        _refuse_held_ids([("a", "region 'a'"), ("b", "region 'b'")], taken)
        assert taken == {"score": "timeline 'score'"}

    def test_an_id_the_scope_holds_is_refused(self) -> None:
        with pytest.raises(ValueError) as raised:
            _refuse_held_ids(
                [("a", "region 'a'"), ("b", "region 'b'")],
                {"b": "timeline 'b'"},
                scope="group 'g'",
            )
        assert str(raised.value) == (
            "region 'b' cannot join group 'g': its id 'b' is already held by "
            "timeline 'b'"
        )

    def test_an_id_two_newcomers_share_is_refused(self) -> None:
        with pytest.raises(ValueError) as raised:
            _refuse_held_ids([("a", "timeline 'a'"), ("a", "region 'a'")], {})
        assert str(raised.value) == (
            "region 'a' cannot join: its id 'a' is already held by timeline 'a'"
        )

    @pytest.mark.parametrize("held", ["clt_0a6t3vcta56ll", "score"])
    def test_minted_and_given_ids_alike(self, held: str) -> None:
        with pytest.raises(ValueError, match="is already held by"):
            _refuse_held_ids([(held, f"timeline '{held}'")], {held: "the manifest"})


# endregion


class TestValidateId:
    """An id is any non-empty string without a control character."""

    @pytest.mark.parametrize(
        "given",
        [
            "performance",
            "a",
            "Bass Kick",
            "score:clt_7",
            "a/b",
            "intro to part 2",
            "https://example.org/op35#m1",
            "start",
            "card",
            "markers_1",
            "regions_12",
        ],
    )
    def test_accepted(self, given: str) -> None:
        assert validate_id(given) == given

    @pytest.mark.parametrize("given", ["", "a\tb", "a\nb", "\x00", "\x1f", 3])
    def test_refused(self, given: Any) -> None:
        with pytest.raises(ValueError) as refused:
            validate_id(given)
        assert str(refused.value) == (
            f"Invalid id {given!r}: an id must be non-empty and contain no "
            "control character"
        )


@pytest.mark.parametrize(
    ("unit", "prefix"),
    [
        (TimeUnit.quarters, "clt"),
        (TimeUnit.ticks, "dlt"),
        (TimeUnit.seconds, "cpt"),
        (TimeUnit.samples, "dpt"),
        (TimeUnit.centimeters, "cgt"),
        (TimeUnit.pixels, "dgt"),
    ],
)
def test_timeline_id_prefix(unit: TimeUnit, prefix: str) -> None:
    assert timeline_id_prefix(unit) == prefix


# endregion


# region ID resolution


class TestResolveIds:
    """An exact id, else a regular expression matching whole ids."""

    def test_exact_match(self) -> None:
        assert resolve_ids("clt_1", ["clt_1", "clt_2", "dlt_1"]) == ["clt_1"]

    def test_exact_match_wins_over_the_expression(self) -> None:
        assert resolve_ids("clt_.", ["clt_1", "clt_.", "clt_2"]) == ["clt_."]

    def test_full_match(self) -> None:
        ids = ["score:clt_1", "perf:dlt_1", "perf:dlt_2"]
        assert resolve_ids("perf:.*", ids) == ["perf:dlt_1", "perf:dlt_2"]
        assert resolve_ids(r"clt_\d+", ["clt_1", "clt_2", "score:clt_1"]) == [
            "clt_1",
            "clt_2",
        ]

    @pytest.mark.parametrize("fragment", ["clt", "score", r"^perf:"])
    def test_a_fragment_matches_nothing(self, fragment: str) -> None:
        assert resolve_ids(fragment, ["score:clt_1", "clt_1", "perf:dlt_1"]) == []

    @pytest.mark.parametrize(
        ("pattern", "ids"),
        [("", ["clt_1", "clt_2"]), ("clt_1", []), ("[invalid", ["clt_1", "clt_2"])],
    )
    def test_nothing(self, pattern: str, ids: list[str]) -> None:
        assert resolve_ids(pattern, ids) == []


class TestResolveId:
    """The first of the matching ids, with a warning for several."""

    def test_exact_match(self) -> None:
        assert resolve_id("clt_1", ["clt_1", "clt_2", "dlt_1"]) == "clt_1"

    def test_several_warn_and_answer_the_first(self) -> None:
        with pytest.warns(UserWarning) as warned:
            assert resolve_id("clt_.*", ["clt_1", "clt_2", "dlt_1"]) == "clt_1"
        assert [str(warning.message) for warning in warned] == [
            "Pattern 'clt_.*' matches 2 IDs: ['clt_1', 'clt_2']. "
            "Returning first match: 'clt_1'"
        ]

    def test_several_without_warning(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert (
                resolve_id(
                    "perf:.*",
                    ["score:clt_1", "perf:dlt_1", "perf:dlt_2"],
                    warn_multiple=False,
                )
                == "perf:dlt_1"
            )

    def test_a_fragment_raises(self) -> None:
        with pytest.raises(KeyError) as refused:
            resolve_id("clt", ["clt_1", "clt_2"])
        assert (
            refused.value.args[0]
            == "No ID matches pattern 'clt'. Available: clt_1, clt_2"
        )


# endregion
