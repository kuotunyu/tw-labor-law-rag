"""Fail-closed provider budget accounting with exact decimal arithmetic."""

from decimal import Decimal

import pytest

from rag.provider_budget import BudgetLedger


def test_ledger_charges_exact_decimal_token_cost() -> None:
    ledger = BudgetLedger(
        cap_usd="5.00",
        authorized_cap_usd="5.00",
        input_per_million="0.30",
        output_per_million="2.50",
    )

    charged = ledger.record_usage(input_tokens=1_000, output_tokens=200)

    assert charged == Decimal("0.000800")
    assert ledger.spent_usd == Decimal("0.000800")
    assert ledger.remaining_usd == Decimal("4.999200")


def test_ledger_stops_before_request_whose_maximum_can_exceed_cap() -> None:
    ledger = BudgetLedger(
        cap_usd="0.01",
        authorized_cap_usd="5.00",
        input_per_million="0.30",
        output_per_million="2.50",
    )

    assert ledger.can_start(max_input_tokens=1_000, max_output_tokens=3_000)
    ledger.record_usage(input_tokens=1_000, output_tokens=1_000)
    assert not ledger.can_start(max_input_tokens=1_000, max_output_tokens=3_000)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cap_usd": "5.01"}, "authorized"),
        ({"cap_usd": "0"}, "positive"),
        ({"input_per_million": "0"}, "positive"),
        ({"output_per_million": "-1"}, "positive"),
    ],
)
def test_ledger_rejects_invalid_or_unauthorized_pricing(kwargs, message) -> None:
    values = {
        "cap_usd": "5.00",
        "authorized_cap_usd": "5.00",
        "input_per_million": "0.30",
        "output_per_million": "2.50",
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        BudgetLedger(**values)


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"),
    [(None, 1), (1, None), (-1, 1), (1, -1), (True, 1), (1, 1.5)],
)
def test_ledger_rejects_missing_or_invalid_actual_usage(
    input_tokens, output_tokens
) -> None:
    ledger = BudgetLedger(
        cap_usd="5.00",
        authorized_cap_usd="5.00",
        input_per_million="0.30",
        output_per_million="2.50",
    )

    with pytest.raises(ValueError, match="token usage"):
        ledger.record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


@pytest.mark.parametrize(
    ("max_input_tokens", "max_output_tokens"),
    [(None, 1), (1, None), (0, 1), (1, 0), (-1, 1), (1, -1)],
)
def test_ledger_rejects_missing_or_nonpositive_request_maxima(
    max_input_tokens, max_output_tokens
) -> None:
    ledger = BudgetLedger(
        cap_usd="5.00",
        authorized_cap_usd="5.00",
        input_per_million="0.30",
        output_per_million="2.50",
    )

    with pytest.raises(ValueError, match="conservative maxima"):
        ledger.can_start(
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
        )


def test_ledger_rejects_actual_charge_beyond_remaining_cap() -> None:
    ledger = BudgetLedger(
        cap_usd="0.001",
        authorized_cap_usd="5.00",
        input_per_million="0.30",
        output_per_million="2.50",
    )

    with pytest.raises(RuntimeError, match="budget cap"):
        ledger.record_usage(input_tokens=1_000, output_tokens=1_000)


def test_public_summary_contains_no_secret_or_request_content() -> None:
    ledger = BudgetLedger(
        cap_usd="5.00",
        authorized_cap_usd="5.00",
        input_per_million="0.30",
        output_per_million="2.50",
    )
    ledger.record_usage(input_tokens=10, output_tokens=20)

    assert ledger.public_summary() == {
        "cap_usd": "5.00",
        "spent_usd": "0.000053",
        "remaining_usd": "4.999947",
        "requests": 1,
        "input_tokens": 10,
        "output_tokens": 20,
        "input_per_million_usd": "0.30",
        "output_per_million_usd": "2.50",
    }
