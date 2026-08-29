"""Exact, fail-closed token budget accounting for paid provider evaluations."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

_MILLION = Decimal(1_000_000)


def _positive_decimal(value: object, *, label: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be a positive decimal") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _token_count(value: object, *, maxima: bool) -> int:
    label = "conservative maxima" if maxima else "token usage"
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must use integer token counts")
    if (maxima and value <= 0) or (not maxima and value < 0):
        qualifier = "positive" if maxima else "non-negative"
        raise ValueError(f"{label} must be {qualifier}")
    return value


class BudgetLedger:
    """Track one provider's spend without floats or post-hoc overruns."""

    def __init__(
        self,
        *,
        cap_usd: object,
        authorized_cap_usd: object,
        input_per_million: object,
        output_per_million: object,
    ) -> None:
        self.cap_usd = _positive_decimal(cap_usd, label="cap_usd")
        authorized = _positive_decimal(
            authorized_cap_usd, label="authorized_cap_usd"
        )
        if self.cap_usd > authorized:
            raise ValueError("cap_usd exceeds the authorized cap")
        self.input_per_million = _positive_decimal(
            input_per_million, label="input_per_million"
        )
        self.output_per_million = _positive_decimal(
            output_per_million, label="output_per_million"
        )
        self.spent_usd = Decimal(0)
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def remaining_usd(self) -> Decimal:
        return self.cap_usd - self.spent_usd

    def cost(self, *, input_tokens: object, output_tokens: object) -> Decimal:
        inputs = _token_count(input_tokens, maxima=False)
        outputs = _token_count(output_tokens, maxima=False)
        return (
            Decimal(inputs) * self.input_per_million
            + Decimal(outputs) * self.output_per_million
        ) / _MILLION

    def can_start(
        self, *, max_input_tokens: object, max_output_tokens: object
    ) -> bool:
        inputs = _token_count(max_input_tokens, maxima=True)
        outputs = _token_count(max_output_tokens, maxima=True)
        maximum_cost = (
            Decimal(inputs) * self.input_per_million
            + Decimal(outputs) * self.output_per_million
        ) / _MILLION
        return maximum_cost <= self.remaining_usd

    def record_usage(
        self, *, input_tokens: object, output_tokens: object
    ) -> Decimal:
        inputs = _token_count(input_tokens, maxima=False)
        outputs = _token_count(output_tokens, maxima=False)
        charge = self.cost(input_tokens=inputs, output_tokens=outputs)
        if charge > self.remaining_usd:
            raise RuntimeError("reported usage would exceed the budget cap")
        self.spent_usd += charge
        self.requests += 1
        self.input_tokens += inputs
        self.output_tokens += outputs
        return charge

    def public_summary(self) -> dict[str, object]:
        """Return only numeric accounting fields safe for public evidence."""
        return {
            "cap_usd": str(self.cap_usd),
            "spent_usd": str(self.spent_usd),
            "remaining_usd": str(self.remaining_usd),
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_per_million_usd": str(self.input_per_million),
            "output_per_million_usd": str(self.output_per_million),
        }
