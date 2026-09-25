"""Display rounding retained for the paper's Churchland scores."""
from decimal import ROUND_FLOOR, Decimal
from beartype import beartype

@beartype
def floor_three(value: float) -> str:
    """Floor a reported score to three decimals, without float-boundary drift."""
    return str(Decimal(str(value)).quantize(Decimal('0.001'), rounding=ROUND_FLOOR))

