from decimal import ROUND_HALF_UP, Decimal


def classic_round(number):
    return int(Decimal(number).to_integral_value(rounding=ROUND_HALF_UP))
