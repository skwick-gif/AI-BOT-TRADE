def has_margin(margin_available: float, est_cost: float) -> bool:
	return margin_available >= est_cost


def valid_limit(pricing_mode: str, limit_price: float | None) -> bool:
	return pricing_mode != "LIMIT" or (
		limit_price is not None and limit_price > 0
	)
