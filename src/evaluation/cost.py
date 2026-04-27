def empty_cost_summary() -> dict:
    return {
        "total_input_tokens": 0,
        "num_model_inputs": 0,
    }


def normalize_cost_summary(cost: dict | None) -> dict:
    if not cost:
        return empty_cost_summary()

    return {
        "total_input_tokens": int(
            cost.get("total_input_tokens", cost.get("total_tokens", 0))
        ),
        "num_model_inputs": int(
            cost.get("num_model_inputs", cost.get("calls", 0))
        ),
    }


def add_input_tokens(cost: dict, input_tokens: int) -> None:
    cost["total_input_tokens"] = int(cost.get("total_input_tokens", 0)) + int(input_tokens)
    cost["num_model_inputs"] = int(cost.get("num_model_inputs", 0)) + 1


def add_cost_summaries(left: dict, right: dict | None) -> dict:
    left = normalize_cost_summary(left)
    right = normalize_cost_summary(right)
    return {
        "total_input_tokens": left["total_input_tokens"] + right["total_input_tokens"],
        "num_model_inputs": left["num_model_inputs"] + right["num_model_inputs"],
    }


def diff_cost_summaries(after: dict | None, before: dict | None) -> dict:
    after = normalize_cost_summary(after)
    before = normalize_cost_summary(before)
    return {
        "total_input_tokens": after["total_input_tokens"] - before["total_input_tokens"],
        "num_model_inputs": after["num_model_inputs"] - before["num_model_inputs"],
    }
