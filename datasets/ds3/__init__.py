def modify_path_stem(stem: str) -> str:
    stem = stem.split(".", maxsplit=1)[0]
    angle = "0"
    distance = "90"
    if stem.startswith(("+", "-")):  # angle
        angle = stem[:-1]
    else:  # distance
        distance = stem[:-1]

    if stem[-1] == "h":
        direction = "horizontal"
    elif stem[-1] == "v":
        direction = "vertical"
    else:
        raise ValueError("invalid path stem")

    return f'{direction}_{distance}_{angle}'
