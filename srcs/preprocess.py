def standardize(raw):
    # tandardize channel positions and names.
    rename = dict()
    for name in raw.ch_names:
        std_name = name.strip(".")
        std_name = std_name.upper()
        if std_name.endswith("Z"):
            std_name = std_name[:-1] + "z"
        if std_name.startswith("FP"):
            std_name = "Fp" + std_name[2:]
        rename[name] = std_name
    raw.rename_channels(rename)
