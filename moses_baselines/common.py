def read_smiles(smiles_file: str) -> list[str]:
    with open(smiles_file) as f:
        return [line.strip() for line in f]
