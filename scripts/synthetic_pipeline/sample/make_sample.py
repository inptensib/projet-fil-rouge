#!/usr/bin/env python3
"""Génère un mini datasets_finaux/ pour prouver le pipeline hors-ligne (sans LLM).

Exerce les 4 chemins : classe sous la cible (GEN+OVS), classe orpheline (ORPH),
classe au-dessus de la cible (laissée intacte), et une fuite val délibérée.
"""
import json, os, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "demo_ds"
os.makedirs(OUT, exist_ok=True)
_pid = [10_000_000]  # ids numériques = lignes réelles


def row(cls, l1, code, i, desig=None, desc=None):
    _pid[0] += 1
    return {
        "productid": str(_pid[0]),
        "designation": desig or f"{cls} modèle {i}",
        "prdtypecode": code,
        "typelabel": l1,
        "producttype": [cls],
        "description": desc or f"{cls} {i} : produit {l1} de bonne qualité, livraison rapide.",
    }


train, val, test = [], [], []

# L2 sous la cible (25 réels) -> GEN + OVS
for i in range(25):
    train.append(row("Tournevis", "Bricolage", 1620, i))

# L2 sous la cible (8 réels) -> GEN + OVS ; la 1re ligne est un doublon de val (fuite)
leak = row("Arrosoir", "Jardin", 1620, 0,
           desig="Arrosoir identique fuite",
           desc="Cette ligne existe aussi dans val : elle doit être exclue du réservoir.")
train.append(leak)
for i in range(1, 8):
    train.append(row("Arrosoir", "Jardin", 1620, i))

# L2 AU-DESSUS de la cible (130 réels) -> laissée intacte (prouve la sélectivité)
for i in range(130):
    train.append(row("Marteau", "Bricolage", 1620, i))

# val / test + la ligne qui entre en collision avec le réservoir
val.append(dict(leak))  # même (designation, description) -> doit déclencher le leak-skip
for i in range(5):
    val.append(row("Tournevis", "Bricolage", 1620, 1000 + i))
for i in range(5):
    test.append(row("Marteau", "Bricolage", 1620, 2000 + i))


def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for o in rows:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


dump(os.path.join(OUT, "train_split.jsonl"), train)
dump(os.path.join(OUT, "val_split.jsonl"), val)
dump(os.path.join(OUT, "test_split.jsonl"), test)
print(f"train={len(train)} val={len(val)} test={len(test)} -> {OUT}")
