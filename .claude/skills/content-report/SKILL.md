---
name: content-report
description: Rapport structure depuis n'importe quelle source (PDF, page web, code, image, notebook, donnees). Auto-detecte le type et produit une analyse detaillee.
user-invocable: true
allowed-tools: Read, Bash, Glob, Grep, WebFetch
argument-hint: <path_or_url>
---

# Content Report

Genere un rapport complet et structure a partir de n'importe quelle source.

## Arguments

- `$0` : chemin local (fichier ou dossier) ou URL (http/https)

## Taches

### 1. Detection du type

Determiner le type de source :

**Si URL** (`$0` commence par `http`) :
- URL finit par `.pdf` → telecharger avec `curl -sL -o /tmp/_content_report.pdf "$0"`, traiter comme PDF
- Sinon → utiliser `WebFetch` pour recuperer le contenu HTML converti en markdown

**Si chemin local** :
- `.pdf` → traiter comme PDF (voir etape 2a)
- `.ipynb` → utiliser `Read` natif (support notebooks)
- `.png`, `.jpg`, `.jpeg`, `.gif`, `.svg`, `.webp` → utiliser `Read` natif (vision multimodale)
- `.py` → utiliser `Read` + analyse structure (voir etape 2c)
- `.md`, `.txt`, `.rst`, `.tex` → utiliser `Read` natif
- `.yaml`, `.yml`, `.json`, `.toml` → utiliser `Read` natif
- `.csv`, `.tsv` → utiliser `Read` + `head` pour apercu
- Dossier → `ls -la` + `Glob` pour inventaire, puis `Read` sur les fichiers cles
- Autre → `file --mime-type "$0"` pour identifier, puis `Read`

### 2a. Traitement PDF

Meme procedure que le skill `/pdf-read` :

```bash
/home/infres/tandriam-25/fil_rouge/venv/bin/python << 'PYEOF'
import fitz, json, sys
PDF_PATH = sys.argv[1]
doc = fitz.open(PDF_PATH)
meta = doc.metadata
toc = doc.get_toc()
print(json.dumps({
    "title": meta.get("title", ""),
    "author": meta.get("author", ""),
    "pages": doc.page_count,
    "toc": [{"level": t[0], "title": t[1], "page": t[2]} for t in toc[:30]]
}, indent=2, ensure_ascii=False))
# Extraire les 40 premieres pages
for i in range(min(40, doc.page_count)):
    page = doc[i]
    text = page.get_text()
    images = page.get_images(full=True)
    drawings = page.get_drawings()
    has_visual = len(images) > 0 or len(drawings) > 5
    if has_visual:
        pix = page.get_pixmap(dpi=150)
        pix.save(f"/tmp/_cr_page_{i}.png")
    tables_md = []
    try:
        for t in page.find_tables().tables:
            md = t.to_markdown()
            if any(c for row in t.extract() for c in row if c and c.strip()):
                tables_md.append(md)
    except Exception:
        pass
    print(json.dumps({
        "page": i+1, "text": text[:2000],
        "images": len(images), "tables": tables_md,
        "rendered": has_visual
    }, ensure_ascii=False))
doc.close()
PYEOF
```

Pour les pages rendues, utiliser `Read` sur `/tmp/_cr_page_N.png` pour vision.

### 2b. Traitement page web

Utiliser `WebFetch` avec le prompt :
> "Extraire le contenu complet de cette page en preservant la structure : titres, paragraphes, listes, code, tableaux, liens. Ne pas resumer."

### 2c. Traitement code Python

1. `Read` le fichier entierement
2. Analyser la structure :

```bash
grep -n "^class \|^def \|^import \|^from " "$0"
```

3. Compter les lignes : `wc -l "$0"`
4. Verifier les docstrings : `grep -c '\"\"\"' "$0"`

### 2d. Traitement image

Utiliser `Read` sur le fichier image. Claude voit l'image visuellement et peut la decrire.

### 2e. Traitement dossier

1. `ls -la "$0"` pour lister le contenu
2. `Glob` pour trouver les fichiers par pattern (`**/*.py`, `**/*.md`, etc.)
3. Lire les fichiers cles (README, __init__.py, configs)
4. Compter les fichiers par type

### 3. Generation du rapport

Produire TOUJOURS ce format :

```markdown
# Content Report: [nom du fichier ou URL]

## Source Info

| Propriete | Valeur |
|-----------|--------|
| Type | PDF / Web / Python / Image / Notebook / ... |
| Source | [chemin ou URL] |
| Taille | [taille du fichier si local] |
| Date | [date de modification si local] |

## Summary

[2-5 phrases resumant le contenu principal]

## Key Content

[Contenu principal organise par sections/chapitres]

## Figures & Diagrams

[Description des elements visuels, ou "Aucun detecte"]

## Tables

[Tableaux en markdown, ou "Aucun detecte"]

## Structure

[Pour PDF: TOC. Pour code: hierarchie classes/fonctions. Pour web: structure des headings. Pour dossier: arborescence]

## References

[Liens, citations, imports, dependances trouves dans le contenu]

## Notes

[Observations : qualite, problemes, contenu manquant, encodage]
```

### 4. Cleanup

```bash
rm -f /tmp/_content_report.pdf /tmp/_cr_page_*.png
```
