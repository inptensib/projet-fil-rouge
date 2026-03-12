---
name: pdf-read
description: Lecture complete d'un PDF (texte, images, tableaux, diagrammes, metadonnees). Accepte un chemin local ou une URL. Utiliser des qu'un PDF doit etre lu ou analyse.
user-invocable: true
allowed-tools: Read, Bash
argument-hint: <path_or_url> [pages]
---

# PDF Read

Extraction complete du contenu d'un PDF : texte structure, images/diagrammes (rendus visuellement), tableaux, et metadonnees.

## Arguments

- `$0` : chemin local vers le PDF ou URL (http/https)
- `$1` : pages a traiter (optionnel). Formats : `1-10`, `5`, `all`. Default : 40 premieres pages.

## Taches

### 1. Resolution de la source

Si `$0` commence par `http` :
```bash
curl -sL -o /tmp/_pdf_read.pdf "$0"
```
Utiliser `/tmp/_pdf_read.pdf` comme chemin. Sinon, utiliser `$0` directement.

### 2. Metadonnees et table des matieres

```bash
/home/infres/tandriam-25/fil_rouge/venv/bin/python << 'PYEOF'
import fitz, json, sys
PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/_pdf_read.pdf"
doc = fitz.open(PDF_PATH)
meta = doc.metadata
toc = doc.get_toc()
info = {
    "title": meta.get("title", ""),
    "author": meta.get("author", ""),
    "subject": meta.get("subject", ""),
    "creator": meta.get("creator", ""),
    "pages": doc.page_count,
    "toc": [{"level": t[0], "title": t[1], "page": t[2]} for t in toc[:30]]
}
print(json.dumps(info, indent=2, ensure_ascii=False))
doc.close()
PYEOF
```

Presenter les metadonnees dans un tableau markdown. Si TOC non vide, lister les sections.

### 3. Extraction page par page

Traiter par chunks de 20 pages. Pour chaque page dans la plage demandee :

```bash
/home/infres/tandriam-25/fil_rouge/venv/bin/python << 'PYEOF'
import fitz, json, sys
PDF_PATH = sys.argv[1]
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0
END = int(sys.argv[3]) if len(sys.argv) > 3 else 20
doc = fitz.open(PDF_PATH)
end = min(END, doc.page_count)
for i in range(START, end):
    page = doc[i]
    text = page.get_text()
    images = page.get_images(full=True)
    drawings = page.get_drawings()
    # Tables
    tables_md = []
    try:
        tabs = page.find_tables()
        for t in tabs.tables:
            md = t.to_markdown()
            # Verifier si le tableau a du contenu (pas que des cellules vides)
            if any(c for row in t.extract() for c in row if c and c.strip()):
                tables_md.append(md)
    except Exception:
        pass
    # Render si contenu visuel
    has_visual = len(images) > 0 or len(drawings) > 5
    if has_visual:
        pix = page.get_pixmap(dpi=150)
        pix.save(f"/tmp/_pdf_page_{i}.png")
    result = {
        "page": i + 1,
        "text_length": len(text),
        "text_preview": text[:500],
        "num_images": len(images),
        "num_drawings": len(drawings),
        "tables": tables_md,
        "rendered": has_visual
    }
    print(json.dumps(result, ensure_ascii=False))
    print("---PAGE_SEP---")
doc.close()
PYEOF
```

### 4. Visualisation des pages avec figures

Pour chaque page rendue en PNG (`/tmp/_pdf_page_N.png`), utiliser l'outil **Read** pour voir l'image visuellement. Decrire le contenu des figures, diagrammes, et schemas.

### 5. Detection de PDF scanne

Si une page a des images mais aucun texte (`text_length == 0` et `num_images > 0`), c'est probablement un PDF scanne. Avertir :
> ⚠ Page N semble scannee (image sans texte). OCR non disponible (tesseract absent). Le rendu visuel est fourni.

### 6. Rapport final

Presenter :
1. **Metadonnees** (tableau)
2. **Table des matieres** (si disponible)
3. **Contenu par page** : texte + tableaux markdown + descriptions des figures
4. **Statistiques** : nombre total de pages, pages avec tableaux, pages avec figures

### 7. Cleanup

```bash
rm -f /tmp/_pdf_read.pdf /tmp/_pdf_page_*.png
```

## Garde-fous

- **PDFs > 50 pages** : avertir et traiter les 40 premieres sauf si l'utilisateur specifie une plage
- **PDFs proteges** : PyMuPDF leve une exception — informer l'utilisateur
- **Resolution PNG** : 150 DPI (equilibre lisibilite / taille fichier)
