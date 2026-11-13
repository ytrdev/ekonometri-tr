"""Belge sitesinin kaynaklarını hazırlar (README, REHBER, defterler, doğrulama, değişiklik günlüğü, fonksiyon başvurusu).

Çalıştırma:
    python scripts/site_hazirla.py
    mkdocs serve          # yerel önizleme: http://127.0.0.1:8000
"""

import inspect
import re
import shutil
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
HEDEF = KOK / "site_kaynak"

BAGLANTILAR = {
    "REHBER.md": "rehber.md",
    "dogrulama/SONUC.md": "dogrulama.md",
    "CHANGELOG.md": "degisiklikler.md",
}
GITHUB = "https://github.com/ytrdev/ekonometri-tr/blob/main/"


def baglantilari_duzelt(metin: str) -> str:
    def degistir(m):
        hedef = m.group(2)
        if hedef.startswith(("http", "#", "mailto:")):
            return m.group(0)
        yol, _, capa = hedef.partition("#")
        if yol in BAGLANTILAR:
            yeni = BAGLANTILAR[yol]
        elif yol.startswith(("defterler/", "ornekler/")):
            yeni = yol
        else:
            yeni = GITHUB + yol  # sitede olmayan dosyalar GitHub'a
        return f"{m.group(1)}({yeni}{'#' + capa if capa else ''})"

    return re.sub(r"(\]|\]\s?)\(([^)\s]+)\)", degistir, metin)


def _ozet(nesne, ad: str) -> str:
    doc = (inspect.getdoc(nesne) or "").strip()
    if doc.startswith(ad + "("):  # dataclass'ın otomatik imza metni
        return ""
    doc = doc.split("\n\n")[0].replace("\n", " ").replace("|", "\\|").replace("``", "`")
    return doc if len(doc) <= 220 else doc[:217].rsplit(" ", 1)[0] + "…"


def _imza(nesne) -> str:
    try:
        imza = str(inspect.signature(nesne))
    except (TypeError, ValueError):
        return "(…)"
    imza = re.sub(r"\s+", " ", imza.replace("'", ""))
    imza = re.sub(r" -> .*$", "", imza)
    imza = imza if len(imza) <= 100 else imza[:97].rsplit(",", 1)[0] + ", …)"
    return imza.replace(" | ", " ∣ ")  # tablo hücresini bölmesin


def ana_sayfa(metin: str) -> str:
    """README'nin ortalanmış HTML başlığı sitede markdown olarak işlenmez; sade bir başlıkla değiştir."""
    metin = re.sub(r"^<div align=\"center\">.*?</div>\s*", "", metin, count=1, flags=re.S)
    baslik = "# ekonometri-tr\n\nTürkçe ekonometri kütüphanesi. Tahmin, Türkçe yorum, varsayım testi ve Word raporu.\n\n"
    return baslik + metin


def fonksiyon_sayfasi() -> str:
    import ekonometri as ek

    islevler, siniflar = [], []
    for ad in sorted(ek.__all__, key=str.lower):
        nesne = getattr(ek, ad, None)
        if nesne is None or ad.startswith("_") or inspect.ismodule(nesne) or not callable(nesne):
            continue
        (siniflar if inspect.isclass(nesne) else islevler).append((ad, nesne))
    satir = [
        "# Fonksiyon başvurusu",
        "",
        "`import ekonometri as ek` ile kullanılabilen fonksiyon ve sınıflar. Ayrıntılı açıklama için Python'da `help(ek.ad)`.",
        "",
        "## Fonksiyonlar",
        "",
        "| Fonksiyon | Açıklama |",
        "|---|---|",
    ]
    satir += [f"| `{ad}{_imza(n)}` | {_ozet(n, ad)} |" for ad, n in islevler]
    satir += ["", "## Sonuç sınıfları", "", "| Sınıf | Açıklama |", "|---|---|"]
    satir += [f"| `{ad}` | {_ozet(n, ad)} |" for ad, n in siniflar]
    return "\n".join(satir) + "\n"


def main() -> None:
    if HEDEF.exists():
        shutil.rmtree(HEDEF)
    (HEDEF / "defterler").mkdir(parents=True)
    for kaynak, hedef in [
        ("README.md", "index.md"),
        ("REHBER.md", "rehber.md"),
        ("dogrulama/SONUC.md", "dogrulama.md"),
        ("CHANGELOG.md", "degisiklikler.md"),
    ]:
        (HEDEF / hedef).write_text(baglantilari_duzelt((KOK / kaynak).read_text(encoding="utf-8")), encoding="utf-8")
    for nb in sorted((KOK / "defterler").glob("*.ipynb")):
        shutil.copy(nb, HEDEF / "defterler" / nb.name)
    index = HEDEF / "index.md"
    index.write_text(ana_sayfa(index.read_text(encoding="utf-8")), encoding="utf-8")
    shutil.copytree(KOK / "ornekler", HEDEF / "ornekler")
    (HEDEF / "fonksiyonlar.md").write_text(fonksiyon_sayfasi(), encoding="utf-8")
    print("hazır:", HEDEF)


if __name__ == "__main__":
    main()
