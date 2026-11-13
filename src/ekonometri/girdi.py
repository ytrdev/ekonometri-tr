"""Kendi verini yükle: Stata (.dta), Excel, CSV, SPSS — ve Stata'daki gibi xtset / tsset / label variable.

>>> veri = ek.oku("anket.dta")              # değişken etiketleri ve değer etiketleri korunur
>>> veri = ek.oku("veri.xlsx", sayfa="2023")
>>> veri = ek.oku("tuik.csv")                # ';' ayırıcı ve ondalık virgül otomatik tanınır
>>> veri = ek.xtset(veri, "il", "yil")       # panel yapısını tanımla (Stata: xtset il yil)
>>> ek.etiketle(veri, {"gsyh": "Kişi başı GSYH (TL)"})   # Stata: label variable gsyh "..."

Etiketler raporlarda değişken adı yerine kullanılır.
"""

from __future__ import annotations

import csv
import io
import keyword
import re
import unicodedata
from pathlib import Path

import pandas as pd


def _hata(m: str):
    from .regresyon import EkonometriHatasi

    return EkonometriHatasi(m)


_TR = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def gecerli_ad(ad: str) -> str:
    """Sütun adını formülde kullanılabilir hâle getirir: 'Aylık Ücret (TL)' -> 'aylik_ucret_tl' (Stata adlandırma kurallarına yakın)."""
    s = str(ad).strip().translate(_TR)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^0-9A-Za-z_]+", "_", s).strip("_").lower()
    if not s:
        s = "degisken"
    if s[0].isdigit():
        s = "v" + s
    return s[:32]


def _adlari_duzelt(df: pd.DataFrame, etiketler: dict) -> tuple[pd.DataFrame, dict]:
    yeni, kullanilan = {}, set()
    for c in df.columns:
        ad = str(c)
        if not ad.isidentifier() or keyword.iskeyword(ad):  # Türkçe harfler geçerli (Stata 14+ da Unicode ada izin verir)
            ad = gecerli_ad(ad)
        taban, i = ad, 2
        while ad in kullanilan:
            ad = f"{taban}_{i}"
            i += 1
        kullanilan.add(ad)
        yeni[c] = ad
        if ad != str(c) and str(c).strip() and c not in etiketler:
            etiketler[ad] = str(c).strip()
        elif c in etiketler and ad != c:
            etiketler[ad] = etiketler.pop(c)
    return df.rename(columns=yeni), etiketler


def _metin_oku(yol: Path, ayirici, ondalik, kodlama) -> pd.DataFrame:
    ham = yol.read_bytes()
    if kodlama is None:
        for k in ("utf-8-sig", "cp1254", "iso-8859-9"):
            try:
                metin = ham.decode(k)
                kodlama = k
                break
            except UnicodeDecodeError:
                continue
        else:
            raise _hata("Dosyanın karakter kodlaması anlaşılamadı; kodlama='utf-8' gibi belirt.")
    else:
        metin = ham.decode(kodlama)
    ornek = metin[:20000]
    if ayirici is None:
        try:
            ayirici = csv.Sniffer().sniff(ornek, delimiters=",;\t|").delimiter
        except csv.Error:
            ayirici = ","
    if ondalik is None:
        # ';' ile ayrılmış ve sayılarda virgül varsa Türkçe/Avrupa biçimi: 1.234,56
        ondalik = "," if ayirici in (";", "\t") and re.search(r"\d,\d", ornek) else "."
    binlik = "." if ondalik == "," else None
    return pd.read_csv(io.StringIO(metin), sep=ayirici, decimal=ondalik, thousands=binlik)


def oku(
    dosya: str | Path,
    sayfa: str | int | None = None,
    ayirici: str | None = None,
    ondalik: str | None = None,
    kodlama: str | None = None,
    adlari_duzelt: bool = True,
) -> pd.DataFrame:
    """Veri dosyasını okur; biçim uzantıdan anlaşılır.

    .dta (Stata)      : değişken etiketleri ``veri.attrs["etiketler"]`` içine alınır, değer etiketli değişkenler
                        kategoriye çevrilir (kategori sırası kod sırasıdır; ilk kategori = Stata'daki temel düzey).
    .xlsx / .xls      : ilk satır değişken adları (Stata ``import excel, firstrow``); ``sayfa`` ile sayfa seçilir.
    .csv / .txt / .tsv: ayırıcı (, ; sekme |), ondalık işareti (virgül/nokta) ve kodlama (UTF-8 / Windows-1254) otomatik bulunur.
    .sav (SPSS)       : ``pyreadstat`` paketi gerekir.

    adlari_duzelt : True ise boşluk / özel karakter içeren sütun adları formülde kullanılabilir hâle getirilir
        ('Aylık Ücret' → 'aylik_ucret'); özgün başlık değişken etiketi olarak saklanır.
    """
    yol = Path(dosya)
    if not yol.exists():
        raise _hata(f"Dosya bulunamadı: {yol}")
    uz = yol.suffix.lower()
    etiketler: dict[str, str] = {}
    deger_etiketleri: dict[str, dict] = {}
    if uz == ".dta":
        with pd.io.stata.StataReader(yol) as okuyucu:
            df = okuyucu.read(convert_categoricals=True, convert_dates=True)
            etiketler = {k: v for k, v in okuyucu.variable_labels().items() if v}
        with pd.io.stata.StataReader(yol) as okuyucu:
            kodlar = okuyucu.read(convert_categoricals=False, convert_dates=False)
        for c in df.columns:  # değer etiketleri: kod -> etiket (Stata ifadelerinde 'cinsiyet == 2' için)
            if isinstance(df[c].dtype, pd.CategoricalDtype):
                eslesme = pd.DataFrame({"k": kodlar[c], "e": df[c].astype(object)}).dropna().drop_duplicates("k")
                deger_etiketleri[c] = {(int(k) if float(k).is_integer() else k): str(e) for k, e in zip(eslesme["k"], eslesme["e"])}
    elif uz in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(yol, sheet_name=0 if sayfa is None else sayfa)
    elif uz in (".csv", ".txt", ".tsv"):
        df = _metin_oku(yol, "\t" if uz == ".tsv" and ayirici is None else ayirici, ondalik, kodlama)
    elif uz == ".sav":
        try:
            import pyreadstat
        except ImportError:
            raise _hata("SPSS dosyaları için 'pyreadstat' gerekli:  pip install pyreadstat") from None
        df, meta = pyreadstat.read_sav(str(yol), apply_value_formats=True, formats_as_category=True, formats_as_ordered_category=True)
        etiketler = {k: v for k, v in zip(meta.column_names, meta.column_labels) if v}
    else:
        raise _hata(f"Desteklenmeyen dosya türü: {uz!r}. Desteklenenler: .dta, .xlsx, .xls, .csv, .txt, .tsv, .sav")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [str(c) for c in df.columns]
    if adlari_duzelt:
        eski = list(df.columns)
        df, etiketler = _adlari_duzelt(df, etiketler)
        deger_etiketleri = {yeni: deger_etiketleri[e] for e, yeni in zip(eski, df.columns) if e in deger_etiketleri}
    df.attrs["etiketler"] = etiketler
    df.attrs["deger_etiketleri"] = deger_etiketleri
    df.attrs["kaynak"] = str(yol)
    return df


def kaydet(veri: pd.DataFrame, dosya: str | Path) -> Path:
    """Veriyi kaydeder: .dta (Stata, değişken etiketleriyle), .xlsx ya da .csv (UTF-8)."""
    yol = Path(dosya)
    uz = yol.suffix.lower()
    if uz == ".dta":
        etiket = {k: v[:80] for k, v in veri.attrs.get("etiketler", {}).items() if k in veri.columns}
        veri.to_stata(yol, write_index=False, variable_labels=etiket, version=118)
    elif uz in (".xlsx", ".xls"):
        veri.to_excel(yol, index=False)
    elif uz == ".csv":
        veri.to_csv(yol, index=False, encoding="utf-8-sig")
    else:
        raise _hata(f"Desteklenmeyen dosya türü: {uz!r}. Seçenekler: .dta, .xlsx, .csv")
    return yol


def etiketle(veri: pd.DataFrame, etiketler: dict[str, str]) -> pd.DataFrame:
    """Değişken etiketi tanımlar (Stata ``label variable x "Etiket"``); raporlarda ad yerine etiket kullanılır."""
    eksik = [c for c in etiketler if c not in veri.columns]
    if eksik:
        raise _hata(f"Veride olmayan değişken(ler): {eksik}")
    veri.attrs.setdefault("etiketler", {})
    veri.attrs["etiketler"] = {**veri.attrs["etiketler"], **etiketler}
    return veri


def xtset(veri: pd.DataFrame, birim: str, zaman: str, sessiz: bool = False) -> pd.DataFrame:
    """Panel yapısını tanımlar (Stata ``xtset birim zaman``). Veriyi birim ve zamana göre sıralar.

    Sonrasında ``ek.panel(...)`` / ``ek.panel_analizi(...)`` birim ve zaman vermeden çalışır.
    """
    for c, rol in ((birim, "birim"), (zaman, "zaman")):
        if c not in veri.columns:
            raise _hata(f"{rol} değişkeni {c!r} veride yok. Mevcut değişkenler: {list(veri.columns)}")
        if veri[c].isna().any():
            raise _hata(f"{rol} değişkeni {c!r} eksik değer içeriyor.")
    if veri.duplicated([birim, zaman]).any():
        ornek = veri.loc[veri.duplicated([birim, zaman], keep=False), [birim, zaman]].head(4).values.tolist()
        raise _hata(f"Aynı {birim}–{zaman} çifti birden fazla kez geçiyor (ör. {ornek}). Stata: 'repeated time values within panel'.")
    attrs = dict(veri.attrs)
    sirali = veri.sort_values([birim, zaman]).reset_index(drop=True)
    sirali.attrs = {**attrs, "xtset": (birim, zaman)}
    if not sessiz:
        Ti = sirali.groupby(birim, observed=True).size()
        z = sirali[zaman]
        print(
            f"Panel değişkeni: {birim} ({'dengeli' if Ti.nunique() == 1 else 'dengesiz'}, {len(Ti)} birim)\n"
            f"Zaman değişkeni: {zaman}, {z.min()} – {z.max()}"
            + ("" if Ti.nunique() == 1 else f"\n Birim başına gözlem: en az {Ti.min()}, en çok {Ti.max()}")
        )
    return sirali


def tsset(veri: pd.DataFrame, zaman: str, sessiz: bool = False) -> pd.DataFrame:
    """Zaman serisi yapısını tanımlar (Stata ``tsset zaman``): veriyi zamana göre sıralar ve dönemi indeks yapar."""
    if zaman not in veri.columns:
        raise _hata(f"Zaman değişkeni {zaman!r} veride yok.")
    if veri[zaman].duplicated().any():
        raise _hata(f"Zaman değişkeni {zaman!r} tekrarlanan değerler içeriyor (Stata: 'repeated time values in sample').")
    attrs = dict(veri.attrs)
    sirali = veri.sort_values(zaman).reset_index(drop=True)
    z = sirali[zaman]
    if pd.api.types.is_integer_dtype(z) and z.between(1000, 3000).all():
        sirali.index = pd.PeriodIndex([pd.Period(str(int(y)), freq="Y") for y in z])  # yıllık dönem
    elif isinstance(z.dtype, pd.PeriodDtype) or pd.api.types.is_datetime64_any_dtype(z):
        sirali.index = pd.Index(z)
    sirali.attrs = {**attrs, "tsset": zaman}
    if not sessiz:
        bosluk = z.diff().dropna()
        print(f"Zaman değişkeni: {zaman}, {z.iloc[0]} – {z.iloc[-1]}" + ("" if bosluk.nunique() <= 1 else " (boşluklu seri)"))
    return sirali


def gecikme(veri: pd.DataFrame, degisken: str, k: int = 1) -> pd.Series:
    """Gecikmeli değişken (Stata ``L.x`` / ``L2.x``): xtset ile tanımlıysa her birim içinde, zaman boşlukları dikkate alınarak."""
    return _zaman_islemi(veri, degisken, k)


def fark(veri: pd.DataFrame, degisken: str, k: int = 1) -> pd.Series:
    """Birinci fark (Stata ``D.x``); k > 1 ise k. derece fark (``D2.x``)."""
    s = veri[degisken].astype(float)
    for _ in range(k):
        gecici = veri[[c for c in veri.columns if c != degisken]].copy()
        gecici[degisken] = s
        gecici.attrs = dict(veri.attrs)
        s = s - _zaman_islemi(gecici, degisken, 1)
    return s


def _zaman_islemi(veri: pd.DataFrame, degisken: str, k: int) -> pd.Series:
    xt = veri.attrs.get("xtset")
    ts = veri.attrs.get("tsset")
    if xt is None and ts is None:
        raise _hata("Gecikme ve fark için önce zaman yapısını tanımla: ek.tsset(veri, 'yil') ya da ek.xtset(veri, 'il', 'yil').")
    birim, zaman = xt if xt else (None, ts)
    z = veri[zaman]
    if pd.api.types.is_numeric_dtype(z):
        # Stata gibi: L.x, t − k dönemindeki değerdir; o dönem yoksa eksik
        anahtar = pd.DataFrame({"b": veri[birim] if birim else 0, "t": z + k, "x": veri[degisken].values})
        hedef = pd.DataFrame({"b": veri[birim] if birim else 0, "t": z})
        sonuc = hedef.merge(anahtar, on=["b", "t"], how="left")["x"]
        sonuc.index = veri.index
        return sonuc.astype(float)
    if birim:
        return veri.groupby(birim, observed=True)[degisken].shift(k).astype(float)
    return veri[degisken].shift(k).astype(float)


__all__ = ["oku", "kaydet", "etiketle", "xtset", "tsset", "gecikme", "fark", "gecerli_ad"]
