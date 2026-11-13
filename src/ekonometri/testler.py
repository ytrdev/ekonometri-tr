"""Klasik doğrusal regresyon varsayımlarının testleri — her test Türkçe karar ve öneri verir.

Tüm testler bağımsız fonksiyon olarak da kullanılabilir:
    ek.breusch_pagan(model), ek.white(model), ek.durbin_watson(model),
    ek.breusch_godfrey(model, gecikme=4), ek.jarque_bera(model), ek.vif(model), ek.reset(model)
Hepsini birden çalıştırmak için: ``model.varsayimlar()``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from scipy import stats

from ._yardimci import ekk_coz, p_cumle, p_yaz, sayi

ALFA = 0.05


@dataclass
class TestSonucu:
    ad: str
    varsayim: str
    h0: str
    istatistik: float
    p_degeri: float | None
    sorun_var: bool
    karar: str
    oneri: str = ""
    detay: dict = field(default_factory=dict)

    @property
    def simge(self) -> str:
        return "⚠️" if self.sorun_var else "✅"

    def __str__(self) -> str:
        p = f", {p_cumle(self.p_degeri)}" if self.p_degeri is not None else ""
        s = f"{self.simge} {self.ad} ({self.varsayim}): istatistik = {sayi(self.istatistik, 3)}{p}\n   H0: {self.h0}\n   {self.karar}"
        if self.sorun_var and self.oneri:
            s += f"\n   👉 Öneri: {self.oneri}"
        return s

    __repr__ = __str__


@dataclass
class VarsayimRaporu:
    testler: list[TestSonucu]
    notlar: list[str] = field(default_factory=list)

    @property
    def sorunlar(self) -> list[TestSonucu]:
        return [t for t in self.testler if t.sorun_var]

    def tablo(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Test": t.ad,
                    "Varsayım": t.varsayim,
                    "İstatistik": round(t.istatistik, 4),
                    "p": None if t.p_degeri is None else round(t.p_degeri, 4),
                    "Sonuç": "sorun var" if t.sorun_var else "uygun",
                }
                for t in self.testler
            ]
        )

    def __str__(self) -> str:
        bas = "🔎 Varsayım kontrolü"
        ozet = (
            f"{len(self.testler)} testten {len(self.sorunlar)} tanesinde sorun görüldü."
            if self.sorunlar
            else f"{len(self.testler)} testin hiçbirinde ciddi sorun görülmedi. 👍"
        )
        govde = "\n\n".join(str(t) for t in self.testler)
        notlar = ("\n\n▸ Notlar\n" + "\n".join(f"  • {n}" for n in self.notlar)) if self.notlar else ""
        return f"{bas}\n{ozet}\n\n{govde}{notlar}"

    __repr__ = __str__

    def _repr_html_(self) -> str:
        satir = "".join(
            f"<tr><td>{t.simge}</td><td style='text-align:left'>{t.ad}</td><td style='text-align:left'>{t.varsayim}</td>"
            f"<td>{sayi(t.istatistik, 3)}</td><td>{p_yaz(t.p_degeri) if t.p_degeri is not None else '—'}</td>"
            f"<td style='text-align:left'>{t.karar}{('<br><b>Öneri:</b> ' + t.oneri) if t.sorun_var and t.oneri else ''}</td></tr>"
            for t in self.testler
        )
        notlar = "".join(f"<li>{n}</li>" for n in self.notlar)
        return (
            "<div><b>🔎 Varsayım kontrolü</b><table><thead><tr><th></th><th style='text-align:left'>Test</th><th style='text-align:left'>Varsayım</th>"
            f"<th>İstatistik</th><th>p</th><th style='text-align:left'>Sonuç ve öneri</th></tr></thead><tbody>{satir}</tbody></table>"
            + (f"<ul>{notlar}</ul>" if notlar else "")
            + "</div>"
        )


# --------------------------------------------------------------------------- yardımcı
def _yardimci_r2(y: np.ndarray, Z: np.ndarray) -> float:
    _, e, _ = ekk_coz(Z, y)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1 - (e @ e) / sst) if sst > 0 else 0.0


def _sabitli(model) -> tuple[np.ndarray, list[int]]:
    """Tasarım matrisi + sabit olmayan sütunların indeksleri."""
    X = model.X.values
    degisken = [i for i, ad in enumerate(model.X.columns) if ad != "Intercept"]
    return X, degisken


def _log_bagimli(model) -> bool:
    from .regresyon import _log_mu

    return _log_mu(model.bagimli) is not None


# --------------------------------------------------------------------------- değişen varyans
def breusch_pagan(model, degiskenler: str | list[str] = "aciklayici", koenker: bool = True) -> TestSonucu:
    """Breusch-Pagan / Cook-Weisberg değişen varyans testi.

    degiskenler : "aciklayici" (tüm açıklayıcı değişkenler — Stata ``rhs``), "tahmin" (tahmin edilen değerler —
        Stata ``estat hettest`` varsayılanı) ya da sütun adları listesi.
    koenker : True → n·R² (Koenker 1981, normallik gerektirmez — Stata ``iid``); False → ESS/2 (Breusch-Pagan 1979
        ve Cook-Weisberg 1983, normal hata varsayar — Stata ``estat hettest`` varsayılanı).

    Stata eşleşmeleri: ``estat hettest`` = breusch_pagan(m, "tahmin", koenker=False);
    ``estat hettest, rhs iid`` = breusch_pagan(m) (bu kütüphanenin varsayılanı).
    """
    e2 = model.kalintilar.values**2
    n = model.n
    if isinstance(degiskenler, str) and degiskenler == "aciklayici":
        Z, _ = _sabitli(model)
        aciklama = "açıklayıcı değişkenler"
    elif isinstance(degiskenler, str) and degiskenler == "tahmin":
        Z = np.column_stack([np.ones(n), model.tahminler.values])
        aciklama = "tahmin edilen değerler"
    else:
        sutunlar = [degiskenler] if isinstance(degiskenler, str) else list(degiskenler)
        veri = model._veri if model._veri is not None else model.X
        eksik = [c for c in sutunlar if c not in veri.columns and c not in model.X.columns]
        if eksik:
            raise ValueError(f"Değişken bulunamadı: {eksik}")
        Z = np.column_stack([np.ones(n)] + [(veri[c] if c in veri.columns else model.X[c]).values.astype(float) for c in sutunlar])
        aciklama = ", ".join(sutunlar)
    if "Intercept" not in model.X.columns and degiskenler == "aciklayici":
        Z = np.column_stack([np.ones(n), Z])
    sd = Z.shape[1] - 1
    if koenker:
        lm = n * _yardimci_r2(e2, Z)
    else:
        g = e2 / (e2.sum() / n)
        _, u, _ = ekk_coz(Z, g)
        lm = float(np.sum((g - g.mean()) ** 2) - u @ u) / 2  # açıklanan kareler toplamı / 2
    p = float(stats.chi2.sf(lm, sd))
    sorun = p < ALFA
    return TestSonucu(
        ad="Breusch-Pagan"
        + ("" if (koenker and aciklama == "açıklayıcı değişkenler") else f" ({aciklama}{', Koenker' if koenker else ', normal hata'})"),
        varsayim="Sabit varyans",
        h0="Hata varyansı sabittir (homoskedastisite).",
        istatistik=lm,
        p_degeri=p,
        sorun_var=sorun,
        karar=(
            "p < 0,05: H0 reddedildi — değişen varyans var. Katsayılar yansız ama standart hatalar (ve t/p değerleri) güvenilmez."
            if sorun
            else "p ≥ 0,05: değişen varyansa dair güçlü kanıt yok."
        ),
        oneri="Sağlam standart hata kullan: ek.ekk(..., kovaryans='HC1') — Stata: regress ..., vce(robust)."
        + ("" if _log_bagimli(model) else " Bağımlı değişkenin logaritmasını almak da varyansı dengeleyebilir."),
        detay={"serbestlik": sd, "degiskenler": aciklama, "koenker": koenker},
    )


def white(model) -> TestSonucu:
    """White testi: e² açıklayıcı değişkenler, kareleri ve çapraz çarpımlarıyla ilişkili mi?"""
    X, deg = _sabitli(model)
    e2 = model.kalintilar.values**2
    cols = [np.ones(model.n)] + [X[:, i] for i in deg]
    for i, j in combinations_with_replacement(deg, 2):
        cols.append(X[:, i] * X[:, j])
    Z = np.column_stack(cols)
    # tekrarlanan sütunları ayıkla (ör. kukla² = kukla)
    _, idx = np.unique(np.round(Z, 12), axis=1, return_index=True)
    Z = Z[:, np.sort(idx)]
    rank = np.linalg.matrix_rank(Z)
    r2 = _yardimci_r2(e2, Z)
    lm = model.n * r2
    sd = rank - 1
    p = float(stats.chi2.sf(lm, sd))
    sorun = p < ALFA
    return TestSonucu(
        ad="White",
        varsayim="Sabit varyans",
        h0="Hata varyansı sabittir (homoskedastisite).",
        istatistik=lm,
        p_degeri=p,
        sorun_var=sorun,
        karar=(
            "p < 0,05: H0 reddedildi — değişen varyans (ya da model kurgu hatası) var."
            if sorun
            else "p ≥ 0,05: değişen varyansa dair güçlü kanıt yok."
        ),
        oneri="Sağlam standart hata kullan: kovaryans='HC1'. White testi kurgu hatasına da duyarlıdır; RESET testine de bak.",
        detay={"serbestlik": sd},
    )


# --------------------------------------------------------------------------- otokorelasyon
def durbin_watson(model) -> TestSonucu:
    """Durbin-Watson: 1. dereceden otokorelasyon (≈2 iyi, <1,5 pozitif, >2,5 negatif otokorelasyon şüphesi)."""
    e = model.kalintilar.values
    dw = float(np.sum(np.diff(e) ** 2) / np.sum(e**2))
    if dw < 1.5:
        sorun, karar = True, f"DW = {sayi(dw, 2)} < 1,5: pozitif otokorelasyon şüphesi."
    elif dw > 2.5:
        sorun, karar = True, f"DW = {sayi(dw, 2)} > 2,5: negatif otokorelasyon şüphesi."
    else:
        sorun, karar = False, f"DW = {sayi(dw, 2)} ≈ 2: belirgin 1. dereceden otokorelasyon yok."
    return TestSonucu(
        ad="Durbin-Watson",
        varsayim="Otokorelasyon yok",
        h0="Hatalar arasında 1. dereceden otokorelasyon yoktur.",
        istatistik=dw,
        p_degeri=None,
        sorun_var=sorun,
        karar=karar + " (Kesin karar için Breusch-Godfrey testine bak.)",
        oneri="Zaman serisinde HAC (Newey-West) standart hata kullan: kovaryans='HAC'; modele gecikmeli değişken eklemeyi düşün.",
    )


def breusch_godfrey(model, gecikme: int = 1) -> TestSonucu:
    """Breusch-Godfrey LM testi: hatalar kendi gecikmeleriyle ilişkili mi? (gecikmeli bağımlı değişken varken de geçerli)"""
    X, _ = _sabitli(model)
    e = model.kalintilar.values
    n = model.n
    gec = np.column_stack([np.concatenate([np.zeros(g), e[:-g]]) for g in range(1, gecikme + 1)])
    Z = np.column_stack([X, gec])
    _, u, _ = ekk_coz(Z, e)
    r2 = 1 - (u @ u) / np.sum((e - e.mean()) ** 2)
    lm = float(n * r2)
    p = float(stats.chi2.sf(lm, gecikme))
    sorun = p < ALFA
    return TestSonucu(
        ad=f"Breusch-Godfrey ({gecikme} gecikme)",
        varsayim="Otokorelasyon yok",
        h0=f"Hatalarda {gecikme}. dereceye kadar otokorelasyon yoktur.",
        istatistik=lm,
        p_degeri=p,
        sorun_var=sorun,
        karar=(
            "p < 0,05: H0 reddedildi — otokorelasyon var. Standart hatalar güvenilmez."
            if sorun
            else "p ≥ 0,05: otokorelasyona dair güçlü kanıt yok."
        ),
        oneri="kovaryans='HAC' ile Newey-West standart hata kullan; dinamik yapı (gecikmeli değişkenler) eksik olabilir.",
        detay={"gecikme": gecikme},
    )


# --------------------------------------------------------------------------- normallik
def jarque_bera(model) -> TestSonucu:
    """Jarque-Bera: kalıntıların çarpıklığı ve basıklığı normal dağılımla uyumlu mu?"""
    e = model.kalintilar.values
    n = len(e)
    m = e - e.mean()
    s = np.mean(m**3) / np.mean(m**2) ** 1.5
    kur = np.mean(m**4) / np.mean(m**2) ** 2
    jb = float(n / 6 * (s**2 + (kur - 3) ** 2 / 4))
    p = float(stats.chi2.sf(jb, 2))
    sorun = p < ALFA
    buyuk = n >= 100
    return TestSonucu(
        ad="Jarque-Bera",
        varsayim="Normal hatalar",
        h0="Hatalar normal dağılır.",
        istatistik=jb,
        p_degeri=p,
        sorun_var=sorun and not buyuk,
        karar=(
            (
                f"p < 0,05: hatalar normal dağılmıyor (çarpıklık = {sayi(s, 2)}, basıklık = {sayi(kur, 2)}). "
                + (
                    "Örneklem büyük (n ≥ 100) olduğu için merkezi limit teoremi sayesinde t/F testleri yaklaşık olarak geçerli; ciddi bir sorun değil."
                    if buyuk
                    else "Küçük örneklemde t/F testleri yanıltıcı olabilir."
                )
            )
            if sorun
            else "p ≥ 0,05: normallikten belirgin sapma yok."
        ),
        oneri="Aykırı değerleri incele, bağımlı değişkenin logaritmasını almayı dene ya da örneklemi büyüt.",
        detay={"carpiklik": float(s), "basiklik": float(kur)},
    )


# --------------------------------------------------------------------------- çoklu doğrusal bağlantı
def vif(model) -> pd.DataFrame:
    """Varyans Şişirme Faktörü (VIF) tablosu. VIF > 10 ciddi, > 5 dikkat."""
    X, deg = _sabitli(model)
    sonuc = []
    for i in deg:
        diger = [j for j in range(X.shape[1]) if j != i]
        r2 = _yardimci_r2(X[:, i], X[:, diger])
        v = 1 / (1 - r2) if r2 < 1 else float("inf")
        sonuc.append({"değişken": model.X.columns[i], "VIF": v})
    return pd.DataFrame(sonuc).set_index("değişken")


def vif_testi(model) -> TestSonucu:
    tablo = vif(model)
    # kare terim ve kendisi arasındaki doğal yüksek VIF'i ayrıca belirt
    yuksek = tablo[tablo["VIF"] > 10]
    kare_iliskili = [ad for ad in yuksek.index if "**" in ad or f"I({ad}" in " ".join(tablo.index)]
    gercek = [ad for ad in yuksek.index if ad not in kare_iliskili]
    en = float(tablo["VIF"].max()) if len(tablo) else 0.0
    sorun = len(gercek) > 0
    if sorun:
        karar = (
            "Yüksek çoklu doğrusal bağlantı (VIF > 10): "
            + ", ".join(f"{a} ({sayi(tablo.loc[a, 'VIF'], 1)})" for a in gercek)
            + ". Bu katsayıların standart hataları şişmiş olabilir."
        )
    elif kare_iliskili:
        karar = "VIF yalnızca bir değişken ile onun karesi arasında yüksek; bu beklenen bir durumdur ve sorun sayılmaz."
    else:
        karar = f"En yüksek VIF = {sayi(en, 2)}: ciddi çoklu doğrusal bağlantı yok."
    return TestSonucu(
        ad="VIF",
        varsayim="Çoklu doğrusal bağlantı yok",
        h0="Açıklayıcı değişkenler birbirinin güçlü doğrusal birleşimi değildir.",
        istatistik=en,
        p_degeri=None,
        sorun_var=sorun,
        karar=karar,
        oneri="Birbirine çok benzeyen değişkenlerden birini çıkar ya da birleştir (ör. endeks oluştur). Tahmin gücü etkilenmez, tek tek katsayılar kararsızlaşır.",
        detay={"tablo": tablo},
    )


# --------------------------------------------------------------------------- kurgu
def reset(model, us: int = 4) -> TestSonucu:
    """Ramsey RESET: modele ŷ², ŷ³, ŷ⁴ eklenince açıklama gücü anlamlı artıyor mu? (fonksiyonel biçim hatası)

    Varsayılan (us=4) Stata ``estat ovtest`` ile aynıdır: F(3, n − k − 3).
    """
    X = model.X.values
    yhat = model.tahminler.values
    # ŷ standartlaştırılarak kuvvetleri alınır: {1, ŷ, ŷ², …} ile {1, z, z², …} aynı uzayı kapsar (F değişmez), hesap kararlı olur
    z = (yhat - yhat.mean()) / yhat.std()
    ek = np.column_stack([z**i for i in range(2, us + 1)])
    Z = np.column_stack([X, ek])
    y = model.y.values
    _, e1, _ = ekk_coz(Z, y)
    q = ek.shape[1]
    sd = model.n - Z.shape[1]
    F = float(((model.sse - e1 @ e1) / q) / ((e1 @ e1) / sd))
    p = float(stats.f.sf(F, q, sd))
    sorun = p < ALFA
    return TestSonucu(
        ad="Ramsey RESET",
        varsayim="Doğru fonksiyonel biçim",
        h0="Model doğru kurulmuştur (doğrusal biçim yeterlidir).",
        istatistik=F,
        p_degeri=p,
        sorun_var=sorun,
        karar=(
            "p < 0,05: H0 reddedildi — fonksiyonel biçim hatası ya da önemli bir değişken eksik olabilir."
            if sorun
            else "p ≥ 0,05: fonksiyonel biçim hatasına dair güçlü kanıt yok."
        ),
        oneri="Logaritma, kare terimi (I(x**2)) ya da etkileşim terimi (x1:x2) eklemeyi dene; atlanmış değişkenleri düşün.",
        detay={"serbestlik": (q, sd)},
    )


# --------------------------------------------------------------------------- rapor
def varsayim_raporu(model, zaman_serisi: bool | None = None, gecikme: int | None = None) -> VarsayimRaporu:
    if zaman_serisi is None:
        zaman_serisi = isinstance(model.y.index, (pd.DatetimeIndex, pd.PeriodIndex)) or getattr(model, "_zaman_serisi", False)
    testler = [breusch_pagan(model), white(model), jarque_bera(model), reset(model)]
    if len(model.X.columns) > 2 or (len(model.X.columns) == 2 and "Intercept" not in model.X.columns):
        testler.append(vif_testi(model))
    notlar = []
    if zaman_serisi:
        g = gecikme or max(1, min(4, model.n // 10))
        testler += [durbin_watson(model), breusch_godfrey(model, g)]
    else:
        notlar.append("Veri yatay kesit görünüyor; otokorelasyon testleri atlandı (zaman serisiyse: model.varsayimlar(zaman_serisi=True)).")
    if model.kovaryans_turu != "klasik":
        notlar.append(f"Model zaten '{model.kovaryans_turu}' standart hatalarla tahmin edildi; değişen varyans çıkarımı etkilemez.")
        for t in testler:
            if t.varsayim == "Sabit varyans" and t.sorun_var:
                t.oneri = "Zaten sağlam standart hata kullanıyorsun; t/p değerleri bu açıdan güvenilir."
    return VarsayimRaporu(testler, notlar)


__all__ = [
    "TestSonucu",
    "VarsayimRaporu",
    "breusch_pagan",
    "white",
    "durbin_watson",
    "breusch_godfrey",
    "jarque_bera",
    "vif",
    "vif_testi",
    "reset",
    "varsayim_raporu",
]
