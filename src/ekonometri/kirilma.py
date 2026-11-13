"""Yapısal kırılmalı birim kök testi: Zivot-Andrews (1992) — Stata ``zandrews`` karşılığı.

>>> ek.zivot_andrews(veri["ln_gsyh"])                              # sabitte kırılma (model A), gecikme AIC ile
>>> ek.zivot_andrews(veri["ln_gsyh"], model="ikisi", gecikme=2)    # sabit ve trendde kırılma (model C), 2 gecikme

H0: Seride (kırılmasız) birim kök vardır. H1: Tek bir bilinmeyen tarihte kırılmalı trend durağandır.
Her aday kırılma tarihi TB için ADF tipi regresyon tahmin edilir:

    Δy_t = μ + β t + θ DU_t + γ DT_t + α y_{t−1} + Σ c_j Δy_{t−j} + e_t,   DU_t = 1(t > TB),  DT_t = (t − TB)·1(t > TB)

Model "sabit" (A) yalnız DU, "trend" (B) yalnız DT, "ikisi" (C) ikisini de içerir. İstatistik, α'nın t değerinin
aday tarihler üzerindeki en küçüğüdür; kırılma tarihi bu en küçük değeri veren TB'dir. Kritik değerler ZA (1992) Tablo 2–4.
Aday tarihler, örneklemin baş ve sonundan ``kirpma`` oranı (varsayılan 0,15; ZA ve Stata zandrews) kadar dışarıda bırakılarak taranır.
Gecikme bilgi ölçütüyle seçilecekse seçim her aday tarih için ayrı yapılır (ZA, 1992).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._yardimci import DIL, l, sayi
from .regresyon import EkonometriHatasi

# Zivot ve Andrews (1992), Tablo 2 (A), 3 (B), 4 (C): asimptotik kritik değerler
KRITIK = {
    "sabit": {"%1": -5.34, "%5": -4.80, "%10": -4.58},
    "trend": {"%1": -4.93, "%5": -4.42, "%10": -4.11},
    "ikisi": {"%1": -5.57, "%5": -5.08, "%10": -4.82},
}
_MODEL_AD = {"sabit": "A (sabitte kırılma)", "trend": "B (trendde kırılma)", "ikisi": "C (sabit ve trendde kırılma)"}
_MODEL_EN = {"sabit": "A (break in intercept)", "trend": "B (break in trend)", "ikisi": "C (break in intercept and trend)"}


def _hata(m: str) -> EkonometriHatasi:
    return EkonometriHatasi(m)


@dataclass
class ZivotAndrewsSonucu:
    seri: str
    model: str
    istatistik: float
    kirilma: object  # kırılma tarihi (verinin indeks etiketi): DU, bu gözlemden SONRA 1 olur
    kirilma_sirasi: int  # 1 tabanlı gözlem sırası (TB)
    gecikme: int
    secim: str
    n: int
    kritik: dict
    t_serisi: pd.Series = field(repr=False, default=None)  # aday kırılma tarihi → t istatistiği
    _y: pd.Series = field(repr=False, default=None)

    @property
    def birim_kok_reddedildi(self) -> bool:
        return self.istatistik < self.kritik["%5"]

    def karar(self) -> str:
        if self.istatistik < self.kritik["%1"]:
            duzey = "%1"
        elif self.istatistik < self.kritik["%5"]:
            duzey = "%5"
        elif self.istatistik < self.kritik["%10"]:
            duzey = "%10"
        else:
            return (
                f"Birim kök boş hipotezi reddedilemedi (t = {sayi(self.istatistik, 3)} > %10 kritik değer {sayi(self.kritik['%10'], 2)}): "
                f"{self.seri} yapısal kırılmaya izin verildiğinde de birim köklüdür."
            )
        return (
            f"Birim kök boş hipotezi {duzey} düzeyinde reddedildi (t = {sayi(self.istatistik, 3)}): {self.seri}, "
            f"{self.kirilma} sonrasındaki yapısal kırılma dikkate alındığında durağandır."
        )

    def __str__(self) -> str:
        sec = {
            "sabit": "sabit",
            "aic": "AIC ile her aday tarihte seçildi",
            "bic": "BIC ile her aday tarihte seçildi",
            "t": "genelden özele t testiyle her aday tarihte seçildi",
        }[self.secim]
        return "\n".join(
            [
                f"Zivot-Andrews yapısal kırılmalı birim kök testi — {self.seri}",
                f"   Model {_MODEL_AD[self.model]} · n = {self.n} · gecikme = {self.gecikme} ({sec})",
                f"   En küçük t = {sayi(self.istatistik, 4)} · kırılma: {self.kirilma} (gözlem {self.kirilma_sirasi}; DU bu gözlemden sonra 1)",
                "   Kritik değerler (ZA 1992): " + "  ".join(f"{k}: {sayi(v, 2)}" for k, v in self.kritik.items()),
                f"   {self.karar()}",
            ]
        )

    __repr__ = __str__

    def grafik(self, baslik: str | None = None):
        """Seri ve tahmin edilen kırılma tarihi (dikey çizgi)."""
        from .grafik import _bitir, _stil, _yeni

        fig, ax = _yeni(7.0, 3.8)
        x = self._y.index
        x = x.to_timestamp() if isinstance(x, pd.PeriodIndex) else x
        ax.plot(x, self._y.values, color="#1f4e79", linewidth=1.4)
        ax.axvline(x[self.kirilma_sirasi - 1], color="#b22222", linestyle="--", linewidth=1.1, label=f"kırılma: {self.kirilma}")
        ax.legend(frameon=False, fontsize=8)
        ax.xaxis._kategorik = True
        _stil(ax, baslik, None, self.seri)
        return _bitir(fig, baslik or f"{self.seri}: Zivot-Andrews kırılma tarihi")

    def rapor(self, dosya=None, baslik=None, etiketler=None, tablo_no: int = 1, dil: str = "tr", bicim: str = "apa"):
        return zivot_andrews_raporu([self], dosya, baslik, etiketler, tablo_no, dil, bicim)


def zivot_andrews_raporu(sonuclar, dosya=None, baslik=None, etiketler=None, tablo_no: int = 1, dil: str = "tr", bicim: str = "apa"):
    """Birden çok seri için tek Zivot-Andrews tablosu ve Türkçe / İngilizce metin."""
    from .rapor import KAYNAKLAR, Rapor, Tablo, _Adlar, rapor_ayari

    sonuclar = list(sonuclar)
    if not sonuclar:
        raise _hata("En az bir Zivot-Andrews sonucu ver.")
    token = rapor_ayari(dil, bicim)
    try:
        en = dil == "en"
        ad = _Adlar(etiketler)
        baslik = baslik or l("Zivot-Andrews Yapısal Kırılmalı Birim Kök Testi", "Zivot-Andrews unit root test with a structural break")
        modeller = sorted({s.model for s in sonuclar})
        model_metni = ", ".join((_MODEL_EN if en else _MODEL_AD)[m] for m in modeller)
        kirp = "15%" if en else "%15"
        if en:
            cumle = [
                "Because conventional unit root tests can mistake a structural break for a unit root, stationarity was also examined "
                f"with the Zivot and Andrews (1992) test, which allows one break at an unknown date (model {model_metni}). "
                f"The break date is chosen where the unit-root t statistic is smallest; {kirp} of the sample is trimmed at each end."
            ]
        else:
            cumle = [
                "Geleneksel birim kök testleri yapısal kırılmayı birim kök olarak algılayabildiğinden durağanlık, bilinmeyen bir tarihte tek "
                f"kırılmaya izin veren Zivot ve Andrews (1992) testiyle de incelenmiştir (model {model_metni}). Kırılma tarihi, birim kök t "
                f"istatistiğinin en küçük olduğu tarih olarak belirlenmiş; örneklemin baştan ve sondan {kirp}'i taramaya alınmamıştır."
            ]
        for s in sonuclar:
            a = ad.metin(s.seri)
            if en:
                cumle.append(
                    f"For {a} the minimum t statistic is {sayi(s.istatistik, 3)} with a break after {s.kirilma}; the unit-root null is "
                    + ("rejected at the 5% level." if s.birim_kok_reddedildi else "not rejected at the 5% level.")
                )
            else:
                cumle.append(
                    f"{a} için en küçük t istatistiği {sayi(s.istatistik, 3)} olup kırılma {s.kirilma} sonrasında gerçekleşmektedir; birim kök "
                    + (
                        "boş hipotezi %5 düzeyinde reddedilmektedir."
                        if s.birim_kok_reddedildi
                        else "boş hipotezi %5 düzeyinde reddedilememektedir."
                    )
                )
        satirlar = [
            [
                ad.yalin(s.seri),
                s.model.replace("sabit", "A").replace("trend", "B").replace("ikisi", "C"),
                str(s.kirilma),
                str(s.gecikme),
                sayi(s.istatistik, 3),
                sayi(s.kritik["%5"], 2),
                l("Reddedildi", "Rejected") if s.birim_kok_reddedildi else l("Reddedilemedi", "Not rejected"),
            ]
            for s in sonuclar
        ]
        t = Tablo(
            tablo_no,
            l("Zivot-Andrews birim kök testi sonuçları", "Zivot-Andrews unit root test results"),
            [
                l("Değişken", "Variable"),
                "Model",
                l("Kırılma", "Break"),
                l("Gecikme", "Lag"),
                "t",
                l("%5 kritik", "5% critical"),
                l("H₀ kararı", "Decision on H₀"),
            ],
            satirlar,
            l(
                "H₀: Seri birim köklüdür. Model A: sabitte, B: trendde, C: sabit ve trendde kırılma. Kritik değerler Zivot ve Andrews (1992); "
                "%1 / %10 değerleri A: −5,34 / −4,58; B: −4,93 / −4,11; C: −5,57 / −4,82.",
                "H₀: the series has a unit root. Model A: break in intercept, B: in trend, C: in both. Critical values from Zivot and "
                "Andrews (1992); 1% / 10% values A: −5.34 / −4.58; B: −4.93 / −4.11; C: −5.57 / −4.82.",
            ),
        )
        r = Rapor(
            baslik,
            [("paragraf", " ".join(cumle)), ("tablo", t), ("baslik", l("Kaynakça", "References")), ("liste", [KAYNAKLAR["zivot_andrews"]])],
            bicim=bicim,
            dil=dil,
        )
        if dosya is not None:
            r.kaydet(dosya)
        return r
    finally:
        DIL.reset(token)


def _regresyon(y: np.ndarray, z: int, model: str, p: int, bas: int) -> tuple[np.ndarray, np.ndarray, float, int]:
    """TB = z (1 tabanlı) için ZA regresyonu: katsayılar, standart hatalar, kalıntı KT, gözlem sayısı.

    Sütun sırası: sabit, t, y_{t−1}, [DU], [DT], Δy_{t−1..p}. bas: ilk kullanılan gözlem (0 tabanlı).
    """
    n = len(y)
    t = np.arange(1, n + 1, dtype=float)
    dy = np.diff(y, prepend=np.nan)
    satir = np.arange(bas, n)
    sut = [np.ones(len(satir)), t[satir], y[satir - 1]]
    if model in ("sabit", "ikisi"):
        sut.append((t[satir] > z).astype(float))
    if model in ("trend", "ikisi"):
        sut.append(np.where(t[satir] > z, t[satir] - z, 0.0))
    sut += [dy[satir - j] for j in range(1, p + 1)]
    X = np.column_stack(sut)
    yy = dy[satir]
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ yy
    e = yy - X @ b
    rss = float(e @ e)
    s2 = rss / (len(yy) - np.linalg.matrix_rank(X))
    return b, np.sqrt(s2 * np.diag(XtX_inv)), rss, len(yy)


def zivot_andrews(
    seri,
    model: str = "sabit",
    gecikme: int | str = "aic",
    en_fazla_gecikme: int | None = None,
    kirpma: float = 0.15,
    ad: str | None = None,
) -> ZivotAndrewsSonucu:
    """Zivot-Andrews (1992) testi (Stata ``zandrews``).

    seri    : pandas Series (indeks tarih / dönem olabilir; kırılma bu etiketle raporlanır) ya da dizi
    model   : "sabit" (A, varsayılan; Stata zandrews break(intercept)), "trend" (B) ya da "ikisi" (C, break(both))
    gecikme : Δy gecikmesi; sayı ya da "aic" (varsayılan) / "bic" / "t" (her aday tarihte ayrı seçim; "t": genelden özele,
              son gecikme %10'da anlamlı). Üst sınır en_fazla_gecikme, verilmezse int(12·(n/100)^{1/4}).
    kirpma  : taranmayan uç oranı (0 ile 0,5 arası; 0 verilirse R urca::ur.za gibi tüm tarihler taranır)
    """
    if model not in KRITIK:
        raise _hata("model: 'sabit' (A), 'trend' (B) ya da 'ikisi' (C) olmalı.")
    if not 0 <= kirpma < 0.5:
        raise _hata("kirpma 0 ile 0,5 arasında olmalı.")
    s = seri if isinstance(seri, pd.Series) else pd.Series(np.asarray(seri, dtype=float))
    s = s.dropna().astype(float)
    ad = ad or (str(s.name) if s.name is not None else "seri")
    y = s.to_numpy()
    n = len(y)
    if n < 20:
        raise _hata("Zivot-Andrews testi için en az 20 gözlem gerekli.")
    if np.ptp(y) == 0:
        raise _hata(f"'{ad}' serisi sabit; test yapılamaz.")
    if isinstance(gecikme, str):
        secim = gecikme.lower()
        if secim not in ("aic", "bic", "t"):
            raise _hata("gecikme sayı ya da 'aic' / 'bic' / 't' olmalı.")
        pmax = en_fazla_gecikme if en_fazla_gecikme is not None else int(12 * (n / 100) ** 0.25)
    else:
        secim = "sabit"
        if int(gecikme) < 0:
            raise _hata("Gecikme negatif olamaz.")
        pmax = int(gecikme)
    bas_ortak = pmax + 1  # Δy_{t−pmax} tanımlı olan ilk gözlem (0 tabanlı)
    if kirpma > 0:
        adaylar = range(max(int(np.floor(kirpma * n)), bas_ortak + 2), min(int(np.ceil((1 - kirpma) * n)), n - 2) + 1)
    else:
        adaylar = range(1, n)
    t_deg, gec = {}, {}
    for z in adaylar:
        if secim == "sabit":
            p = pmax
        elif secim == "t":
            p = 0
            for k in range(pmax, 0, -1):  # genelden özele: son gecikmenin t'si |t| > 1,645 ise dur
                b, se, _, _ = _regresyon(y, z, model, k, bas_ortak)
                if abs(b[-1] / se[-1]) > 1.645:
                    p = k
                    break
        else:
            en_iyi = np.inf
            p = 0
            for k in range(pmax + 1):
                _, _, rss, nn = _regresyon(y, z, model, k, bas_ortak)
                m = 3 + (model == "ikisi") + 1 + k
                ic = np.log(rss / nn) + (2 * m if secim == "aic" else m * np.log(nn)) / nn
                if ic < en_iyi - 1e-12:
                    en_iyi, p = ic, k
        b, se, _, _ = _regresyon(y, z, model, p, p + 1)  # seçilen gecikme kendi örnekleminde
        t_deg[z] = float(b[2] / se[2])
        gec[z] = p
    if not t_deg:
        raise _hata("Aday kırılma tarihi kalmadı; kirpma ya da gecikmeyi küçült.")
    ts = pd.Series(t_deg)
    z = int(ts.idxmin())  # eşitlikte ilk tarih (R urca ile aynı)
    ts.index = [s.index[i - 1] for i in ts.index]
    ts.index.name = "kırılma"
    return ZivotAndrewsSonucu(ad, model, float(ts.min()), s.index[z - 1], z, gec[z], secim, n, dict(KRITIK[model]), ts, s)


__all__ = ["zivot_andrews", "zivot_andrews_raporu", "ZivotAndrewsSonucu", "KRITIK"]
