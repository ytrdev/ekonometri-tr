"""Zaman serisi: durağanlık (ADF, KPSS), eşbütünleşme, hata düzeltme ve Granger nedensellik.

>>> veri = ek.veri.yukle("ihracat")
>>> ek.adf(np.log(veri["ihracat"]))                         # tek seri
>>> ek.duraganlik(veri, ["ihracat", "reel_kur"], log=True)  # tez tablosu: düzey + birinci fark, I(0)/I(1) kararı
>>> ek.engle_granger("np.log(ihracat) ~ np.log(reel_kur)", veri)
>>> ek.hata_duzeltme("np.log(ihracat) ~ np.log(reel_kur)", veri)
>>> ek.granger(np.log(veri["ihracat"]), np.log(veri["reel_kur"]))
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ._mackinnon import TABLOLAR
from ._yardimci import DIL, ekk_coz, l, p_cumle, p_yaz, sayi, yildiz, yuzde
from .rapor import KAYNAKLAR, Rapor, Tablo, _Adlar, _p, _p_hucre

DETERMINISTIK = {"yok": "n", "sabit": "c", "trend": "ct"}
DET_ACIKLAMA = {"n": "sabitsiz", "c": "sabitli", "ct": "sabitli ve trendli"}
KPSS_KRITIK = {"c": [0.347, 0.463, 0.574, 0.739], "ct": [0.119, 0.146, 0.176, 0.216]}  # %10, %5, %2,5, %1 (Kwiatkowski vd., 1992)
KPSS_P = [0.10, 0.05, 0.025, 0.01]


def _hata(m: str):
    from .regresyon import EkonometriHatasi

    return EkonometriHatasi(m)


def _det(deterministik: str) -> str:
    if deterministik not in DETERMINISTIK:
        raise _hata(f"Bilinmeyen deterministik terim: {deterministik!r}. Seçenekler: 'sabit', 'trend' (sabit + trend), 'yok'")
    return DETERMINISTIK[deterministik]


def _seri(x, ad: str | None = None) -> tuple[np.ndarray, str]:
    if isinstance(x, pd.Series):
        ad = ad or (str(x.name) if x.name is not None else "seri")
        x = x.dropna().values
    else:
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        ad = ad or "seri"
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise _hata("Tek boyutlu bir seri ver (ör. veri['ihracat']).")
    if len(x) < 10:
        raise _hata(f"Zaman serisi testleri için en az 10 gözlem gerekli ({len(x)} var).")
    if np.ptp(x) == 0:
        raise _hata(f"'{ad}' serisi sabit; durağanlık testi yapılamaz.")
    return x, ad


def mackinnon_p(istatistik: float, det: str, N: int = 1) -> float:
    """MacKinnon (1994, 1996) yaklaşık asimptotik p-değeri."""
    t = TABLOLAR[det]
    if istatistik > t["max"][N - 1]:
        return 1.0
    if istatistik < t["min"][N - 1]:
        return 0.0
    kat = t["kucuk"][N - 1] if istatistik <= t["star"][N - 1] else t["buyuk"][N - 1]
    return float(stats.norm.cdf(np.polyval(kat[::-1], istatistik)))


def mackinnon_kritik(det: str, N: int = 1, n: float = np.inf) -> dict[str, float]:
    """MacKinnon (2010) kritik değerleri (%1, %5, %10); n verilirse sonlu örneklem düzeltmesiyle."""
    tablo = np.asarray(TABLOLAR[det]["kritik"])[N - 1]
    d = tablo[:, 0] if np.isinf(n) else np.array([np.polyval(r[::-1], 1 / n) for r in tablo])
    return {"%1": float(d[0]), "%5": float(d[1]), "%10": float(d[2])}


# --------------------------------------------------------------------------- sonuç nesnesi
@dataclass
class BirimKokSonucu:
    test: str
    seri: str
    istatistik: float
    p_degeri: float
    deterministik: str
    gecikme: int
    n: int
    kritik: dict[str, float]
    h0: str
    duragan: bool
    karar: str
    detay: dict = field(default_factory=dict)

    def __str__(self) -> str:
        k = " · ".join(f"{a}: {sayi(v, 3)}" for a, v in self.kritik.items())
        return (
            f"{'✅' if self.duragan else '⚠️'} {self.test} — '{self.seri}' ({DET_ACIKLAMA[self.deterministik]}, gecikme = {self.gecikme}, n = {self.n})\n"
            f"   istatistik = {sayi(self.istatistik, 3)}, {p_cumle(self.p_degeri)}   kritik değerler: {k}\n"
            f"   H0: {self.h0}\n   {self.karar}"
        )

    __repr__ = __str__


# --------------------------------------------------------------------------- ADF
def _adf_regresyon(x: np.ndarray, p: int, det: str, bas: int):
    """Δx_t = ρ·x_{t−1} + Σ_{i≤p} γ_i·Δx_{t−i} + [deterministik] + e_t.

    Gözlemler Δx dizisinin 'bas' indeksinden başlar; otomatik gecikme seçiminde tüm gecikmeler aynı (ortak) örneklemle karşılaştırılır.
    """
    dx = np.diff(x)
    y = dx[bas:]
    T = len(y)
    kol = [x[bas : bas + T]]  # x_{t-1}
    for i in range(1, p + 1):
        kol.append(dx[bas - i : bas - i + T])  # Δx_{t-i}
    Z = np.column_stack(kol)
    if det in ("c", "ct"):
        Z = np.column_stack([Z, np.ones(T)])
    if det == "ct":
        Z = np.column_stack([Z, np.arange(1, T + 1)])
    b, e, _ = ekk_coz(Z, y)
    sse = float(e @ e)
    k = Z.shape[1]
    se = np.sqrt(sse / (T - k) * np.linalg.inv(Z.T @ Z)[0, 0])
    ll = -T / 2 * (np.log(2 * np.pi) + np.log(sse / T) + 1)
    return b[0] / se, -2 * ll + 2 * k, -2 * ll + np.log(T) * k, T, b[0]


def adf(
    seri,
    deterministik: str = "sabit",
    gecikme: int | None = None,
    bilgi_olcutu: str = "aic",
    en_fazla_gecikme: int | None = None,
    ad: str | None = None,
) -> BirimKokSonucu:
    """Genişletilmiş Dickey-Fuller (ADF) birim kök testi. H0: seride birim kök var (durağan değil).

    deterministik : "sabit" (varsayılan), "trend" (sabit + doğrusal trend) ya da "yok"
    gecikme : sabit gecikme sayısı; verilmezse ``bilgi_olcutu`` ("aic" / "bic") ile seçilir
    en_fazla_gecikme : otomatik seçimde üst sınır (varsayılan: 12·(n/100)^(1/4), Schwert kuralı)
    """
    det = _det(deterministik)
    x, ad = _seri(seri, ad)
    n = len(x)
    nt = {"n": 0, "c": 1, "ct": 2}[det]
    sinir = n // 2 - nt - 1
    if gecikme is not None:
        if gecikme < 0 or gecikme > sinir:
            raise _hata(f"Gecikme 0 ile {sinir} arasında olmalı.")
        p, secim = gecikme, "sabit"
    else:
        if bilgi_olcutu not in ("aic", "bic"):
            raise _hata("bilgi_olcutu 'aic' ya da 'bic' olmalı.")
        pmax = en_fazla_gecikme if en_fazla_gecikme is not None else min(sinir, int(np.ceil(12 * (n / 100) ** 0.25)))
        if pmax < 0:
            raise _hata("Seri bu deterministik yapı için çok kısa.")
        sec = 1 if bilgi_olcutu == "aic" else 2
        p = min((_adf_regresyon(x, q, det, pmax)[sec], q) for q in range(pmax + 1))[1]
        secim = bilgi_olcutu.upper()
    tau, _, _, T, rho = _adf_regresyon(x, p, det, p)
    pv = mackinnon_p(tau, det)
    duragan = pv < 0.05
    karar = (
        f"p < 0,05: H0 reddedildi — '{ad}' serisi durağan (birim kök yok)."
        if duragan
        else f"p ≥ 0,05: H0 reddedilemedi — '{ad}' serisinde birim kök var (durağan değil). Birinci farkını test et."
    )
    return BirimKokSonucu(
        "ADF",
        ad,
        float(tau),
        pv,
        det,
        p,
        T,
        mackinnon_kritik(det, 1, T),
        "Seride birim kök vardır (seri durağan değildir).",
        duragan,
        karar,
        {"secim": secim, "rho": float(rho)},
    )


# --------------------------------------------------------------------------- KPSS
def kpss(seri, deterministik: str = "sabit", gecikme: int | None = None, ad: str | None = None) -> BirimKokSonucu:
    """KPSS durağanlık testi. H0: seri durağandır (ADF'nin tersi; ikisi birlikte kullanılır).

    gecikme verilmezse Hobijn vd. (1998) otomatik bant genişliği seçimi kullanılır. p-değeri Kwiatkowski vd. (1992)
    tablosundan doğrusal ara değerlemeyle bulunur ve 0,01–0,10 aralığıyla sınırlıdır.
    """
    det = _det(deterministik)
    if det == "n":
        raise _hata("KPSS testi için deterministik 'sabit' ya da 'trend' olmalı.")
    x, ad = _seri(seri, ad)
    n = len(x)
    if det == "ct":
        _, e, _ = ekk_coz(np.column_stack([np.ones(n), np.arange(1, n + 1)]), x)
    else:
        e = x - x.mean()
    if gecikme is None:
        s0 = float(e @ e) / n
        s1 = 0.0
        for i in range(1, int(n ** (2 / 9)) + 1):
            c = float(e[i:] @ e[: n - i]) / (n / 2)
            s0 += c
            s1 += i * c
        gecikme = min(int(1.1447 * ((s1 / s0) ** 2) ** (1 / 3) * n ** (1 / 3)), n - 1)
    elif gecikme >= n:
        raise _hata("Gecikme gözlem sayısından küçük olmalı.")
    s2 = float(e @ e)
    for i in range(1, gecikme + 1):
        s2 += 2 * (1 - i / (gecikme + 1)) * float(e[i:] @ e[:-i])
    s2 /= n
    ist = float(np.sum(np.cumsum(e) ** 2)) / n**2 / s2
    kr = KPSS_KRITIK[det]
    pv = float(np.interp(ist, kr, KPSS_P))
    duragan = pv >= 0.05
    karar = (
        f"p ≥ 0,05: H0 reddedilemedi — '{ad}' serisi {'düzeyde' if det == 'c' else 'trend etrafında'} durağan."
        if duragan
        else f"p < 0,05: H0 reddedildi — '{ad}' serisi durağan değil."
    )
    if pv == 0.01:
        karar += " (Gerçek p-değeri tablodaki en küçük değerden de küçük: p < 0,01.)"
    elif pv == 0.10:
        karar += " (Gerçek p-değeri tablodaki en büyük değerden de büyük: p > 0,10.)"
    return BirimKokSonucu(
        "KPSS",
        ad,
        ist,
        pv,
        det,
        int(gecikme),
        n,
        {"%1": kr[3], "%2,5": kr[2], "%5": kr[1], "%10": kr[0]},
        "Seri durağandır.",
        duragan,
        karar,
        {"sinirda": pv in (0.01, 0.10)},
    )


# --------------------------------------------------------------------------- toplu durağanlık tablosu
@dataclass
class DuraganlikSonucu:
    """Birden çok seri için düzey + birinci fark ADF/KPSS sonuçları ve bütünleşme derecesi kararı."""

    sonuclar: dict  # ad -> {"duzey": {"adf": .., "kpss": ..}, "fark": {...}}
    deterministik: str
    _etiketler: dict = field(default_factory=dict, repr=False)

    def derece(self, ad: str) -> str:
        """ADF'ye göre bütünleşme derecesi: 'I(0)', 'I(1)' ya da 'I(2) olabilir'."""
        s = self.sonuclar[ad]
        if s["duzey"]["adf"].duragan:
            return "I(0)"
        if s["fark"]["adf"].duragan:
            return "I(1)"
        return "I(2) olabilir"

    def _uyumlu(self, ad: str) -> bool:
        s = self.sonuclar[ad]
        return all(s[k]["adf"].duragan == s[k]["kpss"].duragan for k in ("duzey", "fark"))

    @staticmethod
    def _hucre(r: BirimKokSonucu) -> str:
        if r.test == "KPSS":
            return sayi(r.istatistik, 3) + ("**" if r.p_degeri < 0.05 else "")
        y = "***" if r.p_degeri < 0.01 else "**" if r.p_degeri < 0.05 else "*" if r.p_degeri < 0.10 else ""
        return f"{sayi(r.istatistik, 3)}{y} [{r.gecikme}]"

    def _tablo(self, etiketler: dict | None = None, no: int = 1) -> Tablo:
        ad = _Adlar({**self._etiketler, **(etiketler or {})})
        satirlar = [
            [
                ad.yalin(s),
                self._hucre(r["duzey"]["adf"]),
                _p_hucre(r["duzey"]["adf"].p_degeri),
                self._hucre(r["duzey"]["kpss"]),
                self._hucre(r["fark"]["adf"]),
                _p_hucre(r["fark"]["adf"].p_degeri),
                self._hucre(r["fark"]["kpss"]),
                self.derece(s),
            ]
            for s, r in self.sonuclar.items()
        ]
        ornek = next(iter(self.sonuclar.values()))
        if DIL.get() == "en":
            for satir in satirlar:
                if satir[-1] == "I(2) olabilir":
                    satir[-1] = "possibly I(2)"
            det = {"n": "no deterministic terms", "c": "a constant", "ct": "a constant and a linear trend"}[self.deterministik]
            notu = (
                f"The tests use a model with {det}. For ADF, H₀ is a unit root; the lag in brackets was selected by "
                f"{ornek['duzey']['adf'].detay['secim']}; p-values use the MacKinnon (1996) approximation. "
                f"For KPSS, H₀ is stationarity; the 5% critical value is {sayi(ornek['duzey']['kpss'].kritik['%5'], 3)}, and ** marks rejection at that value. "
                "Δ is the first difference. For ADF, *** p < 0.01; ** p < 0.05; * p < 0.10."
            )
            return Tablo(
                no,
                "Unit root test results",
                ["Variable", "ADF (level)", "p", "KPSS (level)", "ADF (Δ)", "p", "KPSS (Δ)", "Decision"],
                satirlar,
                notu,
            )
        notu = (
            f"Testler {DET_ACIKLAMA[self.deterministik]} modelle yapılmıştır. ADF testinde H₀: birim kök vardır; köşeli parantez içindeki değerler "
            f"{ornek['duzey']['adf'].detay['secim']} ile seçilen gecikme sayısıdır; p-değerleri MacKinnon (1996) yaklaşımıyla hesaplanmıştır. "
            f"KPSS testinde H₀: seri durağandır; %5 kritik değeri {sayi(ornek['duzey']['kpss'].kritik['%5'], 3)}, ** bu değerde reddi gösterir. "
            "Δ: birinci fark. ADF için *** p < 0,01; ** p < 0,05; * p < 0,10."
        )
        return Tablo(
            no,
            "Birim kök testi sonuçları",
            ["Değişken", "ADF (düzey)", "p", "KPSS (düzey)", "ADF (Δ)", "p", "KPSS (Δ)", "Karar"],
            satirlar,
            notu,
        )

    def tablo(self, etiketler: dict | None = None) -> pd.DataFrame:
        t = self._tablo(etiketler)
        return pd.DataFrame(t.satirlar, columns=t.sutunlar)

    def __str__(self) -> str:
        return (
            "Birim kök testi sonuçları (Δ: birinci fark)\n"
            + self.tablo().to_string(index=False)
            + "\nADF: *** p < 0,01; ** p < 0,05; * p < 0,10 · köşeli parantez: gecikme · KPSS: ** %5'te durağanlık reddi"
        )

    __repr__ = __str__

    def _repr_html_(self) -> str:
        return Rapor("Durağanlık testleri", [("tablo", self._tablo())]).html()

    def _metin_en(self, no: int, ad: _Adlar) -> str:
        c = [
            f"Stationarity was examined with the ADF test (Dickey and Fuller, 1979) and the KPSS test (Kwiatkowski et al., 1992). The results are reported in Table {no}."
        ]
        gruplar: dict[str, list] = {}
        for s in self.sonuclar:
            gruplar.setdefault(self.derece(s), []).append(ad.metin(s))
        if "I(0)" in gruplar:
            g = gruplar["I(0)"]
            c.append(f"The ADF test finds {', '.join(g)} stationary in levels, so {'it is' if len(g) == 1 else 'they are'} I(0).")
        if "I(1)" in gruplar:
            g = gruplar["I(1)"]
            c.append(
                f"{', '.join(g)} {'has' if len(g) == 1 else 'have'} a unit root in levels and {'becomes' if len(g) == 1 else 'become'} stationary in first differences, so {'it is' if len(g) == 1 else 'they are'} I(1)."
            )
        if "I(2) olabilir" in gruplar:
            c.append(
                f"For {', '.join(gruplar['I(2) olabilir'])}, the unit-root hypothesis is not rejected in the first difference either. The order of integration may exceed I(1), and the second difference should be examined."
            )
        celisen = [ad.metin(s) for s in self.sonuclar if not self._uyumlu(s)]
        if celisen:
            c.append(
                f"The KPSS test disagrees with the ADF test for {', '.join(celisen)}. The order of integration of these series should be read with caution."
            )
        else:
            c.append("The KPSS results support the ADF results.")
        dereceler = {self.derece(s) for s in self.sonuclar}
        if dereceler == {"I(1)"} and len(self.sonuclar) > 1:
            c.append(
                "Because every series is integrated of the same order, I(1), the long-run relationship can be examined with cointegration tests."
            )
        elif dereceler == {"I(0)", "I(1)"}:
            c.append(
                "Some series are I(0) and some are I(1), and none is I(2), so the long-run relationship can be examined with methods that allow mixed orders of integration, such as the ARDL bounds test."
            )
        return " ".join(c)

    def _metin(self, no: int, ad: _Adlar) -> str:
        if DIL.get() == "en":
            return self._metin_en(no, ad)
        c = [
            f"Değişkenlerin durağanlık özellikleri ADF (Dickey ve Fuller, 1979) ve KPSS (Kwiatkowski vd., 1992) testleriyle incelenmiş, sonuçlar Tablo {no} ile sunulmuştur."
        ]
        gruplar: dict[str, list] = {}
        for s in self.sonuclar:
            gruplar.setdefault(self.derece(s), []).append(ad.metin(s))
        if "I(0)" in gruplar:
            g = gruplar["I(0)"]
            c.append(
                f"ADF testine göre {', '.join(g)} {'değişkeni' if len(g) == 1 else 'değişkenleri'} düzeyde durağandır; dolayısıyla I(0) {'sürecidir' if len(g) == 1 else 'süreçleridir'}."
            )
        if "I(1)" in gruplar:
            g = gruplar["I(1)"]
            c.append(
                f"{', '.join(g)} {'değişkeni' if len(g) == 1 else 'değişkenleri'} düzeyde birim kök içermekte, birinci {'farkı' if len(g) == 1 else 'farkları'} alındığında "
                f"durağan hâle gelmektedir; dolayısıyla I(1) {'sürecidir' if len(g) == 1 else 'süreçleridir'}."
            )
        if "I(2) olabilir" in gruplar:
            c.append(
                f"{', '.join(gruplar['I(2) olabilir'])} için birinci farkta da birim kök hipotezi reddedilememiştir; bütünleşme derecesi I(1)'den yüksek olabilir ve ikinci farkın incelenmesi gerekir."
            )
        celisen = [ad.metin(s) for s in self.sonuclar if not self._uyumlu(s)]
        if celisen:
            c.append(
                f"KPSS testi {', '.join(celisen)} için ADF testinden farklı bir sonuca işaret etmektedir; bu değişkenlerin bütünleşme derecesi temkinli yorumlanmalıdır."
            )
        else:
            c.append("KPSS testi sonuçları ADF testi sonuçlarını desteklemektedir.")
        dereceler = {self.derece(s) for s in self.sonuclar}
        if dereceler == {"I(1)"} and len(self.sonuclar) > 1:
            c.append(
                "Tüm değişkenlerin aynı dereceden (I(1)) bütünleşik olması, aralarındaki uzun dönemli ilişkinin eşbütünleşme testleriyle sınanmasına olanak vermektedir."
            )
        elif dereceler == {"I(0)", "I(1)"}:
            c.append(
                "Değişkenlerin bir kısmının I(0), bir kısmının I(1) olması ve hiçbirinin I(2) olmaması, uzun dönem ilişkisinin ARDL sınır testi gibi bütünleşme derecesine esnek yaklaşımlarla incelenmesine uygundur."
            )
        return " ".join(c)

    def rapor(
        self,
        dosya: str | None = None,
        baslik: str | None = None,
        etiketler: dict | None = None,
        tablo_no: int = 1,
        dil: str = "tr",
        bicim: str = "apa",
    ) -> Rapor:
        from .rapor import rapor_ayari

        if baslik is None:
            baslik = "Stationarity analysis" if dil == "en" else "Durağanlık Analizi"
        token = rapor_ayari(dil, bicim)
        try:
            ad = _Adlar({**self._etiketler, **(etiketler or {})})
            r = Rapor(
                baslik,
                [
                    ("paragraf", self._metin(tablo_no, ad)),
                    ("tablo", self._tablo(etiketler, tablo_no)),
                    ("baslik", l("Kaynakça", "References")),
                    ("liste", sorted([KAYNAKLAR["df"], KAYNAKLAR["kpss"], KAYNAKLAR["mackinnon96"]])),
                ],
                bicim=bicim,
                dil=dil,
            )
            if dosya is not None:
                r.kaydet(dosya)
            return r
        finally:
            DIL.reset(token)


def duraganlik(
    veri: pd.DataFrame,
    sutunlar: list[str] | None = None,
    deterministik: str = "sabit",
    log: bool | list[str] = False,
    bilgi_olcutu: str = "aic",
    etiketler: dict | None = None,
) -> DuraganlikSonucu:
    """Her seri için düzeyde ve birinci farkta ADF + KPSS; bütünleşme derecesi (I(0) / I(1)) kararı ve tez tablosu.

    log : True ise tüm serilerin, liste verilirse yalnızca o sütunların doğal logaritması alınır (ad 'ln(x)' olur).
    """
    det_kod = _det(deterministik)
    kpss_det = deterministik if det_kod != "n" else "sabit"
    if sutunlar is None:
        sutunlar = [c for c in veri.columns if pd.api.types.is_numeric_dtype(veri[c])]
    sonuc = {}
    for c in sutunlar:
        if c not in veri.columns:
            raise _hata(f"Veride olmayan sütun: {c!r}. Mevcut sütunlar: {list(veri.columns)}")
        s = veri[c].astype(float)
        ad = c
        if log is True or (isinstance(log, (list, tuple)) and c in log):
            if (s <= 0).any():
                raise _hata(f"'{c}' sıfır ya da negatif değer içeriyor; logaritması alınamaz.")
            s = np.log(s)
            ad = f"ln({c})"
        fark = s.diff()
        sonuc[ad] = {
            "duzey": {"adf": adf(s, deterministik, bilgi_olcutu=bilgi_olcutu, ad=ad), "kpss": kpss(s, kpss_det, ad=ad)},
            "fark": {"adf": adf(fark, deterministik, bilgi_olcutu=bilgi_olcutu, ad=f"Δ{ad}"), "kpss": kpss(fark, kpss_det, ad=f"Δ{ad}")},
        }
    et = {**veri.attrs.get("etiketler", {}), **(etiketler or {})}
    for c in list(et):
        et.setdefault(f"ln({c})", f"ln({et[c]})")
    return DuraganlikSonucu(sonuc, det_kod, et)


# --------------------------------------------------------------------------- Engle-Granger
@dataclass
class EngleGrangerSonucu:
    uzun_donem: object  # EKKSonuc
    adf: BirimKokSonucu
    N: int
    p_degeri: float
    kritik: dict[str, float]
    esbutunlesik: bool

    def __str__(self) -> str:
        k = " · ".join(f"{a}: {sayi(v, 3)}" for a, v in self.kritik.items())
        karar = (
            "p < 0,05: H0 reddedildi — değişkenler eşbütünleşik: aralarında uzun dönemli bir denge ilişkisi var."
            if self.esbutunlesik
            else "p ≥ 0,05: H0 reddedilemedi — eşbütünleşme yok: düzey değişkenlerle kurulan regresyon sahte (spurious) olabilir."
        )
        return (
            f"{'✅' if self.esbutunlesik else '⚠️'} Engle-Granger eşbütünleşme testi ({self.N} değişken)\n"
            f"   Kalıntılara ADF: τ = {sayi(self.adf.istatistik, 3)}, {p_cumle(self.p_degeri)} (gecikme = {self.adf.gecikme})   kritik değerler: {k}\n"
            f"   H0: Değişkenler arasında eşbütünleşme yoktur.\n   {karar}"
        )

    __repr__ = __str__

    def rapor(
        self,
        dosya: str | None = None,
        baslik: str | None = None,
        etiketler: dict | None = None,
        tablo_no: int = 1,
        dil: str = "tr",
        bicim: str = "apa",
    ) -> Rapor:
        from .rapor import rapor_ayari

        if baslik is None:
            baslik = "Cointegration analysis" if dil == "en" else "Eşbütünleşme Analizi"
        token = rapor_ayari(dil, bicim)
        try:
            m = self.uzun_donem
            ad = _Adlar({**m._etiketler, **(etiketler or {})})
            uzun = "; ".join(f"{ad.satir(t)} = {sayi(m.katsayilar[t.sutun], 4)}" for t in m._terimler if t.tur != "sabit")
            istat = f"{sayi(self.adf.istatistik, 3)}"
            kritik5 = sayi(self.kritik["%5"], 3)
            if dil == "en":
                karar = (
                    "The null of no cointegration is rejected at the 5% level. The variables are cointegrated and share a long-run equilibrium. "
                    f"Long-run coefficients: {uzun}. Their standard errors do not follow standard distributions, so significance should not be read from t statistics."
                    if self.esbutunlesik
                    else "The null of no cointegration is not rejected at the 5% level. There is no evidence of a long-run equilibrium, and a regression in levels may be spurious."
                )
                c = (
                    "The long-run relationship was tested with the two-step Engle and Granger (1987) procedure. The levels equation was estimated by OLS "
                    f"(R² = {sayi(m.r2, 3)}), and a constant-free ADF test was applied to its residuals. "
                    f"The residual ADF statistic is {istat} ({_p(self.p_degeri)}; 5% critical value {kritik5}). {karar}"
                )
                baslik_tablo = "Engle-Granger cointegration test"
                sutunlar = ["Test", "Statistic", "p", "1% critical", "5% critical", "10% critical"]
                ad_test = "ADF on residuals (τ)"
                notu = (
                    f"H₀: no cointegration. Number of variables: {self.N}; observations: {m.n}. Residual ADF lag: {self.adf.gecikme} ({self.adf.detay['secim']}). "
                    "Critical values follow MacKinnon (2010); the p-value follows MacKinnon (1996)."
                )
            else:
                c = (
                    "Değişkenler arasındaki uzun dönemli ilişki Engle ve Granger (1987) iki aşamalı yöntemiyle sınanmıştır. İlk aşamada uzun dönem denklemi EKK ile tahmin edilmiş "
                    f"(R² = {sayi(m.r2, 3)}), ikinci aşamada bu denklemin kalıntılarına sabitsiz ADF testi uygulanmıştır. "
                    f"Kalıntılara uygulanan ADF test istatistiği {istat} olarak hesaplanmıştır ({_p(self.p_degeri)}; %5 kritik değeri {kritik5}). "
                    + (
                        "Bu sonuca göre eşbütünleşme olmadığını ifade eden temel hipotez %5 düzeyinde reddedilmektedir; değişkenler eşbütünleşiktir ve aralarında uzun dönemli bir denge ilişkisi bulunmaktadır. "
                        f"Uzun dönem katsayıları: {uzun}. Bu katsayıların standart hataları standart dağılımlara uymadığından anlamlılıkları t-istatistikleriyle değerlendirilmemelidir."
                        if self.esbutunlesik
                        else "Bu sonuca göre eşbütünleşme olmadığını ifade eden temel hipotez %5 düzeyinde reddedilememektedir. Değişkenler arasında uzun dönemli bir denge ilişkisine dair kanıt bulunamamıştır; düzey değişkenlerle tahmin edilen regresyon sahte regresyon olabilir."
                    )
                )
                baslik_tablo = "Engle-Granger eşbütünleşme testi sonuçları"
                sutunlar = ["Test", "İstatistik", "p", "Kritik %1", "Kritik %5", "Kritik %10"]
                ad_test = "Kalıntılara ADF (τ)"
                notu = (
                    f"H₀: eşbütünleşme yoktur. Değişken sayısı: {self.N}; gözlem sayısı: {m.n}. Kalıntı ADF gecikme sayısı: {self.adf.gecikme} ({self.adf.detay['secim']}). "
                    "Kritik değerler MacKinnon (2010), p-değeri MacKinnon (1996)."
                )
            t = Tablo(
                tablo_no,
                baslik_tablo,
                sutunlar,
                [[ad_test, sayi(self.adf.istatistik, 3), _p_hucre(self.p_degeri)] + [sayi(v, 3) for v in self.kritik.values()]],
                notu,
            )
            r = Rapor(
                baslik,
                [
                    ("paragraf", c),
                    ("tablo", t),
                    ("baslik", l("Kaynakça", "References")),
                    ("liste", sorted([KAYNAKLAR["eg"], KAYNAKLAR["mackinnon96"], KAYNAKLAR["mackinnon2010"]])),
                ],
                bicim=bicim,
                dil=dil,
            )
            if dosya is not None:
                r.kaydet(dosya)
            return r
        finally:
            DIL.reset(token)

    def hata_duzeltme(
        self,
        gecikme: int | None = None,
        bilgi_olcutu: str = "aic",
        en_fazla_gecikme: int | None = None,
        es_zamanli: bool = True,
    ):
        """Uzun dönem kalıntısıyla Engle-Granger hata düzeltme modelini tahmin eder."""
        return _hata_duzeltme_kur(self, gecikme, bilgi_olcutu, en_fazla_gecikme, es_zamanli)


def engle_granger(formul: str, veri: pd.DataFrame, gecikme: int | None = None, bilgi_olcutu: str = "aic") -> EngleGrangerSonucu:
    """Engle-Granger iki aşamalı eşbütünleşme testi (sabitli uzun dönem denklemi). H0: eşbütünleşme yoktur.

    >>> ek.engle_granger("np.log(ihracat) ~ np.log(reel_kur) + dunya_buyumesi", veri)
    """
    from .regresyon import ekk

    m = ekk(formul, veri)
    if "Intercept" not in m.X.columns:
        raise _hata("Engle-Granger testi için uzun dönem denkleminde sabit terim olmalı.")
    degiskenler = [t for t in m._terimler if t.tur != "sabit"]
    if any(t.tur not in ("surekli", "log") for t in degiskenler):
        raise _hata(
            "Engle-Granger uzun dönem denkleminde yalnızca sürekli değişkenler ya da logaritmaları kullanılmalı (kukla, kare, etkileşim olmamalı)."
        )
    N = len(degiskenler) + 1
    if N > 6:
        raise _hata("MacKinnon tabloları en fazla 6 değişken içindir.")
    if m.r2 > 1 - 1e-10:
        raise _hata("Değişkenler (neredeyse) kusursuz doğrusal ilişkili; eşbütünleşme testi güvenilir değil.")
    a = adf(m.kalintilar.values, "yok", gecikme=gecikme, bilgi_olcutu=bilgi_olcutu, ad="kalıntı")
    p = mackinnon_p(a.istatistik, "c", N)
    return EngleGrangerSonucu(m, a, N, p, mackinnon_kritik("c", N, m.n - 1), p < 0.05)


# --------------------------------------------------------------------------- Granger nedensellik
def _olcut(bilgi_olcutu: str) -> str:
    if bilgi_olcutu not in ("aic", "bic"):
        raise _hata("bilgi_olcutu 'aic' ya da 'bic' olmalı.")
    return bilgi_olcutu


def _bilgi(ssr: float, T: int, k: int, olcut: str) -> float:
    ll = -T / 2 * (np.log(2 * np.pi) + np.log(ssr / T) + 1)
    return float(-2 * ll + (2 * k if olcut == "aic" else np.log(T) * k))


def _zaman_boslugu_var(index: pd.Index) -> bool:
    """Sayısal ya da tarih indeksinde eşit olmayan adım varsa True (ortada düşmüş gözlem)."""
    if len(index) < 3:
        return False
    try:
        if isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)):
            vals = np.asarray(index.asi8, dtype=float)
        elif pd.api.types.is_numeric_dtype(index):
            vals = index.to_numpy(dtype=float)
        else:
            return False
    except (TypeError, ValueError, AttributeError):
        return False
    if not np.isfinite(vals).all():
        return False
    adim = np.diff(vals)
    return not bool(np.all(np.isclose(adim, adim[0])) and adim[0] != 0)


def _kenar_eksik_kes(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sonlu = np.isfinite(y) & np.isfinite(x)
    if not sonlu.any():
        raise _hata("Serilerde kullanılabilir gözlem yok.")
    ilk = int(np.argmax(sonlu))
    son = len(sonlu) - int(np.argmax(sonlu[::-1]))
    if not bool(sonlu[ilk:son].all()):
        raise _hata("Serilerin ortasında eksik gözlem var; gecikmeler bozulur. Eksikleri doldur ya da kesintisiz bir aralık seç.")
    return y[ilk:son], x[ilk:son]


def _seri_cifti(y, x) -> tuple[np.ndarray, np.ndarray, str, str]:
    """İki seriyi hizalar. Series verilirse ortak indeks kullanılır; diziler aynı uzunlukta olmalıdır."""
    if isinstance(y, pd.Series) and isinstance(x, pd.Series):
        y_ad = str(y.name) if y.name is not None else "y"
        x_ad = str(x.name) if x.name is not None else "x"
        if y_ad == x_ad:
            raise _hata("İki seri aynı ada sahip. Granger testi için serilere farklı ad ver.")
        df = pd.concat([y.rename("_y"), x.rename("_x")], axis=1, join="inner")
        if df.empty:
            raise _hata("İki serinin ortak gözlemi yok.")
        if _zaman_boslugu_var(df.index):
            raise _hata("Seriler hizalanınca zaman boşluğu oluşuyor. Eksik dönemleri doldur ya da kesintisiz bir aralık seç.")
        yy, xx = _kenar_eksik_kes(df["_y"].to_numpy(dtype=float), df["_x"].to_numpy(dtype=float))
    else:
        y_ad = str(y.name) if isinstance(y, pd.Series) and y.name is not None else "y"
        x_ad = str(x.name) if isinstance(x, pd.Series) and x.name is not None else "x"
        if y_ad == x_ad:
            raise _hata("İki seri aynı ada sahip. Granger testi için serilere farklı ad ver.")
        yy = np.asarray(y, dtype=float).ravel()
        xx = np.asarray(x, dtype=float).ravel()
        if yy.ndim != 1 or yy.shape != xx.shape:
            raise _hata("Granger testi için aynı uzunlukta iki seri ver.")
        yy, xx = _kenar_eksik_kes(yy, xx)
    if len(yy) < 12:
        raise _hata(f"Granger testi için en az 12 gözlem gerekli ({len(yy)} var).")
    if np.ptp(yy) == 0 or np.ptp(xx) == 0:
        raise _hata("Sabit seri ile Granger testi yapılamaz.")
    return yy, xx, y_ad, x_ad


def _granger_ssr(sol: np.ndarray, sag: np.ndarray | None, p: int, bas: int) -> tuple[float, int, int]:
    """sol_t = sabit + sol gecikmeleri [+ sag gecikmeleri]. Döner: (hata kareleri toplamı, katsayı sayısı, gözlem)."""
    t = np.arange(bas, len(sol))
    cols = [np.ones(len(t))]
    for i in range(1, p + 1):
        cols.append(sol[t - i])
        if sag is not None:
            cols.append(sag[t - i])
    Z = np.column_stack(cols)
    _, e, rank = ekk_coz(Z, sol[t])
    if rank < Z.shape[1]:
        raise _hata("Gecikmeler doğrusal bağımlı; daha küçük bir gecikme dene.")
    ssr = float(e @ e)
    if ssr <= 0:
        raise _hata("Model hatasız uyuyor; Granger istatistiği hesaplanamıyor.")
    return ssr, Z.shape[1], len(t)


def _granger_olc(y: np.ndarray, x: np.ndarray, p: int, bas: int, olcut: str, cift: bool) -> float:
    ssr, k, T = _granger_ssr(y, x, p, bas)
    deger = _bilgi(ssr, T, k, olcut)
    if cift:
        ssr2, k2, T2 = _granger_ssr(x, y, p, bas)
        deger += _bilgi(ssr2, T2, k2, olcut)
    return deger


def _granger_pmax(n: int) -> int:
    """Payda serbestliği en az 5 kalacak şekilde üst sınır; Schwert kuralı ve 8 ile de kırpılır."""
    serbest = int((n - 6) // 3)
    schwert = int(np.floor(12 * (n / 100) ** 0.25))
    pmax = min(8, serbest, schwert)
    if pmax < 1:
        raise _hata(f"Granger testi için seri çok kısa ({n} gözlem).")
    return pmax


def _granger_istatistik(etkilenen: np.ndarray, neden: np.ndarray, p: int) -> dict:
    """SSR'ye dayalı F ve χ². H0: 'neden' gecikmelerinin katsayıları sıfırdır."""
    if p < 1:
        raise _hata("Granger testinde gecikme en az 1 olmalı.")
    k_u = 1 + 2 * p
    if len(etkilenen) - p <= k_u:
        raise _hata(f"Gecikme {p} için gözlem yetersiz ({len(etkilenen) - p} gözlem, {k_u} katsayı).")
    ssr_u, _, T = _granger_ssr(etkilenen, neden, p, p)
    ssr_r, _, _ = _granger_ssr(etkilenen, None, p, p)
    df_payda = T - k_u
    F = ((ssr_r - ssr_u) / p) / (ssr_u / df_payda)
    chi2 = T * (ssr_r - ssr_u) / ssr_u
    return {
        "F": float(F),
        "p": float(stats.f.sf(F, p, df_payda)),
        "df_pay": p,
        "df_payda": df_payda,
        "chi2": float(chi2),
        "chi2_p": float(stats.chi2.sf(chi2, p)),
        "n": T,
    }


@dataclass
class GrangerYon:
    """Tek yön: 'neden', 'etkilenen'in Granger nedeni midir?"""

    neden: str
    etkilenen: str
    gecikme: int
    F: float
    p_degeri: float
    serbestlik: tuple[int, int]
    chi2: float
    chi2_p: float
    n: int

    @property
    def nedensel(self) -> bool:
        return self.p_degeri < 0.05

    def karar(self) -> str:
        iliski = f"'{self.neden}', '{self.etkilenen}' değişkeninin Granger nedeni"
        if self.nedensel:
            return f"p < 0,05: H0 reddedildi — {iliski}dir."
        return f"p ≥ 0,05: H0 reddedilemedi — {iliski} olduğuna dair kanıt yoktur."


@dataclass
class GrangerSonucu:
    """Granger nedensellik. ``yonler[0]`` istenen yöndür (x, y'nin nedeni midir); ``cift=True`` ise ters yön de vardır."""

    yonler: list[GrangerYon]
    gecikme: int
    secim: str

    @property
    def p_degeri(self) -> float:
        return self.yonler[0].p_degeri

    @property
    def nedensel(self) -> bool:
        return self.yonler[0].nedensel

    def __str__(self) -> str:
        satirlar = [
            f"Granger nedensellik testi — gecikme = {self.gecikme} ({self.secim})",
            "   H0: Aday seri, diğerinin Granger nedeni değildir. Sonuç öngörüseldir; yapısal nedensellik değildir.",
        ]
        for yon in self.yonler:
            isaret = "✅" if yon.nedensel else "⚠️"
            satirlar.append(
                f"   {isaret} '{yon.neden}' → '{yon.etkilenen}': "
                f"F({yon.serbestlik[0]}, {yon.serbestlik[1]}) = {sayi(yon.F, 3)}, {p_cumle(yon.p_degeri)} (n = {yon.n})"
            )
            satirlar.append(f"      {yon.karar()}")
        return "\n".join(satirlar)

    __repr__ = __str__

    def _repr_html_(self) -> str:
        return self.rapor().html()

    def rapor(
        self,
        dosya: str | None = None,
        baslik: str | None = None,
        etiketler: dict | None = None,
        tablo_no: int = 1,
        dil: str = "tr",
        bicim: str = "apa",
    ) -> Rapor:
        from .rapor import rapor_ayari

        if baslik is None:
            baslik = "Granger causality test" if dil == "en" else "Granger Nedensellik Testi"
        token = rapor_ayari(dil, bicim)
        try:
            ad = _Adlar(etiketler)
            en = dil == "en"
            if self.secim == "sabit":
                gecikme_cumle = f"The lag length is {self.gecikme}." if en else f"Gecikme sayısı {self.gecikme} olarak belirlenmiştir."
            elif len(self.yonler) > 1:
                gecikme_cumle = (
                    f"The lag length {self.gecikme} was selected by summing {self.secim} across both equations on a common sample."
                    if en
                    else f"Gecikme sayısı, her iki denklemin {self.secim} değerleri toplanarak ortak örneklemde {self.gecikme} olarak seçilmiştir."
                )
            else:
                gecikme_cumle = (
                    f"The lag length {self.gecikme} was selected by {self.secim} for the unrestricted equation on a common sample."
                    if en
                    else f"Gecikme sayısı, kısıtsız denklemin {self.secim} değerine göre ortak örneklemde {self.gecikme} olarak seçilmiştir."
                )
            if en:
                cumle = [
                    "The predictive relationship was examined with the Granger (1969) causality test. "
                    "The null hypothesis is that lags of the candidate series add nothing to the forecast of the other series. "
                    + gecikme_cumle
                ]
            else:
                cumle = [
                    "Değişkenler arasındaki öngörüsel ilişki Granger (1969) nedensellik testiyle incelenmiştir. "
                    "Sınanan temel hipotez, aday serinin gecikmelerinin diğer seriyi öngörmeye ek katkı sağlamadığıdır. " + gecikme_cumle
                ]
            for yon in self.yonler:
                if en:
                    sonuc = (
                        f"Thus {ad.metin(yon.neden)} Granger-causes {ad.metin(yon.etkilenen)}."
                        if yon.nedensel
                        else f"Thus there is no evidence that {ad.metin(yon.neden)} Granger-causes {ad.metin(yon.etkilenen)}."
                    )
                    cumle.append(
                        f"The hypothesis that {ad.metin(yon.neden)} does not Granger-cause {ad.metin(yon.etkilenen)} was tested with "
                        f"F({yon.serbestlik[0]}, {yon.serbestlik[1]}) = {sayi(yon.F, 2)} ({_p(yon.p_degeri)}) and is "
                        f"{'rejected' if yon.nedensel else 'not rejected'} at the 5% level. {sonuc}"
                    )
                else:
                    karar = "reddedilmektedir" if yon.nedensel else "reddedilememektedir"
                    sonuc = (
                        f"Buna göre {ad.metin(yon.neden)} değişkeni, {ad.metin(yon.etkilenen)} değişkeninin Granger nedenidir."
                        if yon.nedensel
                        else (
                            f"Buna göre {ad.metin(yon.neden)} değişkeninin {ad.metin(yon.etkilenen)} değişkeninin "
                            "Granger nedeni olduğuna dair kanıt bulunamamıştır."
                        )
                    )
                    cumle.append(
                        f"{ad.metin(yon.neden)} değişkeninin {ad.metin(yon.etkilenen)} değişkeninin Granger nedeni olmadığı hipotezi "
                        f"F({yon.serbestlik[0]}, {yon.serbestlik[1]}) = {sayi(yon.F, 2)} ({_p(yon.p_degeri)}) istatistiğiyle sınanmış "
                        f"ve %5 düzeyinde {karar}. {sonuc}"
                    )
            cumle.append(
                "The test does not establish structural causality; it measures only the contribution of lagged values to the forecast."
                if en
                else "Bu test yapısal bir neden-sonuç ilişkisi kurmaz; yalnızca gecikmeli öngörü katkısını ölçer. Sonuç bu çerçevede yorumlanmalıdır."
            )
            satirlar = [
                [
                    (
                        f"{ad.yalin(yon.neden)} does not Granger-cause {ad.yalin(yon.etkilenen)}"
                        if en
                        else f"{ad.yalin(yon.neden)} değişkeni, {ad.yalin(yon.etkilenen)} değişkeninin Granger nedeni değildir"
                    ),
                    str(yon.gecikme),
                    f"{sayi(yon.F, 3)} ({yon.serbestlik[0]}, {yon.serbestlik[1]})",
                    _p_hucre(yon.p_degeri),
                    (l("Reddedildi", "Rejected") if yon.nedensel else l("Reddedilemedi", "Not rejected")),
                ]
                for yon in self.yonler
            ]
            t = Tablo(
                tablo_no,
                l("Granger nedensellik testi sonuçları", "Granger causality test results"),
                [l("Temel hipotez", "Null hypothesis"), l("Gecikme", "Lag"), "F (sd)", "p", l("Karar", "Decision")],
                satirlar,
                l(
                    "H₀, aday serinin gecikmeli değerlerinin diğer seriyi öngörmeye katkı sağlamadığıdır. "
                    "F istatistiği kısıtlı ve kısıtsız denklemlerin hata kareleri toplamından hesaplanır. "
                    "Test öngörüseldir; yapısal nedensellik anlamına gelmez.",
                    "H₀ states that lags of the candidate series do not help forecast the other series. "
                    "The F statistic is computed from the residual sums of squares of the restricted and unrestricted equations. "
                    "The test is predictive and does not establish structural causality.",
                ),
            )
            r = Rapor(
                baslik,
                [
                    ("paragraf", " ".join(cumle)),
                    ("tablo", t),
                    ("baslik", l("Kaynakça", "References")),
                    ("liste", [KAYNAKLAR["granger"]]),
                ],
                bicim=bicim,
                dil=dil,
            )
            if dosya is not None:
                r.kaydet(dosya)
            return r
        finally:
            DIL.reset(token)


def granger(
    y,
    x,
    gecikme: int | None = None,
    en_fazla_gecikme: int | None = None,
    bilgi_olcutu: str = "aic",
    cift: bool = True,
) -> GrangerSonucu:
    """x, y'nin Granger nedeni midir? H0: değildir.

    y sonuç (etkilenen) seri, x neden adayıdır. ``cift=True`` ise ters yön de aynı gecikmeyle sınanır;
    ``sonuc.p_degeri`` her zaman istenen yöne (x → y) aittir.

    gecikme verilmezse adaylar, en uzun gecikmenin örnekleminde AIC (ya da BIC) ile karşılaştırılır;
    raporlanan istatistik seçilen gecikmenin kendi örnekleminde hesaplanır. Sabit bir gecikmedeki
    F ve χ², statsmodels ``grangercausalitytests`` sonuçlarıyla aynıdır.

    >>> ek.granger(veri["ihracat"], veri["reel_kur"], gecikme=2)
    """
    olcut = _olcut(bilgi_olcutu)
    yy, xx, y_ad, x_ad = _seri_cifti(y, x)
    if gecikme is not None:
        if gecikme < 1:
            raise _hata("Granger testinde gecikme en az 1 olmalı.")
        p, secim = int(gecikme), "sabit"
    else:
        if en_fazla_gecikme is None:
            pmax = _granger_pmax(len(yy))
        else:
            if en_fazla_gecikme < 1:
                raise _hata("en_fazla_gecikme en az 1 olmalı.")
            if len(yy) - 3 * en_fazla_gecikme - 1 < 1:
                raise _hata("en_fazla_gecikme bu seri için çok büyük.")
            pmax = int(en_fazla_gecikme)
        p = min((_granger_olc(yy, xx, q, pmax, olcut, cift), q) for q in range(1, pmax + 1))[1]
        secim = olcut.upper()

    def yon(etkilenen: np.ndarray, neden: np.ndarray, etkilenen_ad: str, neden_ad: str) -> GrangerYon:
        s = _granger_istatistik(etkilenen, neden, p)
        return GrangerYon(neden_ad, etkilenen_ad, p, s["F"], s["p"], (s["df_pay"], s["df_payda"]), s["chi2"], s["chi2_p"], s["n"])

    yonler = [yon(yy, xx, y_ad, x_ad)]
    if cift:
        yonler.append(yon(xx, yy, x_ad, y_ad))
    return GrangerSonucu(yonler, p, secim)


# --------------------------------------------------------------------------- hata düzeltme modeli
def _k_ecm(p: int, nx: int, es_zamanli: bool) -> int:
    return 2 + p + nx * (p + (1 if es_zamanli else 0))


def _ecm_veri(
    y: np.ndarray,
    X: np.ndarray,
    ect: np.ndarray,
    p: int,
    es_zamanli: bool,
    y_ad: str,
    x_adlar: list[str],
    bas: int,
) -> tuple[pd.DataFrame, list[str], dict[str, str], str]:
    """Δy_t ~ ECT_{t-1} + Δy gecikmeleri + Δx farkları. Satır sırası zaman sırasıdır."""
    if p < 0:
        raise _hata("Gecikme negatif olamaz.")
    if bas < p + 1:
        raise _hata("İç hata: hata düzeltme örneklemi gecikmeden kısa.")
    t = np.arange(bas, len(y))
    parcalar = ["ect_l1"]
    gorunen = {"ect_l1": "ECT(-1)"}
    veri = pd.DataFrame({"d_y": y[t] - y[t - 1], "ect_l1": ect[t - 1]})
    for i in range(1, p + 1):
        ad = f"d_y_l{i}"
        veri[ad] = y[t - i] - y[t - i - 1]
        parcalar.append(ad)
        gorunen[ad] = f"Δ{y_ad}(-{i})"
    for j, x_ad in enumerate(x_adlar):
        for lag in range(0 if es_zamanli else 1, p + 1):
            ad = f"d_x{j}_l{lag}"
            veri[ad] = X[t - lag, j] - X[t - lag - 1, j]
            parcalar.append(ad)
            gorunen[ad] = f"Δ{x_ad}" if lag == 0 else f"Δ{x_ad}(-{lag})"
    k = 1 + len(parcalar)
    if len(t) <= k:
        raise _hata(f"Gecikme {p} için gözlem yetersiz ({len(t)} gözlem, {k} katsayı). Daha küçük bir gecikme seç.")
    return veri, parcalar, gorunen, f"Δ{y_ad}"


@dataclass
class HataDuzeltmeSonucu:
    """Engle-Granger hata düzeltme modeli. Uzun dönem ``.esbutunlesme`` içindedir; kısa dönem ``.kisa_donem`` bir EKK sonucudur."""

    esbutunlesme: EngleGrangerSonucu
    kisa_donem: object
    gorunen: dict
    bagimli_fark: str
    gecikme: int
    es_zamanli: bool
    secim: str
    y_ad: str
    x_adlar: list

    @property
    def uzun_donem(self):
        return self.esbutunlesme.uzun_donem

    @property
    def lamda(self) -> float:
        return float(self.kisa_donem.katsayilar["ect_l1"])

    @property
    def yari_omur(self) -> float | None:
        """Dengesizliğin yarısının kapandığı dönem sayısı. Yalnızca -1 < λ < 0 ise tanımlıdır."""
        if -1 < self.lamda < 0:
            return float(np.log(0.5) / np.log(1 + self.lamda))
        return None

    def _ect_cumle(self) -> str:
        p = float(self.kisa_donem.p_degerleri["ect_l1"])
        sh = float(self.kisa_donem.standart_hatalar["ect_l1"])
        if DIL.get() == "en":
            temel = f"The error-correction coefficient is {sayi(self.lamda, 4)} ({_p(p)}; SE = {sayi(sh, 4)}). "
            if not self.esbutunlesme.esbutunlesik:
                return temel + "Because the series are not cointegrated, this coefficient cannot be interpreted as a speed of adjustment."
            if self.lamda < 0 and p < 0.05:
                if self.yari_omur is None:
                    return (
                        temel
                        + "The coefficient is negative and significant at the 5% level; the disequilibrium more than closes in the next period."
                    )
                return temel + (
                    f"The coefficient is negative and significant at the 5% level. About {yuzde(abs(self.lamda) * 100, 1)} of a period's disequilibrium "
                    "closes in the next period. "
                    f"At this speed, half of the disequilibrium closes in about {sayi(self.yari_omur, 2)} periods."
                )
            if self.lamda < 0:
                return (
                    temel + "The sign is consistent with adjustment toward equilibrium, but the effect is not significant at the 5% level."
                )
            return temel + "The coefficient is positive, so an adjustment-toward-equilibrium mechanism is not supported."
        temel = f"Hata düzeltme katsayısı {sayi(self.lamda, 4)} olarak tahmin edilmiştir ({_p(p)}; SH = {sayi(sh, 4)}). "
        if not self.esbutunlesme.esbutunlesik:
            return temel + "Eşbütünleşme bulunamadığından bu katsayı dengeye dönüş hızı olarak yorumlanamaz."
        if self.lamda < 0 and p < 0.05:
            if self.yari_omur is None:
                return temel + "Katsayı negatif ve %5 düzeyinde anlamlıdır; dengesizlik bir sonraki dönemde fazlasıyla kapanmaktadır."
            return temel + (
                f"Katsayı negatif ve %5 düzeyinde anlamlıdır; bir dönemdeki dengesizliğin yaklaşık {yuzde(abs(self.lamda) * 100, 1)} kadarı "
                "bir sonraki dönemde kapanmaktadır. "
                f"Bu hızla dengesizliğin yarısı yaklaşık {sayi(self.yari_omur, 2)} dönemde kapanmaktadır."
            )
        if self.lamda < 0:
            return temel + "Katsayının işareti dengeye dönüşle uyumludur; ancak etki %5 düzeyinde anlamlı değildir."
        return temel + "Katsayı pozitiftir; dengeye dönüş mekanizması desteklenmemektedir."

    def __str__(self) -> str:
        m = self.kisa_donem
        g = 78
        satirlar = [
            "═" * g,
            f" Hata düzeltme modeli — bağımlı değişken: {self.bagimli_fark}",
            "═" * g,
            f" Gözlem sayısı: {m.n:<12} R²: {sayi(m.r2, 4):<12} Gecikme: {self.gecikme} ({self.secim})",
            f" Eşbütünleşme: {'var' if self.esbutunlesme.esbutunlesik else 'yok'} ({p_cumle(self.esbutunlesme.p_degeri)})",
            f" Standart hatalar: {m.kovaryans_aciklama}",
            "─" * g,
            f" {'Değişken':<28}{'Katsayı':>12}{'Std. hata':>12}{'t':>9}{'p':>10}",
            "─" * g,
        ]
        for ad in m.katsayilar.index:
            gorunen = "Sabit" if ad == "Intercept" else self.gorunen[ad]
            if len(gorunen) > 27:
                gorunen = gorunen[:26] + "…"
            pv = m.p_degerleri[ad]
            satirlar.append(
                f" {gorunen:<28}{sayi(m.katsayilar[ad], 4):>12}{sayi(m.standart_hatalar[ad], 4):>12}"
                f"{sayi(m.t_degerleri[ad], 2):>9}{p_yaz(pv):>10} {yildiz(pv)}"
            )
        satirlar += ["─" * g, " " + self._ect_cumle(), "═" * g]
        return "\n".join(satirlar)

    __repr__ = __str__

    def _repr_html_(self) -> str:
        return self.rapor().html()

    def rapor(
        self,
        dosya: str | None = None,
        baslik: str | None = None,
        etiketler: dict | None = None,
        tablo_no: int = 1,
        dil: str = "tr",
        bicim: str = "apa",
    ) -> Rapor:
        from .rapor import rapor_ayari

        if baslik is None:
            baslik = "Error-correction model" if dil == "en" else "Hata Düzeltme Modeli"
        token = rapor_ayari(dil, bicim)
        try:
            m = self.kisa_donem
            eg = self.esbutunlesme
            uzun = self.uzun_donem
            ad = _Adlar({**getattr(uzun, "_etiketler", {}), **(etiketler or {})})
            xs = [t for t in uzun._terimler if t.tur != "sabit"]
            uzun_metin = "; ".join(f"{ad.satir(t)} = {sayi(uzun.katsayilar[t.sutun], 4)}" for t in xs)
            en = dil == "en"
            if self.gecikme == 0:
                gecikme_cumle = (
                    "No lags of the short-run differences were included." if en else "Kısa dönem farklarının gecikmesi modele alınmamıştır."
                )
            elif self.secim == "sabit":
                gecikme_cumle = (
                    f"The short-run lag length is {self.gecikme}." if en else f"Kısa dönem gecikmesi {self.gecikme} olarak belirlenmiştir."
                )
            else:
                gecikme_cumle = (
                    f"The short-run lag length {self.gecikme} was selected by {self.secim} on a common sample."
                    if en
                    else f"Kısa dönem gecikmesi, ortak örneklemde {self.secim} ölçütüyle {self.gecikme} olarak seçilmiştir."
                )
            if en:
                es = (
                    "The variables are therefore cointegrated. "
                    if eg.esbutunlesik
                    else "The cointegration hypothesis is not supported at the 5% level, so the error-correction coefficient cannot be interpreted as a speed of adjustment. "
                )
                fark = ", the contemporaneous differences of the regressors" if self.es_zamanli else ""
                c = (
                    "Short-run adjustment was estimated with the Engle and Granger (1987) error-correction model. "
                    f"In the first step the levels equation was estimated by OLS (R² = {sayi(uzun.r2, 3)}; {uzun_metin}). "
                    "These coefficients do not follow a standard t distribution, so their significance was not assessed with t statistics; "
                    "the residuals were passed to the second step as the error-correction term. "
                    f"In the second step the first difference of {ad.metin(self.y_ad)} is explained by one lag of the error-correction term{fark} "
                    f"and the short-run differences. {gecikme_cumle} "
                    f"The cointegration test statistic is {sayi(eg.adf.istatistik, 3)} "
                    f"({_p(eg.p_degeri)}; 5% critical value {sayi(eg.kritik['%5'], 3)}). {es}{self._ect_cumle()}"
                )
            else:
                es = (
                    "Bu sonuca göre değişkenler eşbütünleşiktir. "
                    if eg.esbutunlesik
                    else "Eşbütünleşme hipotezi %5 düzeyinde desteklenmemiştir; hata düzeltme katsayısı dengeye dönüş hızı olarak yorumlanamaz. "
                )
                fark = ", açıklayıcı değişkenlerin eşzamanlı farkı" if self.es_zamanli else ""
                c = (
                    "Kısa dönem uyarlanması Engle ve Granger (1987) hata düzeltme modeliyle tahmin edilmiştir. "
                    f"Birinci aşamada düzey denklemi EKK ile tahmin edilmiştir (R² = {sayi(uzun.r2, 3)}; {uzun_metin}). "
                    "Bu katsayılar standart t dağılımına uymadığından anlamlılıkları t-istatistikleriyle değerlendirilmemiştir; "
                    "kalıntılar hata düzeltme terimi olarak ikinci aşamaya aktarılmıştır. "
                    f"İkinci aşamada {ad.metin(self.y_ad)} değişkeninin birinci farkı, hata düzeltme teriminin bir gecikmesi{fark} "
                    f"ve kısa dönem farklarıyla açıklanmıştır. {gecikme_cumle} "
                    f"Eşbütünleşme testi istatistiği {sayi(eg.adf.istatistik, 3)} olarak hesaplanmıştır "
                    f"({_p(eg.p_degeri)}; %5 kritik değeri {sayi(eg.kritik['%5'], 3)}). {es}{self._ect_cumle()}"
                )
            satirlar = []
            for sutun in m.katsayilar.index:
                gorunen = l("Sabit", "Constant") if sutun == "Intercept" else self.gorunen[sutun]
                pv = float(m.p_degerleri[sutun])
                satirlar.append(
                    [
                        gorunen,
                        sayi(m.katsayilar[sutun], 4),
                        sayi(m.standart_hatalar[sutun], 4),
                        sayi(m.t_degerleri[sutun], 2),
                        _p_hucre(pv) + yildiz(pv),
                    ]
                )
            if en:
                notu = (
                    f"Dependent variable: {self.bagimli_fark}. ECT(-1) is one lag of the long-run residual. "
                    f"Lag length: {self.gecikme} ({self.secim}). "
                    + (
                        "Contemporaneous differences of the regressors are included. "
                        if self.es_zamanli
                        else "Contemporaneous differences are not included. "
                    )
                    + "Standard errors are classical OLS. *** p < 0.01; ** p < 0.05; * p < 0.10."
                )
                t = Tablo(tablo_no, "Error-correction model estimates", ["Variable", "Coefficient", "Std. error", "t", "p"], satirlar, notu)
            else:
                notu = (
                    f"Bağımlı değişken: {self.bagimli_fark}. ECT(-1), uzun dönem kalıntısının bir gecikmesidir. "
                    f"Gecikme: {self.gecikme} ({self.secim}). "
                    + (
                        "Açıklayıcı değişkenlerin eşzamanlı farkı denkleme dahildir. "
                        if self.es_zamanli
                        else "Eşzamanlı fark dahil edilmemiştir. "
                    )
                    + "Standart hatalar klasik EKK. *** p < 0,01; ** p < 0,05; * p < 0,10."
                )
                t = Tablo(tablo_no, "Hata düzeltme modeli tahmin sonuçları", ["Değişken", "Katsayı", "Std. hata", "t", "p"], satirlar, notu)
            r = Rapor(
                baslik,
                [
                    ("paragraf", c),
                    ("tablo", t),
                    ("baslik", l("Kaynakça", "References")),
                    ("liste", sorted([KAYNAKLAR["eg"], KAYNAKLAR["mackinnon96"], KAYNAKLAR["mackinnon2010"]])),
                ],
                bicim=bicim,
                dil=dil,
            )
            if dosya is not None:
                r.kaydet(dosya)
            return r
        finally:
            DIL.reset(token)


def _hata_duzeltme_kur(
    eg: EngleGrangerSonucu,
    gecikme: int | None,
    bilgi_olcutu: str,
    en_fazla_gecikme: int | None,
    es_zamanli: bool,
) -> HataDuzeltmeSonucu:
    from .regresyon import bagimli_gorunumu, ekk, gorunen_ad

    if not isinstance(es_zamanli, bool):
        raise _hata("es_zamanli True ya da False olmalı.")
    olcut = _olcut(bilgi_olcutu)
    m = eg.uzun_donem
    if _zaman_boslugu_var(m.y.index):
        raise _hata("Uzun dönem örnekleminde zaman boşluğu var. Hata düzeltme modeli için kesintisiz bir seri kullan.")
    y = m.y.to_numpy(dtype=float)
    ect = m.kalintilar.to_numpy(dtype=float)
    xs = m.X.drop(columns=["Intercept"])
    X = xs.to_numpy(dtype=float)
    y_ad = bagimli_gorunumu(m.bagimli)
    x_adlar = [gorunen_ad(m, c) for c in xs.columns]
    n, nx = len(y), X.shape[1]

    def aday(p: int, bas: int):
        veri, parcalar, gorunen, bagimli = _ecm_veri(y, X, ect, p, es_zamanli, y_ad, x_adlar, bas)
        kisa = ekk("d_y ~ " + " + ".join(parcalar), veri)
        return kisa, gorunen, bagimli

    if gecikme is not None:
        p, secim = int(gecikme), "sabit"
    else:
        if en_fazla_gecikme is not None:
            if en_fazla_gecikme < 0:
                raise _hata("en_fazla_gecikme negatif olamaz.")
            if n - en_fazla_gecikme - 1 <= _k_ecm(en_fazla_gecikme, nx, es_zamanli):
                raise _hata("en_fazla_gecikme bu seri için çok büyük.")
            pmax = int(en_fazla_gecikme)
        else:
            pmax = 8
            while pmax >= 0 and n - pmax - 1 <= _k_ecm(pmax, nx, es_zamanli) + 4:
                pmax -= 1
            if pmax < 0:
                raise _hata("Hata düzeltme modeli için seri çok kısa.")
        bas = pmax + 1
        secilen = []
        for q in range(0, pmax + 1):
            kisa, _, _ = aday(q, bas)
            secilen.append((kisa.aic if olcut == "aic" else kisa.bic, q))
        p, secim = min(secilen)[1], olcut.upper()
    kisa, gorunen, bagimli = aday(p, p + 1)
    return HataDuzeltmeSonucu(eg, kisa, gorunen, bagimli, p, es_zamanli, secim, y_ad, x_adlar)


def hata_duzeltme(
    formul: str,
    veri: pd.DataFrame,
    gecikme: int | None = None,
    bilgi_olcutu: str = "aic",
    en_fazla_gecikme: int | None = None,
    es_zamanli: bool = True,
) -> HataDuzeltmeSonucu:
    """Engle-Granger iki aşamalı hata düzeltme modeli.

    Uzun dönem: y_t = c + β'x_t + u_t. Kısa dönem:
    Δy_t = α + λ û_{t-1} + Σ_{i=1}^{p} φ_i Δy_{t-i} + Σ_j Σ_{k=k0}^{p} θ_{jk} Δx_{j,t-k} + e_t.
    ``es_zamanli=True`` ise açıklayıcıların cari farkı da girer (k0 = 0); aksi halde k0 = 1.

    Gecikme verilmezse p, ortak örneklemde AIC/BIC ile seçilir ve seçilen model kendi örnekleminde yeniden tahmin edilir.
    Eşbütünleşme yoksa model yine tahmin edilir; λ dengeye dönüş hızı olarak yorumlanmaz.

    >>> ek.hata_duzeltme("np.log(ihracat) ~ np.log(reel_kur)", veri, gecikme=1)
    """
    return _hata_duzeltme_kur(engle_granger(formul, veri), gecikme, bilgi_olcutu, en_fazla_gecikme, es_zamanli)


# --------------------------------------------------------------------------- çok değişkenli Granger
def _var_denklem(Y: np.ndarray, p: int, bas: int, denklem: int, haric: set[int] | None = None) -> tuple[np.ndarray, int, int]:
    """Bir VAR denklemi. haric içindeki değişkenlerin gecikmeleri kısıtta düşülür."""
    t = np.arange(bas, len(Y))
    cols = [np.ones(len(t))]
    for j in range(Y.shape[1]):
        if haric and j in haric:
            continue
        for i in range(1, p + 1):
            cols.append(Y[t - i, j])
    Z = np.column_stack(cols)
    _, e, rank = ekk_coz(Z, Y[t, denklem])
    if rank < Z.shape[1]:
        raise _hata("VAR gecikmeleri doğrusal bağımlı; daha küçük bir gecikme dene.")
    if float(e @ e) <= 0:
        raise _hata("VAR denklemi hatasız uyuyor; Granger istatistiği hesaplanamıyor.")
    return e, Z.shape[1], len(t)


def _var_olcut(Y: np.ndarray, p: int, bas: int) -> float:
    """Lütkepohl AIC: ln|Σ| + 2(pk² + k) / T. Σ, olabilirlik kovaryansıdır."""
    E = np.column_stack([_var_denklem(Y, p, bas, i)[0] for i in range(Y.shape[1])])
    T, k = E.shape
    işaret, ld = np.linalg.slogdet((E.T @ E) / T)
    if işaret <= 0:
        raise _hata("VAR kalıntı kovaryansı tekil; gecikme seçilemiyor.")
    return float(ld + 2 * (p * k * k + k) / T)


@dataclass
class VARGrangerYon:
    neden: str
    etkilenen: str
    gecikme: int
    F: float
    p_degeri: float
    serbestlik: tuple[int, int]
    chi2: float
    chi2_p: float
    n: int

    @property
    def nedensel(self) -> bool:
        return self.p_degeri < 0.05


@dataclass
class VARGrangerSonucu:
    """VAR içinde koşullu Granger (blok dışsallık). Her satır, diğer değişkenler kontrol edildikten sonraki tek yönlü testtir."""

    yonler: list[VARGrangerYon]
    gecikme: int
    secim: str
    degiskenler: list[str]

    def __str__(self) -> str:
        satirlar = [
            f"VAR Granger nedensellik — gecikme = {self.gecikme} ({self.secim}), değişkenler: {', '.join(self.degiskenler)}",
            "   H0: Adayın gecikmeleri, diğer değişkenler kontrol edildikten sonra denkleme katkı sağlamaz.",
        ]
        for yon in self.yonler:
            isaret = "✅" if yon.nedensel else "⚠️"
            satirlar.append(
                f"   {isaret} '{yon.neden}' → '{yon.etkilenen}': "
                f"F({yon.serbestlik[0]}, {yon.serbestlik[1]}) = {sayi(yon.F, 3)}, {p_cumle(yon.p_degeri)}"
            )
        return "\n".join(satirlar)

    __repr__ = __str__

    def _repr_html_(self) -> str:
        return self.rapor().html()

    def rapor(
        self,
        dosya: str | None = None,
        baslik: str | None = None,
        etiketler: dict | None = None,
        tablo_no: int = 1,
        dil: str = "tr",
        bicim: str = "apa",
    ) -> Rapor:
        from ._yardimci import DIL, l

        if dil not in ("tr", "en") or bicim not in ("apa", "izgara"):
            raise _hata("dil 'tr' ya da 'en'; bicim 'apa' ya da 'izgara' olmalı.")
        token = DIL.set(dil)
        try:
            ad = _Adlar(etiketler)
            if self.secim == "sabit":
                gecikme = l(f"Gecikme uzunluğu {self.gecikme} olarak belirlenmiştir.", f"The lag length is {self.gecikme}.")
            else:
                gecikme = l(
                    f"Gecikme uzunluğu, ortak örneklemde {self.secim} ile {self.gecikme} olarak seçilmiştir.",
                    f"The lag length {self.gecikme} was selected by {self.secim} on a common sample.",
                )
            cumle = l(
                "Öngörüsel ilişki, tüm değişkenleri içeren bir VAR içinde Granger (1969) blok dışsallık testiyle incelenmiştir. "
                + gecikme
                + " Her hipotez, aday değişkenin gecikmelerinin ilgili denklemde sıfır olduğunu söyler; diğer değişkenlerin gecikmeleri kontrol edilmiştir. "
                "Test yapısal nedensellik kurmaz.",
                "Predictive relationships were examined with Granger (1969) block-exogeneity tests inside a VAR that includes every listed variable. "
                + gecikme
                + " Each hypothesis sets the lags of the candidate variable to zero in one equation, controlling for the lags of the other variables. "
                "The test does not establish structural causality.",
            )
            satirlar = [
                [
                    l(
                        f"{ad.yalin(y.neden)} değişkeni, {ad.yalin(y.etkilenen)} değişkeninin Granger nedeni değildir",
                        f"{ad.yalin(y.neden)} does not Granger-cause {ad.yalin(y.etkilenen)}",
                    ),
                    str(y.gecikme),
                    f"{sayi(y.F, 3)} ({y.serbestlik[0]}, {y.serbestlik[1]})",
                    _p_hucre(y.p_degeri),
                    l("Reddedildi", "Rejected") if y.nedensel else l("Reddedilemedi", "Not rejected"),
                ]
                for y in self.yonler
            ]
            t = Tablo(
                tablo_no,
                l("VAR Granger nedensellik testi", "VAR Granger causality test"),
                [l("Temel hipotez", "Null hypothesis"), l("Gecikme", "Lag"), "F (sd)", "p", l("Karar", "Decision")],
                satirlar,
                l(
                    "F testi tek denklem EKK kısıtıdır. sd, pay ve payda serbestlik dereceleridir.",
                    "The F test is a single-equation OLS restriction. The degrees of freedom are the numerator and the denominator.",
                ),
            )
            r = Rapor(
                baslik or l("VAR Granger Nedensellik Testi", "VAR Granger Causality Test"),
                [("paragraf", cumle), ("tablo", t), ("baslik", l("Kaynakça", "References")), ("liste", [KAYNAKLAR["granger"]])],
                bicim=bicim,
                dil=dil,
            )
            if dosya is not None:
                r.kaydet(dosya)
            return r
        finally:
            DIL.reset(token)


def _var_matris(veri: pd.DataFrame, sutunlar: list[str]) -> tuple[np.ndarray, list[str]]:
    if len(sutunlar) < 3:
        raise _hata("Çok değişkenli Granger için en az üç seri ver. İki seri için ek.granger kullan.")
    eksik = [c for c in sutunlar if c not in veri.columns]
    if eksik:
        raise _hata(f"Veride olmayan sütun: {eksik[0]!r}.")
    parca = veri.loc[:, sutunlar].apply(pd.to_numeric, errors="coerce")
    if _zaman_boslugu_var(parca.index):
        raise _hata("Serilerde zaman boşluğu var. VAR için kesintisiz bir aralık seç.")
    sonlu = np.isfinite(parca.to_numpy(float)).all(axis=1)
    if not sonlu.any():
        raise _hata("Serilerde kullanılabilir gözlem yok.")
    ilk = int(np.argmax(sonlu))
    son = len(sonlu) - int(np.argmax(sonlu[::-1]))
    if not bool(sonlu[ilk:son].all()):
        raise _hata("Serilerin ortasında eksik gözlem var; gecikmeler bozulur.")
    Y = parca.to_numpy(float)[ilk:son]
    if len(Y) < 15:
        raise _hata(f"VAR Granger testi için en az 15 gözlem gerekli ({len(Y)} var).")
    if np.any(np.ptp(Y, axis=0) == 0):
        raise _hata("Sabit bir seri ile VAR kurulamaz.")
    return Y, list(sutunlar)


def var_granger(
    veri: pd.DataFrame,
    sutunlar: list[str],
    gecikme: int | None = None,
    en_fazla_gecikme: int | None = None,
    bilgi_olcutu: str = "aic",
) -> VARGrangerSonucu:
    """VAR içinde koşullu Granger testi. Her sıralı çift için, diğer gecikmeler kontrol edildikten sonra F testi yapılır.

    Gecikme verilmezse ortak örneklemde Lütkepohl AIC ile seçilir. İstatistik, seçilen gecikmenin kendi örnekleminde
    tek denklem EKK F testidir (EViews blok dışsallık tablosunun F karşılığı).

    >>> ek.var_granger(veri, ["ihracat", "reel_kur", "dunya_buyumesi"], gecikme=2)
    """
    if bilgi_olcutu != "aic":
        raise _hata("Çok değişkenli Granger gecikme seçimi şimdilik yalnız 'aic' kullanır.")
    Y, adlar = _var_matris(veri, sutunlar)
    n, k = Y.shape

    def sigar(p: int) -> bool:
        return n - p > 1 + k * p + 4

    if gecikme is not None:
        if gecikme < 1 or not sigar(int(gecikme)):
            raise _hata("Gecikme en az 1 olmalı ve seriye sığmalı.")
        p, secim = int(gecikme), "sabit"
    else:
        pmax = 4 if en_fazla_gecikme is None else int(en_fazla_gecikme)
        if pmax < 1:
            raise _hata("en_fazla_gecikme en az 1 olmalı.")
        while pmax > 1 and not sigar(pmax):
            pmax -= 1
        if not sigar(pmax):
            raise _hata("VAR Granger testi için seri çok kısa.")
        p = min((_var_olcut(Y, q, pmax), q) for q in range(1, pmax + 1))[1]
        secim = "AIC"

    yonler = []
    for etkilenen in range(k):
        e_u, k_u, T = _var_denklem(Y, p, p, etkilenen)
        ssr_u = float(e_u @ e_u)
        df_payda = T - k_u
        for neden in range(k):
            if neden == etkilenen:
                continue
            e_r, _, _ = _var_denklem(Y, p, p, etkilenen, {neden})
            ssr_r = float(e_r @ e_r)
            F = ((ssr_r - ssr_u) / p) / (ssr_u / df_payda)
            chi2 = T * (ssr_r - ssr_u) / ssr_u
            yonler.append(
                VARGrangerYon(
                    adlar[neden],
                    adlar[etkilenen],
                    p,
                    float(F),
                    float(stats.f.sf(F, p, df_payda)),
                    (p, df_payda),
                    float(chi2),
                    float(stats.chi2.sf(chi2, p)),
                    T,
                )
            )
    return VARGrangerSonucu(yonler, p, secim, adlar)


__all__ = [
    "adf",
    "kpss",
    "duraganlik",
    "engle_granger",
    "hata_duzeltme",
    "granger",
    "var_granger",
    "BirimKokSonucu",
    "DuraganlikSonucu",
    "EngleGrangerSonucu",
    "HataDuzeltmeSonucu",
    "GrangerSonucu",
    "GrangerYon",
    "VARGrangerSonucu",
    "VARGrangerYon",
    "mackinnon_p",
    "mackinnon_kritik",
]
