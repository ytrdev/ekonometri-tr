"""Johansen eşbütünleşme testi: iz ve en büyük özdeğer.

>>> ek.johansen(veri, ["y", "x", "z"], gecikme=1, deterministik="sabit")

İstatistikler, fark denkleminden deterministik terimler ayıklandıktan sonra indirgenmiş rank regresyonunun özdeğerleridir.
Kritik değerler MacKinnon, Haug ve Michelis (1999) tablosudur.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._johansen_cv import kritik
from ._yardimci import DIL, l, sayi
from .rapor import KAYNAKLAR, Rapor, Tablo, _Adlar
from .zaman import _zaman_boslugu_var

DET = {"yok": -1, "sabit": 0, "trend": 1}
_DET_EN = {"yok": "no deterministic terms", "sabit": "a constant", "trend": "a constant and a linear trend"}


def _hata(m: str):
    from .regresyon import EkonometriHatasi

    return EkonometriHatasi(m)


def _durult(y: np.ndarray, derece: int) -> np.ndarray:
    """y'yi linspace(-1, 1) üzerinde derece. dereceli polinom kalıntısına çevirir. derece=-1 ise dokunmaz."""
    if derece < 0:
        return y
    n = len(y)
    x = np.vander(np.linspace(-1.0, 1.0, n), derece + 1)
    b, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ b


def _gecikmeler(x: np.ndarray, p: int) -> np.ndarray:
    """1..p gecikmeleri; ilk p satır atılır (statsmodels lagmat, trim='forward')."""
    if p == 0:
        return np.zeros((len(x), 0))
    n = len(x)
    kolonlar = [x[p - i : n - i] for i in range(1, p + 1)]
    return np.column_stack(kolonlar)


def _kalinti(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return y
    return y - x @ (np.linalg.pinv(x) @ y)


@dataclass
class JohansenSonucu:
    """Johansen rank testi. ``derece`` iz testinin %5 kararındaki eşbütünleşme sayısıdır."""

    iz: np.ndarray
    ozdeger_istatistik: np.ndarray
    ozdeger: np.ndarray
    vektor: np.ndarray
    iz_kritik: np.ndarray
    oz_kritik: np.ndarray
    derece: int
    gecikme: int
    deterministik: str
    degiskenler: list[str]
    n: int

    def __str__(self) -> str:
        satirlar = [
            f"Johansen eşbütünleşme — {self.deterministik}, gecikme farkı = {self.gecikme}, n = {self.n}",
            f"   İz testine göre eşbütünleşme derecesi: {self.derece}",
        ]
        for r, (iz, oz) in enumerate(zip(self.iz, self.ozdeger_istatistik)):
            satirlar.append(
                f"   r ≤ {r}: iz = {sayi(float(iz), 3)} (%5 = {sayi(float(self.iz_kritik[r, 1]), 3)}); "
                f"λ-max = {sayi(float(oz), 3)} (%5 = {sayi(float(self.oz_kritik[r, 1]), 3)})"
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
        from .rapor import rapor_ayari

        if baslik is None:
            baslik = "Johansen Cointegration" if dil == "en" else "Johansen Eşbütünleşme Testi"
        token = rapor_ayari(dil, bicim)
        try:
            ad = _Adlar(etiketler)
            adlar = ", ".join(ad.yalin(a) for a in self.degiskenler)
            govde = l(
                f"Değişkenler ({adlar}) arasındaki eşbütünleşme derecesi Johansen (1995) iz ve en büyük özdeğer testleriyle incelenmiştir. "
                f"Modelde {self.gecikme} gecikmeli fark ve deterministik yapı '{self.deterministik}' kullanılmıştır. "
                f"İz testi, %5 anlamlılık düzeyinde eşbütünleşme derecesini {self.derece} olarak vermektedir. "
                "Sıfır hipotezi, eşbütünleşme sayısının satırdaki r değerini aşmadığıdır.",
                f"The cointegration rank of {adlar} was examined with the trace and maximum-eigenvalue tests of Johansen (1995). "
                f"The model uses {self.gecikme} lagged difference(s) and {_DET_EN[self.deterministik]}. "
                f"At the 5% level the trace test selects rank {self.derece}. "
                "The null hypothesis is that the cointegration rank does not exceed the r shown in the row.",
            )
            satirlar = [
                [
                    f"r ≤ {r}",
                    sayi(float(self.iz[r]), 3),
                    sayi(float(self.iz_kritik[r, 1]), 3),
                    l("Reddedildi", "Rejected") if self.iz[r] > self.iz_kritik[r, 1] else l("Reddedilemedi", "Not rejected"),
                    sayi(float(self.ozdeger_istatistik[r]), 3),
                    sayi(float(self.oz_kritik[r, 1]), 3),
                    l("Reddedildi", "Rejected")
                    if self.ozdeger_istatistik[r] > self.oz_kritik[r, 1]
                    else l("Reddedilemedi", "Not rejected"),
                ]
                for r in range(len(self.iz))
            ]
            t = Tablo(
                tablo_no,
                l("Johansen eşbütünleşme testi", "Johansen cointegration test"),
                [
                    "H₀",
                    l("İz", "Trace"),
                    l("İz %5", "Trace 5%"),
                    l("İz kararı", "Trace decision"),
                    "λ-max",
                    "λ-max %5",
                    l("λ-max kararı", "λ-max decision"),
                ],
                satirlar,
                l(
                    f"N = {self.n}. Kritik değerler MacKinnon, Haug ve Michelis (1999).",
                    f"N = {self.n}. Critical values are from MacKinnon, Haug, and Michelis (1999).",
                ),
            )
            r = Rapor(
                baslik,
                [
                    ("paragraf", govde),
                    ("tablo", t),
                    ("baslik", l("Kaynakça", "References")),
                    ("liste", [KAYNAKLAR["johansen"], KAYNAKLAR["mackinnon1999"]]),
                ],
                bicim=bicim,
                dil=dil,
            )
            if dosya is not None:
                r.kaydet(dosya)
            return r
        finally:
            DIL.reset(token)


def johansen(
    veri: pd.DataFrame,
    sutunlar: list[str],
    gecikme: int = 1,
    deterministik: str = "sabit",
) -> JohansenSonucu:
    """Johansen iz ve en büyük özdeğer testi.

    ``gecikme``, VECM'deki fark gecikmesi sayısıdır (``k_ar_diff``). ``deterministik`` 'yok', 'sabit' ya da 'trend'.
    """
    if deterministik not in DET:
        raise _hata("deterministik 'yok', 'sabit' ya da 'trend' olmalı.")
    if gecikme < 0:
        raise _hata("gecikme negatif olamaz.")
    if len(sutunlar) < 2:
        raise _hata("Johansen testi için en az iki seri gerekli.")
    if len(sutunlar) > 12:
        raise _hata("Kritik değer tablosu en fazla 12 seri içindir.")
    eksik = [c for c in sutunlar if c not in veri.columns]
    if eksik:
        raise _hata(f"Veride olmayan sütun: {eksik[0]!r}.")
    parca = veri.loc[:, sutunlar].apply(pd.to_numeric, errors="coerce")
    if _zaman_boslugu_var(parca.index):
        raise _hata("Serilerde zaman boşluğu var. Johansen için kesintisiz bir aralık seç.")
    sonlu = np.isfinite(parca.to_numpy(float)).all(axis=1)
    if not sonlu.any():
        raise _hata("Serilerde kullanılabilir gözlem yok.")
    ilk = int(np.argmax(sonlu))
    son = len(sonlu) - int(np.argmax(sonlu[::-1]))
    if not bool(sonlu[ilk:son].all()):
        raise _hata("Serilerin ortasında eksik gözlem var.")
    endog = parca.to_numpy(float)[ilk:son]
    if len(endog) < gecikme + 8:
        raise _hata("Johansen testi için seri çok kısa.")
    det = DET[deterministik]
    f = 0 if det > -1 else det
    endog = _durult(endog, det)
    dx = np.diff(endog, axis=0)
    z = _gecikmeler(dx, gecikme)
    z = _durult(z, f) if z.size else z
    dx = _durult(dx[gecikme:], f)
    r0 = _kalinti(dx, z)
    duzey = endog[: len(endog) - gecikme][1:]
    duzey = _durult(duzey, f)
    rk = _kalinti(duzey, z)
    t = rk.shape[0]
    skk = rk.T @ rk / t
    sk0 = rk.T @ r0 / t
    s00 = r0.T @ r0 / t
    sig = sk0 @ np.linalg.inv(s00) @ sk0.T
    oz, vektor = np.linalg.eig(np.linalg.solve(skk, sig))
    oz = np.real_if_close(oz, tol=1e8)
    oz = np.real(oz)
    sira = np.argsort(oz)[::-1]
    oz = oz[sira]
    vektor = np.real(vektor[:, sira])
    if np.any(vektor != 0):
        vektor *= np.sign(vektor.flat[np.flatnonzero(vektor)[0]])
    k = endog.shape[1]
    iz = np.array([-t * np.sum(np.log(1 - oz[i:])) for i in range(k)])
    lam = np.array([-t * np.log(1 - oz[i]) for i in range(k)])
    iz_k = np.vstack([kritik(k - i, det, True) for i in range(k)])
    oz_k = np.vstack([kritik(k - i, det, False) for i in range(k)])
    derece = k
    for r in range(k):
        if iz[r] <= iz_k[r, 1]:
            derece = r
            break
    return JohansenSonucu(iz, lam, oz, vektor, iz_k, oz_k, derece, gecikme, deterministik, list(sutunlar), int(t))


__all__ = ["johansen", "JohansenSonucu"]
