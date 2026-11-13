"""Arellano-Bond fark GMM (xtabond): bir gecikmeli dinamik panel.

>>> ek.xtabond("np.log(satis) ~ np.log(calisan)", veri, birim="firma", zaman="yil")

Bağımlı değişkenin gecikmesi içseldir ve daha derin düzey gecikmeleriyle araçlanır.
Formüldeki diğer değişkenler katı dışsal kabul edilir; farkları hem açıklayıcı hem araçtır.
Tek adımlı GMM, fark alınmış hatanın MA(1) ağırlık matrisini kullanır. Standart hatalar sağlam sandviç tahminidir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import patsy
from scipy import stats

from ._yardimci import DIL, l, sayi, yildiz
from .panel import _xtset_al
from .rapor import KAYNAKLAR, Rapor, Tablo, _Adlar, _p, _p_hucre


def _hata(m: str):
    from .regresyon import EkonometriHatasi

    return EkonometriHatasi(m)


def _agirlik(m: int) -> np.ndarray:
    h = np.eye(m) * 2.0
    if m > 1:
        i = np.arange(m - 1)
        h[i, i + 1] = h[i + 1, i] = -1.0
    return h


@dataclass
class DinamikSonuc:
    """Arellano-Bond tek adımlı fark GMM sonucu."""

    katsayilar: pd.Series
    standart_hatalar: pd.Series
    z_degerleri: pd.Series
    p_degerleri: pd.Series
    sargan: float
    sargan_sd: int
    sargan_p: float
    n: int
    birim_sayisi: int
    gecikme: int
    adlar: list[str]
    kovaryans_turu: str = "klasik"
    ar_testleri: dict = None  # {1: (z, p), 2: (z, p)} — Arellano-Bond fark kalıntılarında otokorelasyon (Stata estat abond)
    sigma2: float = float("nan")

    def __str__(self) -> str:
        satirlar = [
            f"Arellano-Bond fark GMM — gecikme = {self.gecikme}, birim = {self.birim_sayisi}, fark denklemi = {self.n}",
            f"   Standart hatalar: {'tek adımlı sağlam (sandviç)' if self.kovaryans_turu == 'saglam' else 'tek adımlı klasik'}",
            f"   Sargan χ²({self.sargan_sd}) = {sayi(self.sargan, 3)}, {_p(self.sargan_p)}",
        ]
        for j, (zj, pj) in sorted((self.ar_testleri or {}).items()):
            satirlar.append(f"   Arellano-Bond AR({j}) z = {sayi(zj, 3)}, {_p(pj)}")
        for ad in self.katsayilar.index:
            satirlar.append(
                f"   {ad}: {sayi(float(self.katsayilar[ad]), 4)} (SH = {sayi(float(self.standart_hatalar[ad]), 4)}, {_p(float(self.p_degerleri[ad]))})"
            )
        return "\n".join(satirlar)

    __repr__ = __str__

    def _ar_metin(self, dil: str) -> str:
        ar = self.ar_testleri or {}
        if 1 not in ar or 2 not in ar:
            return ""
        (z1, p1), (z2, p2) = ar[1], ar[2]
        if dil == "en":
            return (
                f"The Arellano-Bond test finds first-order autocorrelation in the differenced residuals (z = {sayi(z1, 2)}; {_p(p1)}), as expected, "
                + (
                    "and no second-order autocorrelation"
                    if p2 >= 0.05
                    else "but also second-order autocorrelation, which invalidates the instruments"
                )
                + f" (z = {sayi(z2, 2)}; {_p(p2)}). "
            )
        return (
            f"Arellano-Bond testine göre fark kalıntılarında beklendiği gibi birinci derece otokorelasyon vardır (z = {sayi(z1, 2)}; {_p(p1)}); "
            + (
                "ikinci derece otokorelasyon ise yoktur"
                if p2 >= 0.05
                else "ancak ikinci derece otokorelasyon da vardır, bu durum araçların geçerliliğini bozar"
            )
            + f" (z = {sayi(z2, 2)}; {_p(p2)}). "
        )

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
            baslik = "Arellano-Bond GMM" if dil == "en" else "Arellano-Bond Dinamik Panel"
        token = rapor_ayari(dil, bicim)
        try:
            ad = _Adlar(etiketler)
            govde = l(
                f"Dinamik panel, Arellano ve Bond (1991) tek adımlı fark GMM tahmincisiyle tahmin edilmiştir. "
                f"Bağımlı değişkenin {self.gecikme} gecikmesi içsel kabul edilmiş, daha derin düzey değerleriyle araçlanmıştır. "
                "Diğer açıklayıcılar katı dışsaldır. Birim etkileri birinci farkla elenmiştir. "
                f"Aşırı tanımlama Sargan testiyle sınanmıştır (χ²({self.sargan_sd}) = {sayi(self.sargan, 2)}; {_p(self.sargan_p)}). "
                + self._ar_metin("tr")
                + (
                    "Standart hatalar tek adımlı sağlam sandviç tahminidir."
                    if self.kovaryans_turu == "saglam"
                    else "Standart hatalar tek adımlı klasik tahmindir."
                ),
                f"The dynamic panel was estimated by the one-step difference GMM estimator of Arellano and Bond (1991). "
                f"The {self.gecikme} lag(s) of the dependent variable are endogenous and instrumented by deeper lags in levels. "
                "The remaining regressors are treated as strictly exogenous. Unit effects are removed by first differencing. "
                f"Overidentification was tested with the Sargan statistic (χ²({self.sargan_sd}) = {sayi(self.sargan, 2)}; {_p(self.sargan_p)}). "
                + self._ar_metin("en")
                + (
                    "Standard errors are the one-step robust sandwich estimator."
                    if self.kovaryans_turu == "saglam"
                    else "Standard errors are the one-step conventional estimator."
                ),
            )
            satirlar = []
            for ad_k in self.katsayilar.index:
                gorunen = ad.yalin(ad_k[2:]) if ad_k.startswith("L.") else ad.yalin(ad_k)
                if ad_k.startswith("L."):
                    gorunen = f"L.{gorunen}"
                pv = float(self.p_degerleri[ad_k])
                satirlar.append(
                    [
                        gorunen,
                        sayi(float(self.katsayilar[ad_k]), 4) + yildiz(pv),
                        sayi(float(self.standart_hatalar[ad_k]), 4),
                        sayi(float(self.z_degerleri[ad_k]), 2),
                        _p_hucre(pv),
                    ]
                )
            t = Tablo(
                tablo_no,
                l("Arellano-Bond tahmin sonuçları", "Arellano-Bond estimates"),
                [l("Değişken", "Variable"), l("Katsayı", "Coefficient"), l("Std. hata", "Std. error"), "z", "p"],
                satirlar,
                l(
                    f"Birim sayısı = {self.birim_sayisi}; fark denklemi sayısı = {self.n}. Sargan χ²({self.sargan_sd}) = {sayi(self.sargan, 2)} ({_p(self.sargan_p)}).",
                    f"Units = {self.birim_sayisi}; differenced equations = {self.n}. Sargan χ²({self.sargan_sd}) = {sayi(self.sargan, 2)} ({_p(self.sargan_p)}).",
                ),
            )
            r = Rapor(
                baslik,
                [
                    ("paragraf", govde),
                    ("tablo", t),
                    ("baslik", l("Kaynakça", "References")),
                    ("liste", [KAYNAKLAR["arellano"], KAYNAKLAR["sargan"]]),
                ],
                bicim=bicim,
                dil=dil,
            )
            if dosya is not None:
                r.kaydet(dosya)
            return r
        finally:
            DIL.reset(token)


def _birimler(y: np.ndarray, X: np.ndarray, kod: np.ndarray, zaman: np.ndarray, gecikme: int):
    bloklar = []
    for g in np.unique(kod):
        sec = np.where(kod == g)[0]
        sira = sec[np.argsort(zaman[sec], kind="mergesort")]
        t = zaman[sira]
        if len(t) >= 2:
            adim = np.diff(t.astype(float))
            if not np.allclose(adim, adim[0]) or np.any(adim <= 0):
                raise _hata("Bir birimin zaman indeksinde boşluk var. xtabond kesintisiz dönem ister.")
        if len(sira) < gecikme + 2:
            continue
        bloklar.append((y[sira], X[sira]))
    if len(bloklar) < 2:
        raise _hata("xtabond için gecikmeye yetecek en az iki birim gerekli.")
    return bloklar


def _tasarim(bloklar, gecikme: int):
    """Her birim için (y, X, Z, H). Sütunlar tüm birimlerde ortaktır."""
    anahtar = []
    for yy, _xx in bloklar:
        t = len(yy)
        for p in range(gecikme + 1, t):
            for s in range(0, p - gecikme):
                anahtar.append((p, s))
    anahtar = list(dict.fromkeys(anahtar))
    nx = bloklar[0][1].shape[1]
    q_y = len(anahtar)
    yer = {k: i for i, k in enumerate(anahtar)}

    parcalar = []
    for yy, xx in bloklar:
        t = len(yy)
        poz = list(range(gecikme + 1, t))
        m = len(poz)
        y = np.empty(m)
        x = np.empty((m, gecikme + nx))
        z = np.zeros((m, q_y + nx))
        for j, p in enumerate(poz):
            y[j] = yy[p] - yy[p - 1]
            for lag in range(1, gecikme + 1):
                x[j, lag - 1] = yy[p - lag] - yy[p - lag - 1]
            if nx:
                x[j, gecikme:] = xx[p] - xx[p - 1]
                z[j, q_y:] = xx[p] - xx[p - 1]
            for s in range(0, p - gecikme):
                z[j, yer[(p, s)]] = yy[s]
        parcalar.append((y, x, z, _agirlik(m)))
    return parcalar


def xtabond(
    formul: str,
    veri: pd.DataFrame,
    birim: str | None = None,
    zaman: str | None = None,
    gecikme: int = 1,
    kovaryans: str = "klasik",
) -> DinamikSonuc:
    """Arellano-Bond (1991) tek adımlı fark GMM (Stata ``xtabond``).

    Formüldeki bağımlı değişkenin ``gecikme`` kadar gecikmesi içseldir. Diğer değişkenler katı dışsaldır.
    kovaryans : "klasik" (Stata varsayılanı; σ̂²·(X'ZAZ'X)⁻¹) ya da "saglam" (Stata ``vce(robust)``, sandviç).
    Sargan testi homoskedastik tek adımlı biçimdedir; σ̂² fark kalıntılarından Δê'Δê / (2(n − k)) ile tahmin edilir.
    AR(1) ve AR(2) Arellano-Bond testleri (Stata ``estat abond``) sağlam biçimde hesaplanır.
    """
    if kovaryans not in ("klasik", "saglam"):
        raise _hata("kovaryans 'klasik' ya da 'saglam' olmalı.")
    if gecikme < 1:
        raise _hata("gecikme en az 1 olmalı.")
    birim, zaman = _xtset_al(veri, birim, zaman)
    if birim not in veri.columns or zaman not in veri.columns:
        raise _hata("Birim ve zaman sütunları veride olmalı.")
    env = patsy.EvalEnvironment.capture(1).with_outer_namespace({"np": np, "log": np.log, "ln": np.log, "exp": np.exp, "sqrt": np.sqrt})
    try:
        y_df, X_df = patsy.dmatrices(formul, veri, return_type="dataframe", NA_action="drop", eval_env=env)
    except patsy.PatsyError as h:
        raise _hata(f"Formül okunamadı: {h}") from None
    xs = X_df.drop(columns=["Intercept"], errors="ignore")
    idx = y_df.index
    y = y_df.iloc[:, 0].to_numpy(float)
    X = xs.to_numpy(float) if xs.shape[1] else np.zeros((len(y), 0))
    kod = pd.factorize(veri.loc[idx, birim])[0]
    takvim = pd.to_numeric(veri.loc[idx, zaman], errors="coerce").to_numpy(float)
    if not np.isfinite(takvim).all():
        raise _hata("Zaman sütunu sayıya çevrilemedi.")
    bloklar = _birimler(y, X, kod, takvim, gecikme)
    parcalar = _tasarim(bloklar, gecikme)
    k = parcalar[0][1].shape[1]
    q = parcalar[0][2].shape[1]
    if q < k:
        raise _hata("Araç sayısı katsayı sayısından az. Daha uzun bir panel ya da daha küçük gecikme gerekli.")
    a_ham = np.zeros((q, q))
    zx = np.zeros((q, k))
    zy = np.zeros(q)
    for yi, xi, zi, h in parcalar:
        a_ham += zi.T @ h @ zi
        zx += zi.T @ xi
        zy += zi.T @ yi
    a = np.linalg.pinv(a_ham)
    w = zx.T @ a @ zx
    g = zx.T @ a @ zy
    try:
        b = np.linalg.solve(w, g)
    except np.linalg.LinAlgError:
        b = np.linalg.lstsq(w, g, rcond=None)[0]
    anlar = np.zeros(q)
    dis = np.zeros((q, q))
    artik = []
    for yi, xi, zi, _ in parcalar:
        e = yi - xi @ b
        artik.append(e)
        gi = zi.T @ e
        anlar += gi
        dis += np.outer(gi, gi)
    sargan_pay = float(anlar @ a @ anlar)
    orta = zx.T @ a @ dis @ a @ zx
    w_inv = np.linalg.pinv(w)
    V_saglam = w_inv @ orta @ w_inv
    e_tum = np.concatenate(artik)
    # Var(Δε) = σ²·H ve H'nin köşegeni 2 olduğundan σ̂² fark kalıntılarının kareler ortalamasının yarısıdır
    sigma2 = float(e_tum @ e_tum) / (2 * (len(e_tum) - k))
    V = V_saglam if kovaryans == "saglam" else sigma2 * w_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    z = b / np.where(se == 0, np.nan, se)
    p = 2 * stats.norm.sf(np.abs(z))
    sargan = sargan_pay / sigma2
    sd = q - k
    # Arellano-Bond (1991) m_j testleri, sağlam biçim (plm::mtest ile aynı formül)
    ar = {}
    for j in (1, 2):
        pay, s_ww, wx, zuw = 0.0, 0.0, np.zeros(k), np.zeros(q)
        for (_yi, xi, zi, _), ei in zip(parcalar, artik):
            if len(ei) <= j:
                continue
            wi, ui = ei[:-j], ei[j:]
            c = float(wi @ ui)
            pay += c
            s_ww += c * c
            wx += wi @ xi[j:]
            zuw += (zi.T @ ei) * c
        orta_terim = wx @ w_inv @ zx.T @ a @ zuw
        payda = s_ww - 2 * orta_terim + wx @ V_saglam @ wx
        if payda > 0:
            zj = pay / np.sqrt(payda)
            ar[j] = (float(zj), float(2 * stats.norm.sf(abs(zj))))
    y_ad = y_df.columns[0]
    from .regresyon import _log_mu

    kaynak = _log_mu(y_ad) or y_ad
    adlar = [f"L{'' if lag == 1 else lag}.{kaynak}" for lag in range(1, gecikme + 1)] + list(xs.columns)

    def seri(v, ad):
        return pd.Series(v, index=adlar, name=ad)

    return DinamikSonuc(
        seri(b, "katsayı"),
        seri(se, "std. hata"),
        seri(z, "z"),
        seri(p, "p"),
        float(sargan),
        int(sd),
        float(stats.chi2.sf(sargan, sd)),
        int(len(e_tum)),
        len(parcalar),
        gecikme,
        adlar,
        kovaryans,
        ar,
        sigma2,
    )


__all__ = ["xtabond", "DinamikSonuc"]
