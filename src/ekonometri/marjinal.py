"""Tahmin marjları ve koşullu marjinal etkiler — Stata ``margins`` karşılığı (EKK, logit, probit).

>>> ek.marjlar(model, at={"egitim": [8, 12, 16]})                 # margins, at(egitim=(8 12 16))
>>> ek.marjlar(model, grup="cinsiyet")                             # margins cinsiyet
>>> ek.marjlar(model, dydx="egitim", at={"tecrube": [5, 20, 35]}) # margins, dydx(egitim) at(tecrube=(5 20 35))
>>> ek.marjinal_grafik(ek.marjlar(model, at={"egitim": [8, 12, 16]}, grup="cinsiyet"))   # marginsplot

Tahminler, diğer değişkenler gözlenen değerlerinde tutularak hesaplanır (Stata ``margins`` varsayılanı, "ortalama tahmin").
Standart hatalar delta yöntemiyle hesaplanır. EKK'den sonra t (kalıntı serbestlik derecesiyle), logit/probit'ten sonra z kullanılır.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import patsy
from scipy import stats

from ._yardimci import p_yaz, sayi
from .regresyon import EkonometriHatasi


@dataclass
class MarjSonucu:
    tablo: pd.DataFrame
    tur: str  # "tahmin" | "etki"
    aciklama: str

    def __str__(self) -> str:
        t = self.tablo.copy()
        bicim = {c: (lambda x: sayi(x, 4)) for c in ("Marj", "Std. hata", "%95 GA alt", "%95 GA üst")}
        bicim[t.columns[[c in ("t", "z") for c in t.columns].index(True)]] = lambda x: sayi(x, 2)
        bicim["p"] = p_yaz
        return f"{self.aciklama}\n" + t.to_string(index=False, formatters=bicim)

    __repr__ = __str__


def _tahmin_islevi(model):
    """(tasarım matrisi, katsayılar) → (gözlem başına tahmin, gözlem başına türev ∂tahmin/∂b)."""
    from .ikili import _F, IkiliSonuc, _f

    if isinstance(model, IkiliSonuc):
        tur = model.tur

        def f(X, b):
            xb = X @ b
            return _F(xb, tur), _f(xb, tur)[:, None] * X

        return f, model._design_info, model._veri, getattr(model, "_agirliklar", None), "z", None

    if not hasattr(model, "_design_info") or model._design_info is None:
        raise EkonometriHatasi("marjlar() EKK, logit ve probit modelleri için kullanılabilir.")

    def f(X, b):
        return X @ b, X

    sd = getattr(model, "cikarim_sd", None) or model.sd
    return f, model._design_info, model._veri, getattr(model, "_agirlik_vektor", None), "t", sd


def _kategoriler(veri: pd.DataFrame, ad: str) -> list:
    x = veri[ad]
    if isinstance(x.dtype, pd.CategoricalDtype):
        return [c for c in x.cat.categories if (x == c).any()]
    return sorted(x.dropna().unique())


def _sabitle(veri: pd.DataFrame, ad: str, deger) -> pd.DataFrame:
    d = veri.copy()
    if isinstance(veri[ad].dtype, pd.CategoricalDtype):
        d[ad] = pd.Categorical([deger] * len(d), categories=veri[ad].cat.categories)
    else:
        d[ad] = deger
    return d


def marjlar(
    model,
    at: dict[str, list] | None = None,
    grup: str | None = None,
    dydx: str | None = None,
    guven: float = 0.95,
) -> MarjSonucu:
    """Tahmin marjları ya da koşullu marjinal etkiler (Stata ``margins``).

    at   : {"x": [değerler]} — x her değere sabitlenerek ortalama tahmin (``margins, at(x=(...))``)
    grup : kategorik değişken — her grup için ortalama tahmin (``margins grup``)
    dydx : sürekli ya da kategorik değişken — ortalama marjinal etki; ``at`` / ``grup`` ile birlikte koşullu etki
    """
    f, di, veri, agirlik, ist, sd = _tahmin_islevi(model)
    if veri is None:
        raise EkonometriHatasi("Modelin veri kopyası yok; marjlar() bu model için kullanılamaz.")
    b = model.katsayilar.values
    V = model.kovaryans_matrisi.values
    at = dict(at or {})
    for ad in list(at) + ([grup] if grup else []) + ([dydx] if dydx else []):
        if ad not in veri.columns:
            raise EkonometriHatasi(f"{ad!r} modelde kullanılan bir değişken değil. Modeldeki değişkenler: {list(veri.columns)}")
    if grup and grup not in at:
        at[grup] = _kategoriler(veri, grup)

    def tasarim(d):
        (X,) = patsy.build_design_matrices([di], d)
        return np.asarray(X, dtype=float)

    def ortalama(d):
        tah, turev = f(tasarim(d), b)
        return float(np.average(tah, weights=agirlik)), np.average(turev, axis=0, weights=agirlik)

    def etki(d):
        """dydx: sürekli değişkende sayısal türev, kategorik değişkende her düzeyin referansa farkı."""
        if isinstance(d[dydx].dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(d[dydx]):
            duzeyler = _kategoriler(veri, dydx)
            ref, *diger = duzeyler
            m0, g0 = ortalama(_sabitle(d, dydx, ref))
            sonuc = []
            for lv in diger:
                m1, g1 = ortalama(_sabitle(d, dydx, lv))
                sonuc.append((f"{dydx}: {lv}", m1 - m0, g1 - g0))
            return sonuc
        h = 1e-5 * max(float(veri[dydx].std()), 1e-8)
        art, eks = d.copy(), d.copy()
        art[dydx] = d[dydx].astype(float) + h
        eks[dydx] = d[dydx].astype(float) - h
        (m1, g1), (m2, g2) = ortalama(art), ortalama(eks)
        return [(dydx, (m1 - m2) / (2 * h), (g1 - g2) / (2 * h))]

    adlar = list(at)
    satirlar = []
    for degerler in itertools.product(*[at[a] for a in adlar]) if adlar else [()]:
        d = veri.copy()
        for a, v in zip(adlar, degerler):
            d = _sabitle(d, a, v)
        konum = {f"{'grup' if a == grup else 'x'}:{a}": v for a, v in zip(adlar, degerler)}
        if dydx:
            for ad, m, g in etki(d):
                satirlar.append({**konum, "Etki": ad, "Marj": m, "_g": g})
        else:
            m, g = ortalama(d)
            satirlar.append({**konum, "Marj": m, "_g": g})
    t = pd.DataFrame(satirlar)
    se = np.array([np.sqrt(float(g @ V @ g)) for g in t.pop("_g")])
    t["Std. hata"] = se
    t[ist] = t["Marj"] / se
    if ist == "t":
        t["p"] = 2 * stats.t.sf(np.abs(t[ist]), sd)
        kritik = stats.t.ppf(0.5 + guven / 2, sd)
    else:
        t["p"] = 2 * stats.norm.sf(np.abs(t[ist]))
        kritik = stats.norm.ppf(0.5 + guven / 2)
    t["%95 GA alt"] = t["Marj"] - kritik * se
    t["%95 GA üst"] = t["Marj"] + kritik * se
    if dydx:
        aciklama = f"Ortalama marjinal etki: d(tahmin)/d({dydx})" + (f", koşul: {', '.join(adlar)}" if adlar else "") + " — delta yöntemi"
    else:
        aciklama = "Tahmin marjları (ortalama tahmin)" + (f", koşul: {', '.join(adlar)}" if adlar else "") + " — delta yöntemi"
    return MarjSonucu(t, "etki" if dydx else "tahmin", aciklama)


__all__ = ["marjlar", "MarjSonucu"]
