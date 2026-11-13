"""Katsayıların doğrusal birleşimi — Stata ``lincom`` karşılığı.

>>> ek.lincom(model, "egitim + 2*tecrube")          # β_egitim + 2·β_tecrube, SH ve güven aralığı
>>> ek.lincom(model, "x1 - x2")                       # iki katsayının farkı (H0: eşitler)
>>> ek.lincom(model, "_cons + C(cinsiyet)[T.Kadın]")  # sabit + kukla: kadınlar için kesişim
>>> ek.lincom(model, {"egitim": 1, "tecrube": -1})    # adlarında boşluk / işaret olan terimler için sözlük

Tahmin c'β, standart hata √(c' V c). İstatistik ve serbestlik derecesi modelle aynı kurala uyar (Stata gibi):
EKK ve sabit etkilerde t (kümelenmişte G − 1 sd), rassal etkilerde, logit/probit, sıralı / sayım / tobit, 2SLS (small yoksa)
ve xtabond'da z.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ._yardimci import p_cumle, p_yaz, sayi
from .regresyon import EkonometriHatasi


def _hata(m: str) -> EkonometriHatasi:
    return EkonometriHatasi(m)


@dataclass
class LincomSonucu:
    ifade: str
    tahmin: float
    std_hata: float
    istatistik: float
    ist_adi: str  # "t" ya da "z"
    sd: int | None
    p: float
    guven_araligi: tuple[float, float]
    guven: float
    agirliklar: dict

    def __str__(self) -> str:
        ist = f"{self.ist_adi}({self.sd})" if self.sd else self.ist_adi
        return "\n".join(
            [
                f"Doğrusal birleşim: {self.ifade}   (H0: = 0)",
                f"   {'Tahmin':>12}{'Std. hata':>12}{ist:>12}{'p':>10}   %{int(self.guven * 100)} güven aralığı",
                f"   {sayi(self.tahmin, 4):>12}{sayi(self.std_hata, 4):>12}{sayi(self.istatistik, 2):>12}{p_yaz(self.p):>10}   "
                f"[{sayi(self.guven_araligi[0], 4)}; {sayi(self.guven_araligi[1], 4)}]",
                f"   {'Birleşim sıfırdan anlamlı biçimde farklı' if self.p < 0.05 else 'Birleşim sıfırdan anlamlı biçimde farklı değil'} ({p_cumle(self.p)}).",
            ]
        )

    __repr__ = __str__


def _model_bilgisi(model) -> tuple[pd.Series, np.ndarray, str, int | None]:
    """(katsayılar, kovaryans, "t"/"z", sd) — modelin kendi çıkarım kuralına göre."""
    from .arac import IVSonuc
    from .dinamik import DinamikSonuc
    from .ikili import IkiliSonuc
    from .panel import PanelSonuc
    from .regresyon import EKKSonuc
    from .secim import MLSonuc

    b = getattr(model, "katsayilar", None)
    V = getattr(model, "kovaryans_matrisi", None)
    if not isinstance(b, pd.Series) or V is None:
        raise _hata(f"lincom bu sonuç için kullanılamaz ({type(model).__name__}); tahmin edilmiş bir model ver.")
    V = np.asarray(V, dtype=float)[: len(b), : len(b)]  # ML modellerinde ek parametreler (kesim, α, σ) sonda
    if isinstance(model, EKKSonuc):
        return b, V, "t", int(model.cikarim_sd or model.sd)
    if isinstance(model, PanelSonuc):
        return (b, V, "z", None) if model.yontem == "rassal" else (b, V, "t", int(model.cikarim_sd or model.sd))
    if isinstance(model, IVSonuc):
        return (b, V, "t", int(model.sd)) if model.kucuk_orneklem else (b, V, "z", None)
    if isinstance(model, (IkiliSonuc, MLSonuc, DinamikSonuc)):
        return b, V, "z", None
    raise _hata(f"lincom bu model türünü tanımıyor: {type(model).__name__}")


def _bol(ifade: str) -> list[str]:
    """'a + 2*b - c' → ['a', '+2*b', '-c']; parantez / köşeli parantez içindeki işaretlere ve 1e-3 gibi sayılara dokunmaz."""
    parca, derinlik, bas = [], 0, 0
    for i, c in enumerate(ifade):
        if c in "([":
            derinlik += 1
        elif c in ")]":
            derinlik -= 1
        elif c in "+-" and derinlik == 0 and i > 0:
            onceki = ifade[:i].rstrip()
            if onceki and onceki[-1] not in "*/" and not re.search(r"\d[eE]$", onceki):
                parca.append(ifade[bas:i])
                bas = i
    parca.append(ifade[bas:])
    return [p.strip() for p in parca if p.strip()]


def _cozumle(ifade: str, adlar: list[str], esle) -> tuple[dict, float]:
    agirlik: dict[str, float] = {}
    sabit = 0.0
    for terim in _bol(ifade):
        isaret = -1.0 if terim.startswith("-") else 1.0
        terim = terim.lstrip("+-").strip()
        carpan, ad = 1.0, None
        for oge in [o.strip() for o in re.split(r"\*(?![^\[]*\])(?![^(]*\))", terim)]:
            try:
                carpan *= float(oge.replace(",", "."))
                continue
            except ValueError:
                pass
            if "/" in oge and ad is None:
                sol, _, sag = oge.rpartition("/")
                try:
                    bolen = float(sag)
                    carpan /= bolen
                    oge = sol.strip()
                except ValueError:
                    pass
            if ad is not None:
                raise _hata(f"Doğrusal olmayan terim: {terim!r} (iki katsayının çarpımı lincom ile yapılamaz).")
            ad = esle(oge)
        if ad is None:
            sabit += isaret * carpan
        else:
            agirlik[ad] = agirlik.get(ad, 0.0) + isaret * carpan
    if not agirlik:
        raise _hata("İfadede katsayı yok. Örnek: ek.lincom(model, 'x1 + 2*x2')")
    return agirlik, sabit


def lincom(model, ifade, guven: float = 0.95) -> LincomSonucu:
    """Katsayıların doğrusal birleşimi, standart hatası, testi (H0: birleşim = 0) ve güven aralığı (Stata ``lincom``).

    ifade: "x1 + 2*x2 - 3", "x1 - x2", "_cons + C(g)[T.B]" ya da {ad: ağırlık}. ``_cons`` sabit terimdir.
    """
    b, V, ist_adi, sd = _model_bilgisi(model)
    adlar = list(b.index)

    def esle(ad: str) -> str:
        ad = ad.strip()
        m = re.fullmatch(r"_b\[(.+)\]", ad)  # Stata _b[x]
        if m:
            ad = m.group(1).strip()
        if ad in adlar:
            return ad
        if ad in ("_cons", "Sabit", "sabit") and "Intercept" in adlar:
            return "Intercept"
        bosluksuz = {a.replace(" ", ""): a for a in adlar}
        if ad.replace(" ", "") in bosluksuz:
            return bosluksuz[ad.replace(" ", "")]
        raise _hata(f"Modelde olmayan terim: {ad!r}. Terimler: {adlar}")

    if isinstance(ifade, dict):
        agirlik = {esle(k): float(v) for k, v in ifade.items()}
        sabit = 0.0
        metin = " + ".join(f"{v:g}*{k}" for k, v in agirlik.items())
    else:
        agirlik, sabit = _cozumle(str(ifade), adlar, esle)
        metin = str(ifade).strip()
    c = np.array([agirlik.get(a, 0.0) for a in adlar])
    tahmin = float(c @ b.values + sabit)
    se = float(np.sqrt(c @ V @ c))
    if not np.isfinite(se) or se == 0:
        raise _hata("Birleşimin standart hatası hesaplanamadı (sıfır ya da tanımsız).")
    istatistik = tahmin / se
    if ist_adi == "t":
        p = float(2 * stats.t.sf(abs(istatistik), sd))
        k = float(stats.t.ppf(0.5 + guven / 2, sd))
    else:
        p = float(2 * stats.norm.sf(abs(istatistik)))
        k = float(stats.norm.ppf(0.5 + guven / 2))
    return LincomSonucu(metin, tahmin, se, istatistik, ist_adi, sd, p, (tahmin - k * se, tahmin + k * se), guven, agirlik)


__all__ = ["lincom", "LincomSonucu"]
