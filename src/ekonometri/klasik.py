"""Klasik testler ve tahminciler: t testi, χ² bağımsızlık testi, tek yönlü ANOVA, Prais-Winsten / Cochrane-Orcutt.

>>> ek.ttest(veri, "ucret", by="cinsiyet")             # Stata: ttest ucret, by(cinsiyet)
>>> ek.ki_kare(veri, "cinsiyet", "sektor")             # Stata: tabulate cinsiyet sektor, chi2
>>> ek.anova(veri, "ucret", "bolge")                   # Stata: oneway ucret bolge
>>> ek.prais("np.log(ihracat) ~ np.log(reel_kur)", v)  # Stata: prais lnihr lnkur
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ._yardimci import p_cumle, p_yaz, sayi
from .regresyon import EkonometriHatasi


# =========================================================================== t testi
@dataclass
class TTestSonucu:
    tur: str  # tek | iki | eslesik
    aciklama: str
    gruplar: pd.DataFrame  # gruplara göre N, ortalama, std. hata, std. sapma, %95 GA
    fark: float
    std_hata: float
    t: float
    sd: float
    h0: str
    p_iki: float
    p_sol: float
    p_sag: float
    esit_varyans: bool = True

    @property
    def p_degeri(self) -> float:
        return self.p_iki

    def __str__(self) -> str:
        g = self.gruplar.to_string(float_format=lambda x: sayi(x, 4))
        sd = sayi(self.sd, 0) if float(self.sd).is_integer() else sayi(self.sd, 2)
        karar = (
            "p < 0,05: H0 reddedildi — fark istatistiksel olarak anlamlı."
            if self.p_iki < 0.05
            else "p ≥ 0,05: H0 reddedilemedi — anlamlı bir farka dair kanıt yok."
        )
        return (
            f"{self.aciklama}\n{g}\n"
            f"fark = {sayi(self.fark, 4)}   std. hata = {sayi(self.std_hata, 4)}   t = {sayi(self.t, 4)}   sd = {sd}"
            + ("" if self.esit_varyans else " (Satterthwaite)")
            + f"\nH0: {self.h0}\n"
            f"Ha: fark < 0  p = {p_yaz(self.p_sol)}    Ha: fark ≠ 0  p = {p_yaz(self.p_iki)}    Ha: fark > 0  p = {p_yaz(self.p_sag)}\n{karar}"
        )

    __repr__ = __str__


def _ozet_satir(x: np.ndarray) -> dict:
    n = len(x)
    ort = float(np.mean(x))
    ss = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    sh = ss / np.sqrt(n) if n > 1 else float("nan")
    t = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
    return {"N": n, "Ortalama": ort, "Std. hata": sh, "Std. sapma": ss, "%95 GA alt": ort - t * sh, "%95 GA üst": ort + t * sh}


def _t_sonuc(tur, aciklama, gruplar, fark, sh, sd, h0, esit=True) -> TTestSonucu:
    t = fark / sh
    return TTestSonucu(
        tur,
        aciklama,
        gruplar,
        fark,
        sh,
        t,
        sd,
        h0,
        float(2 * stats.t.sf(abs(t), sd)),
        float(stats.t.cdf(t, sd)),
        float(stats.t.sf(t, sd)),
        esit,
    )


def ttest(
    veri: pd.DataFrame, degisken: str, by: str | None = None, deger: float = 0.0, ikinci: str | None = None, esit_varyans: bool = True
) -> TTestSonucu:
    """Ortalama testleri — Stata ``ttest``.

    - Tek örneklem: ``ttest(veri, "x", deger=10)``  (Stata: ttest x == 10)
    - İki grup: ``ttest(veri, "x", by="g")``  (Stata: ttest x, by(g)); esit_varyans=False → Welch/Satterthwaite (``unequal``)
    - Eşleştirilmiş: ``ttest(veri, "x", ikinci="y")``  (Stata: ttest x == y)
    Fark Stata'daki gibi birinci grup (küçük değer / ilk kategori) eksi ikinci gruptur.
    """
    if degisken not in veri.columns:
        raise EkonometriHatasi(f"Değişken bulunamadı: {degisken!r}")
    if ikinci is not None:
        d = veri[[degisken, ikinci]].dropna().astype(float)
        fark = (d[degisken] - d[ikinci]).values
        g = pd.DataFrame(
            [_ozet_satir(d[degisken].values), _ozet_satir(d[ikinci].values), _ozet_satir(fark)], index=[degisken, ikinci, "fark"]
        )
        o = _ozet_satir(fark)
        return _t_sonuc(
            "eslesik", f"Eşleştirilmiş t testi: {degisken} − {ikinci}", g, o["Ortalama"], o["Std. hata"], o["N"] - 1, "Ortalama fark = 0"
        )
    if by is None:
        x = veri[degisken].dropna().astype(float).values
        o = _ozet_satir(x)
        g = pd.DataFrame([o], index=[degisken])
        return _t_sonuc(
            "tek",
            f"Tek örneklem t testi: ortalama({degisken}) = {sayi(deger, 4)}",
            g,
            o["Ortalama"] - deger,
            o["Std. hata"],
            o["N"] - 1,
            f"ortalama = {sayi(deger, 4)}",
        )
    d = veri[[degisken, by]].dropna()
    s = d[by]
    duzeyler = list(s.cat.categories[s.cat.categories.isin(s.unique())]) if isinstance(s.dtype, pd.CategoricalDtype) else sorted(s.unique())
    if len(duzeyler) != 2:
        raise EkonometriHatasi(f"{by} değişkeninin tam 2 düzeyi olmalı (bulunan: {duzeyler}). Stata: more than 2 groups found.")
    x1 = d.loc[s == duzeyler[0], degisken].astype(float).values
    x2 = d.loc[s == duzeyler[1], degisken].astype(float).values
    o1, o2 = _ozet_satir(x1), _ozet_satir(x2)
    n1, n2 = len(x1), len(x2)
    v1, v2 = o1["Std. sapma"] ** 2, o2["Std. sapma"] ** 2
    if esit_varyans:
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        sh = np.sqrt(sp2 * (1 / n1 + 1 / n2))
        sd = n1 + n2 - 2
    else:
        sh = np.sqrt(v1 / n1 + v2 / n2)
        sd = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    tum = _ozet_satir(np.concatenate([x1, x2]))
    g = pd.DataFrame([o1, o2, tum], index=[str(duzeyler[0]), str(duzeyler[1]), "toplam"])
    return _t_sonuc(
        "iki",
        f"İki örneklem t testi ({'eşit varyans' if esit_varyans else 'eşit olmayan varyans'}): {degisken}, gruplar: {by}",
        g,
        o1["Ortalama"] - o2["Ortalama"],
        float(sh),
        float(sd),
        f"ortalama({duzeyler[0]}) − ortalama({duzeyler[1]}) = 0",
        esit_varyans,
    )


# =========================================================================== χ² bağımsızlık testi
@dataclass
class KiKareSonucu:
    tablo: pd.DataFrame
    beklenen: pd.DataFrame
    ki_kare: float
    sd: int
    p_degeri: float
    cramer_v: float
    olabilirlik_orani: float
    lr_p: float

    def __str__(self) -> str:
        az = int((self.beklenen.values < 5).sum())
        uyari = f"\nUyarı: {az} hücrede beklenen frekans 5'in altında; χ² yaklaşımı güvenilir olmayabilir." if az else ""
        karar = (
            "p < 0,05: H0 reddedildi — değişkenler arasında ilişki var."
            if self.p_degeri < 0.05
            else "p ≥ 0,05: H0 reddedilemedi — değişkenlerin bağımsız olmadığına dair kanıt yok."
        )
        return (
            f"{self.tablo.to_string()}\n\nPearson χ²({self.sd}) = {sayi(self.ki_kare, 4)}   {p_cumle(self.p_degeri)}\n"
            f"Olabilirlik oranı χ²({self.sd}) = {sayi(self.olabilirlik_orani, 4)}   {p_cumle(self.lr_p)}\n"
            f"Cramér V = {sayi(self.cramer_v, 4)}\nH0: İki değişken birbirinden bağımsızdır.\n{karar}{uyari}"
        )

    __repr__ = __str__


def ki_kare(veri: pd.DataFrame, satir: str, sutun: str) -> KiKareSonucu:
    """İki kategorik değişken için Pearson χ² bağımsızlık testi — Stata ``tabulate satir sutun, chi2 lrchi2 V``."""
    tablo = pd.crosstab(veri[satir], veri[sutun])
    if tablo.shape[0] < 2 or tablo.shape[1] < 2:
        raise EkonometriHatasi("χ² testi için her iki değişkenin de en az 2 düzeyi olmalı.")
    chi2, p, sd, beklenen = stats.chi2_contingency(tablo.values, correction=False)
    g, pg, _, _ = stats.chi2_contingency(tablo.values, correction=False, lambda_="log-likelihood")
    n = tablo.values.sum()
    k = min(tablo.shape) - 1
    v = np.sqrt(chi2 / (n * k))
    if tablo.shape == (2, 2):  # Stata 2×2 tabloda işaretli V verir
        a, b, c, d = tablo.values.ravel()
        v = (a * d - b * c) / np.sqrt(tablo.values.sum(axis=0).prod() * tablo.values.sum(axis=1).prod())
    tablo_t = tablo.copy()
    tablo_t.loc["Toplam"] = tablo.sum()
    tablo_t["Toplam"] = tablo_t.sum(axis=1)
    return KiKareSonucu(
        tablo_t,
        pd.DataFrame(beklenen, index=tablo.index, columns=tablo.columns),
        float(chi2),
        int(sd),
        float(p),
        float(v),
        float(g),
        float(pg),
    )


# =========================================================================== tek yönlü ANOVA
@dataclass
class AnovaSonucu:
    tablo: pd.DataFrame
    F: float
    sd: tuple[int, int]
    p_degeri: float
    bartlett: float
    bartlett_p: float
    gruplar: pd.DataFrame

    def __str__(self) -> str:
        karar = (
            "p < 0,05: H0 reddedildi — grup ortalamalarından en az biri farklı."
            if self.p_degeri < 0.05
            else "p ≥ 0,05: H0 reddedilemedi — grup ortalamalarının farklı olduğuna dair kanıt yok."
        )
        return (
            f"{self.gruplar.to_string(float_format=lambda x: sayi(x, 4))}\n\n{self.tablo.to_string(float_format=lambda x: sayi(x, 4))}\n"
            f"H0: Tüm grup ortalamaları eşittir.  F({self.sd[0]}, {self.sd[1]}) = {sayi(self.F, 4)}, {p_cumle(self.p_degeri)}\n{karar}\n"
            f"Bartlett eşit varyans testi: χ²({self.sd[0]}) = {sayi(self.bartlett, 4)}, {p_cumle(self.bartlett_p)}"
        )

    __repr__ = __str__


def anova(veri: pd.DataFrame, y: str, grup: str) -> AnovaSonucu:
    """Tek yönlü varyans analizi ve Bartlett eşit varyans testi — Stata ``oneway y grup, tabulate``."""
    d = veri[[y, grup]].dropna()
    gruplar = [(k, g[y].astype(float).values) for k, g in d.groupby(grup, observed=True, sort=True)]
    if len(gruplar) < 2:
        raise EkonometriHatasi("ANOVA için en az 2 grup gerekli.")
    x = d[y].astype(float).values
    ort = x.mean()
    ara = sum(len(g) * (g.mean() - ort) ** 2 for _, g in gruplar)
    ic = sum(((g - g.mean()) ** 2).sum() for _, g in gruplar)
    k, n = len(gruplar), len(x)
    F = (ara / (k - 1)) / (ic / (n - k))
    p = float(stats.f.sf(F, k - 1, n - k))
    b, bp = stats.bartlett(*[g for _, g in gruplar]) if all(len(g) > 1 for _, g in gruplar) else (float("nan"), float("nan"))
    tablo = pd.DataFrame(
        {"KT": [ara, ic, ara + ic], "sd": [k - 1, n - k, n - 1], "KO": [ara / (k - 1), ic / (n - k), (ara + ic) / (n - 1)]},
        index=["Gruplar arası", "Gruplar içi", "Toplam"],
    )
    ozet = pd.DataFrame({str(a): {"N": len(g), "Ortalama": g.mean(), "Std. sapma": g.std(ddof=1)} for a, g in gruplar}).T
    return AnovaSonucu(tablo, float(F), (k - 1, n - k), p, float(b), float(bp), ozet)


# =========================================================================== Prais-Winsten / Cochrane-Orcutt
def prais(
    formul: str,
    veri: pd.DataFrame,
    corc: bool = False,
    kovaryans: str = "klasik",
    tolerans: float = 1e-6,
    en_fazla: int = 100,
    birimler: dict | None = None,
):
    """AR(1) hatalı doğrusal model: yinelemeli Prais-Winsten (varsayılan) ya da Cochrane-Orcutt (``corc=True``) — Stata ``prais``.

    ρ, Stata'nın varsayılanı gibi kalıntıların bir dönem gecikmesine sabitsiz regresyonuyla tahmin edilir (rhotype(regress))
    ve ρ değişimi tolerans altına inene kadar yinelenir. Veri zaman sırasına göre sıralı olmalıdır (ek.tsset).
    Döndürülen sonuçta katsayılar dönüştürülmüş modelden, R² dönüştürülmüş regresyondan gelir; ``.rho``, ``.dw_ilk`` ve ``.dw_donusmus`` eklidir.
    """
    from .regresyon import _kovaryans, ekk

    if kovaryans not in ("klasik", "HC1"):
        raise EkonometriHatasi("prais için kovaryans 'klasik' ya da 'HC1' (Stata vce(robust)) olmalı.")
    taban = ekk(formul, veri, birimler=birimler)
    X, y = taban.X.values.astype(float), taban.y.values.astype(float)
    n, k = X.shape
    b = taban.katsayilar.values.copy()
    rho, yineleme = 0.0, 0
    for _ in range(en_fazla):
        yineleme += 1
        e = y - X @ b
        yeni = float(e[1:] @ e[:-1] / (e[:-1] @ e[:-1]))
        if not -1 < yeni < 1:
            raise EkonometriHatasi(f"Tahmin edilen ρ = {yeni:.4f} birim çember dışında; model uygun değil.")
        Xs, ys = X[1:] - yeni * X[:-1], y[1:] - yeni * y[:-1]
        if not corc:
            c = np.sqrt(1 - yeni**2)
            Xs, ys = np.vstack([c * X[:1], Xs]), np.concatenate([[c * y[0]], ys])
        b = np.linalg.lstsq(Xs, ys, rcond=None)[0]
        degisim = abs(yeni - rho)
        rho = yeni
        if degisim < tolerans:
            break
    else:
        raise EkonometriHatasi("Prais-Winsten yakınsamadı.")
    es = ys - Xs @ b
    m = len(ys)
    XtX_inv = np.linalg.inv(Xs.T @ Xs)
    V, aciklama = _kovaryans(Xs, es, XtX_inv, kovaryans, None)
    se = np.sqrt(np.diag(V))
    tv = b / se
    sdf = m - k
    pv = 2 * stats.t.sf(np.abs(tv), sdf)
    sse = float(es @ es)
    yort = ys.mean()
    r2 = 1 - sse / float(((ys - yort) ** 2).sum())  # Stata: dönüştürülmüş regresyonun R²'si
    idx = [i for i, c in enumerate(taban.X.columns) if c != "Intercept"]
    F = float(b[idx] @ np.linalg.inv(V[np.ix_(idx, idx)]) @ b[idx]) / len(idx) if idx else float("nan")
    e_orj = y - X @ b
    ad = list(taban.X.columns)
    seri = lambda v, isim: pd.Series(v, index=ad, name=isim)  # noqa: E731
    sonuc = dataclasses.replace(
        taban,
        katsayilar=seri(b, "katsayı"),
        standart_hatalar=seri(se, "std. hata"),
        t_degerleri=seri(tv, "t"),
        p_degerleri=seri(pv, "p"),
        kovaryans_matrisi=pd.DataFrame(V, index=ad, columns=ad),
        kovaryans_turu=kovaryans,
        kovaryans_aciklama=("Cochrane-Orcutt" if corc else "Prais-Winsten") + f" AR(1), ρ = {sayi(rho, 4)}; standart hatalar: " + aciklama,
        r2=float(r2),
        duzeltilmis_r2=float(1 - (1 - r2) * (m - 1) / sdf),
        f_istatistigi=F,
        f_p_degeri=float(stats.f.sf(F, len(idx), sdf)) if idx else float("nan"),
        f_serbestlik=(len(idx), sdf),
        n=m,
        sse=sse,
        kalintilar=pd.Series(e_orj, index=taban.y.index, name="kalıntı"),
        tahminler=pd.Series(X @ b, index=taban.y.index, name="tahmin"),
    )
    sonuc.model_turu = "Cochrane-Orcutt" if corc else "Prais-Winsten"
    sonuc.rho = rho
    sonuc.yineleme = yineleme
    sonuc.dw_ilk = float(np.sum(np.diff(taban.kalintilar.values) ** 2) / np.sum(taban.kalintilar.values**2))
    sonuc.dw_donusmus = float(np.sum(np.diff(es) ** 2) / np.sum(es**2))
    return sonuc


__all__ = ["ttest", "TTestSonucu", "ki_kare", "KiKareSonucu", "anova", "AnovaSonucu", "prais"]
