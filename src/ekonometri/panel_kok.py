"""Panel birim kök testleri — Stata ``xtunitroot`` karşılığı: Levin-Lin-Chu (LLC), Im-Pesaran-Shin (IPS), Fisher-ADF.

>>> veri = ek.xtset(veri, "ulke", "yil")
>>> ek.panel_birim_kok(veri, "ln_gsyh", test="llc")                       # xtunitroot llc ln_gsyh
>>> ek.panel_birim_kok(veri, "ln_gsyh", test="ips", gecikme="aic")        # xtunitroot ips ln_gsyh, lags(aic 4)
>>> ek.panel_birim_kok(veri, "ln_gsyh", test="fisher", gecikme=1)         # xtunitroot fisher ln_gsyh, dfuller lags(1)
>>> ek.panel_birim_kok(veri, "ln_gsyh", test="llc", deterministik="trend", ortalamadan_arindir=True)   # trend demean

Tüm testlerde H0: tüm birimlerde birim kök vardır. LLC'de alternatif "tüm birimler durağan (ortak ρ)", IPS ve Fisher'da
"en az bir kısım birim durağan"dır.

Hesaplama ayrıntıları:

* Birim başına ADF regresyonu: Δy_it = ρ y_i,t−1 + Σ γ_L Δy_i,t−L + deterministik + e_it.
* LLC (Levin, Lin ve Chu, 2002): birim başına kısa dönem σ̂_εi, Bartlett çekirdekli uzun dönem varyans
  (gecikme = round(3,21·T^{1/3})), havuzlanmış normalize regresyon ve Tablo 2'deki μ*, σ* düzeltmesi (T̃'ye göre doğrusal
  ara değer). Dengeli panel gerekir. Küçük T'de (≈50'nin altı) LLC boş hipotezi bir miktar fazla reddetme eğilimindedir;
  sonucu IPS / Fisher ile birlikte raporlamak iyi olur.
* IPS (Im, Pesaran ve Shin, 2003): birim ADF t istatistiklerinin ortalaması t-bar ve Tablo 3 momentleriyle
  W-t-bar = √N (t-bar − E) / √Var ~ N(0,1). Dengesiz panelde de kullanılabilir.
* Fisher (Maddala ve Wu, 1999; Choi, 2001): birim ADF p-değerleri (MacKinnon) birleştirilir: P (χ²), Z (ters normal),
  L* (logit) ve Pm (düzeltilmiş ters χ²).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ._panel_tablolar import IPS, IPS_T, LLC, LLC_T
from ._yardimci import p_cumle, p_yaz, sayi
from .regresyon import EkonometriHatasi

_DET = {"yok": "n", "sabit": "c", "trend": "ct"}
_TEST_AD = {
    "llc": "Levin-Lin-Chu (LLC)",
    "ips": "Im-Pesaran-Shin (IPS)",
    "fisher": "Fisher-ADF (Maddala-Wu / Choi)",
}


@dataclass
class PanelBirimKokSonucu:
    test: str
    degisken: str
    deterministik: str
    istatistikler: pd.DataFrame  # sütunlar: Değer, p
    N: int
    T: float  # ortalama zaman boyutu
    gecikmeler: pd.Series  # birim → seçilen gecikme
    birim_sonuclari: pd.DataFrame  # birim başına ADF: ρ, t, p
    ek: dict = field(default_factory=dict)

    @property
    def ana_istatistik(self) -> tuple[str, float, float]:
        ad = {"llc": "Düzeltilmiş t*", "ips": "W-t-bar", "fisher": "Pm"}[self.test]
        return ad, float(self.istatistikler.loc[ad, "Değer"]), float(self.istatistikler.loc[ad, "p"])

    @property
    def p(self) -> float:
        return self.ana_istatistik[2]

    @property
    def duragan(self) -> bool:
        return self.p < 0.05

    def yorum(self, anlamlilik: float = 0.05) -> str:
        ad, deger, p = self.ana_istatistik
        if p < anlamlilik:
            alt = "panelin tüm birimlerinde seri durağandır" if self.test == "llc" else "birimlerin en az bir kısmında seri durağandır"
            return (
                f"{_TEST_AD[self.test]} testine göre {self.degisken} değişkeni için birim kök boş hipotezi reddedilmiştir "
                f"({ad} = {sayi(deger, 3)}, {p_cumle(p)}); {alt}."
            )
        return (
            f"{_TEST_AD[self.test]} testine göre {self.degisken} değişkeni için birim kök boş hipotezi reddedilememiştir "
            f"({ad} = {sayi(deger, 3)}, {p_cumle(p)}); seri düzeyde durağan değildir, farkı alınarak yeniden test edilmelidir."
        )

    def __str__(self) -> str:
        det = {"yok": "yok", "sabit": "sabit", "trend": "sabit + trend"}[self.deterministik]
        g = self.gecikmeler
        gec = (
            f"{int(g.iloc[0])}"
            if g.nunique() == 1
            else f"ortalama {sayi(float(g.mean()), 2)} (en az {int(g.min())}, en çok {int(g.max())})"
        )
        satir = [
            f"{_TEST_AD[self.test]} panel birim kök testi — {self.degisken}",
            "H0: tüm birimlerde birim kök var  ·  "
            + ("H1: tüm birimler durağan" if self.test == "llc" else "H1: birimlerin bir kısmı durağan"),
            f"Birim sayısı N = {self.N} · zaman T = {sayi(self.T, 2 if self.T % 1 else 0)} · deterministik: {det} · ADF gecikmesi: {gec}"
            + (" · kesitsel ortalamadan arındırılmış" if self.ek.get("arindir") else ""),
        ]
        if self.test == "llc":
            satir.append(f"Uzun dönem varyans: Bartlett çekirdeği, {self.ek['bant']} gecikme")
        satir.append("")
        satir.append(f"   {'İstatistik':<24}{'Değer':>12}{'p':>10}")
        for ad, r in self.istatistikler.iterrows():
            satir.append(f"   {ad:<24}{sayi(r['Değer'], 4):>12}{(p_yaz(r['p']) if pd.notna(r['p']) else ''):>10}")
        satir.append("")
        satir.append(self.yorum())
        return "\n".join(satir)

    __repr__ = __str__


# --------------------------------------------------------------------------- yardımcılar
def _ols(X: np.ndarray, y: np.ndarray):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return b, e, float(e @ e)


def _deterministik(n: int, det: str) -> np.ndarray:
    if det == "yok":
        return np.empty((n, 0))
    if det == "sabit":
        return np.ones((n, 1))
    return np.column_stack([np.ones(n), np.arange(1, n + 1)])


def _adf_parcalar(x: np.ndarray, p: int, det: str, bas: int | None = None):
    """Δx_t'yi x_{t−1}, Δx_{t−1..p} ve deterministik terimlere regresle. bas: ilk kullanılan gözlem (sabit örneklem için)."""
    T = len(x)
    dx = np.diff(x, prepend=np.nan)
    bas = p + 1 if bas is None else bas
    satir = np.arange(bas, T)
    D = _deterministik(T, det)[satir]
    gec = np.column_stack([dx[satir - j] for j in range(1, p + 1)]) if p else np.empty((len(satir), 0))
    return dx[satir], x[satir - 1], gec, D


def _adf(x: np.ndarray, p: int, det: str, dfcor: bool):
    y, lx, gec, D = _adf_parcalar(x, p, det)
    X = np.column_stack([lx, gec, D])
    n, k = X.shape
    if n <= k:
        raise EkonometriHatasi("Bir birimin zaman boyutu seçilen gecikme için çok kısa.")
    b, e, rss = _ols(X, y)
    s2 = rss / (n - k) if dfcor else rss / n
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    return {
        "rho": float(b[0]),
        "se": float(se),
        "t": float(b[0] / se),
        "rss": rss,
        "n": n,
        "y": y,
        "lx": lx,
        "Z": np.column_stack([gec, D]),
    }


def _gecikme_sec(x: np.ndarray, det: str, en_fazla: int, olcut: str) -> int:
    """Birim başına bilgi ölçütüyle ADF gecikmesi (ortak örneklem: ilk en_fazla+1 gözlem atılır)."""
    en_iyi, secim = np.inf, 0
    for p in range(en_fazla + 1):
        y, lx, gec, D = _adf_parcalar(x, p, det, bas=en_fazla + 1)
        X = np.column_stack([lx, gec, D])
        n, k = X.shape
        if n <= k + 1:
            break
        _, _, rss = _ols(X, y)
        ceza = 2 * k / n if olcut == "aic" else (k * np.log(n) / n if olcut == "bic" else 2 * k * np.log(np.log(n)) / n)
        ic = np.log(rss / n) + ceza
        if ic < en_iyi - 1e-12:
            en_iyi, secim = ic, p
    return secim


def _uzun_donem_varyans(x: np.ndarray, det: str, q: int) -> float:
    """LLC (2002) Δy'nin Bartlett çekirdekli uzun dönem varyansı."""
    dx = np.diff(x)
    if det == "sabit":
        dx = dx - dx.mean()
    elif det == "trend":
        dx = _ols(np.column_stack([np.ones(len(dx)), np.arange(1, len(dx) + 1)]), dx)[1]
    T = len(x)
    v = float(dx @ dx) / (T - 1)
    for L in range(1, q + 1):
        v += 2 * (1 - L / (q + 1)) * float(dx[L:] @ dx[:-L]) / (T - 1)
    return v


def _ara_deger(tablo_T: list, degerler: list, t: float, uyarilar: set | None = None) -> float:
    Ts = np.asarray(tablo_T, float)
    if (t < Ts[0] or t > Ts[-1]) and uyarilar is not None:
        uyarilar.add((float(t), int(Ts[0]), int(Ts[-1])))
    return float(np.interp(t, Ts, np.asarray(degerler, float)))


def _uyar(uyarilar: set) -> None:
    if uyarilar:
        ts = sorted({u[0] for u in uyarilar})
        alt, ust = next(iter(uyarilar))[1:]
        warnings.warn(
            f"Zaman boyutu ({', '.join(sayi(t, 0) for t in ts)}) düzeltme tablosunun aralığı dışında ({alt}–{ust}); "
            "tablonun en yakın değeri kullanıldı. Kısa panelde sonucu temkinli yorumla.",
            stacklevel=3,
        )


def _seriler(veri, degisken, birim, zaman, arindir):
    from .panel import _xtset_al

    birim, zaman = _xtset_al(veri, birim, zaman)
    for c in (degisken, birim, zaman):
        if c not in veri.columns:
            raise EkonometriHatasi(f"Veride olmayan değişken: {c!r}")
    d = veri[[birim, zaman, degisken]].dropna().sort_values([birim, zaman])
    if arindir:  # Stata demean: her dönemde kesitsel ortalamayı çıkar
        d[degisken] = d[degisken] - d.groupby(zaman)[degisken].transform("mean")
    seriler = {}
    for b, g in d.groupby(birim, sort=True):
        zt = g[zaman].to_numpy()
        if np.issubdtype(zt.dtype, np.number) and len(zt) > 1 and np.any(np.diff(zt) != np.median(np.diff(zt))):
            raise EkonometriHatasi(f"{b!r} biriminde zaman boşluğu var; panel birim kök testleri ardışık gözlem ister.")
        seriler[b] = g[degisken].to_numpy(float)
    if len(seriler) < 2:
        raise EkonometriHatasi("Panel birim kök testi için en az iki birim gerekli.")
    return seriler


# --------------------------------------------------------------------------- ana işlev
def panel_birim_kok(
    veri: pd.DataFrame,
    degisken: str,
    test: str = "llc",
    birim: str | None = None,
    zaman: str | None = None,
    gecikme: int | str = 0,
    en_fazla_gecikme: int | None = None,
    deterministik: str = "sabit",
    ortalamadan_arindir: bool = False,
    bant: int | None = None,
    _llc_tablo: str = "T_tilde",
) -> PanelBirimKokSonucu:
    """Panel birim kök testi (Stata ``xtunitroot llc|ips|fisher``).

    test           : "llc", "ips" ya da "fisher"
    gecikme        : ADF gecikmesi; sayı (tüm birimler aynı) ya da "aic" / "bic" / "hqic" (birim başına seçim, Stata ``lags(aic #)``)
    en_fazla_gecikme: bilgi ölçütüyle seçimde üst sınır (varsayılan: T'ye göre 12·(T/100)^{1/4}, en fazla 8)
    deterministik  : "sabit" (varsayılan), "trend" (sabit + trend) ya da "yok" (yalnızca LLC ve Fisher)
    ortalamadan_arindir: her dönemde kesitsel ortalamayı çıkar (Stata ``demean``; kesitsel bağımlılığı azaltır)
    bant           : LLC uzun dönem varyansı için Bartlett gecikmesi (varsayılan round(3,21·T^{1/3}))
    """
    test = test.lower()
    if test not in _TEST_AD:
        raise EkonometriHatasi(f"Bilinmeyen test: {test!r}. Seçenekler: 'llc', 'ips', 'fisher'")
    if deterministik not in _DET:
        raise EkonometriHatasi("deterministik: 'sabit', 'trend' ya da 'yok' olmalı.")
    if test == "ips" and deterministik == "yok":
        raise EkonometriHatasi("IPS testi deterministik='yok' ile kullanılamaz (Stata'da da noconstant seçeneği yoktur).")
    seriler = _seriler(veri, degisken, birim, zaman, ortalamadan_arindir)
    adlar = list(seriler)
    N = len(adlar)
    Ti = np.array([len(seriler[b]) for b in adlar])

    if isinstance(gecikme, str):
        olcut = gecikme.lower()
        if olcut not in ("aic", "bic", "hqic"):
            raise EkonometriHatasi("gecikme sayı ya da 'aic' / 'bic' / 'hqic' olmalı.")
        pmax = en_fazla_gecikme if en_fazla_gecikme is not None else min(8, int(12 * (Ti.min() / 100) ** 0.25))
        p = np.array([_gecikme_sec(seriler[b], deterministik, pmax, olcut) for b in adlar])
    else:
        if int(gecikme) < 0:
            raise EkonometriHatasi("Gecikme negatif olamaz.")
        p = np.full(N, int(gecikme))

    dfcor = test != "llc"  # LLC: σ̂²_εi = RSS / (T − p − 1) (makale); IPS/Fisher: olağan ADF t istatistiği
    adf = [_adf(seriler[b], int(pi), deterministik, dfcor) for b, pi in zip(adlar, p)]
    t_i = np.array([a["t"] for a in adf])
    birim_tab = pd.DataFrame({"gecikme": p, "ρ": [a["rho"] for a in adf], "t": t_i}, index=pd.Index(adlar, name="birim"))
    if test != "llc":
        from .zaman import mackinnon_p

        birim_tab["p"] = [mackinnon_p(t, _DET[deterministik]) for t in t_i]
    ek: dict = {"arindir": ortalamadan_arindir}
    uyarilar: set = set()

    if test == "llc":
        if len(set(Ti)) > 1:
            raise EkonometriHatasi("LLC testi dengeli panel ister; dengesiz panelde test='ips' ya da test='fisher' kullan.")
        T = int(Ti[0])
        q = int(np.round(3.21 * T ** (1 / 3))) if bant is None else int(bant)
        e_tum, v_tum, s_i = [], [], []
        for b, a in zip(adlar, adf):
            sig = np.sqrt(a["rss"] / a["n"])
            Z = a["Z"]
            if Z.shape[1]:
                e = _ols(Z, a["y"])[1]
                v = _ols(Z, a["lx"])[1]
            else:
                e, v = a["y"], a["lx"]
            e_tum.append(e / sig)
            v_tum.append(v / sig)
            s_i.append(np.sqrt(_uzun_donem_varyans(seriler[b], deterministik, q)) / sig)
        e, v = np.concatenate(e_tum), np.concatenate(v_tum)
        delta = float(v @ e) / float(v @ v)
        rss = float(((e - delta * v) ** 2).sum())
        T_tilde = T - p.mean() - 1
        sigma2 = rss / (N * T_tilde)
        se_delta = np.sqrt(sigma2 / float(v @ v))
        t_delta = delta / se_delta
        S_N = float(np.mean(s_i))
        tab_T = T_tilde if _llc_tablo == "T_tilde" else T
        mu = _ara_deger(LLC_T, LLC[deterministik]["mu"], tab_T, uyarilar)
        sg = _ara_deger(LLC_T, LLC[deterministik]["sigma"], tab_T, uyarilar)
        t_star = (t_delta - N * T_tilde * S_N / sigma2 * se_delta * mu) / sg
        ist = pd.DataFrame(
            {"Değer": [t_delta, t_star], "p": [np.nan, float(stats.norm.cdf(t_star))]},
            index=["Düzeltilmemiş t", "Düzeltilmiş t*"],
        )
        ek.update({"bant": q, "delta": delta, "S_N": S_N, "sigma2": sigma2, "T_tilde": T_tilde, "mu": mu, "sigma_ayar": sg})
    elif test == "ips":
        L = Ti - p - 1
        if p.max() > 8:
            raise EkonometriHatasi("IPS W-t-bar düzeltme tablosu en fazla 8 gecikme içerir.")
        E = np.array([_ara_deger(IPS_T, IPS[deterministik]["ortalama"][pi], li, uyarilar) for pi, li in zip(p, L)])
        Vv = np.array([_ara_deger(IPS_T, IPS[deterministik]["varyans"][pi], li, uyarilar) for pi, li in zip(p, L)])
        if np.isnan(E).any() or np.isnan(Vv).any():
            raise EkonometriHatasi("Seçilen gecikme için zaman boyutu IPS düzeltme tablosunda yok (daha kısa gecikme dene).")
        tbar = float(t_i.mean())
        W = np.sqrt(N) * (tbar - E.mean()) / np.sqrt(Vv.mean())
        ist = pd.DataFrame({"Değer": [tbar, W], "p": [np.nan, float(stats.norm.cdf(W))]}, index=["t-bar", "W-t-bar"])
    else:
        pv = np.clip(birim_tab["p"].to_numpy(), 1e-300, 1 - 1e-16)
        P = float(-2 * np.log(pv).sum())
        Zs = float(stats.norm.ppf(pv).sum() / np.sqrt(N))
        k = 3 * (5 * N + 4) / (np.pi**2 * N * (5 * N + 2))
        Ls = float(np.sqrt(k) * np.log(pv / (1 - pv)).sum())
        Pm = float((-2 * np.log(pv) - 2).sum() / (2 * np.sqrt(N)))
        ist = pd.DataFrame(
            {
                "Değer": [P, Zs, Ls, Pm],
                "p": [
                    float(stats.chi2.sf(P, 2 * N)),
                    float(stats.norm.cdf(Zs)),
                    float(stats.t.cdf(Ls, 5 * N + 4)),
                    float(stats.norm.sf(Pm)),
                ],
            },
            index=["P (ters χ²)", "Z (ters normal)", "L* (ters logit)", "Pm"],
        )
    _uyar(uyarilar)
    return PanelBirimKokSonucu(test, degisken, deterministik, ist, N, float(Ti.mean()), birim_tab["gecikme"], birim_tab, ek)


def panel_duraganlik(
    veri: pd.DataFrame,
    degiskenler: list[str],
    birim: str | None = None,
    zaman: str | None = None,
    gecikme: int | str = "aic",
    deterministik: str = "sabit",
) -> pd.DataFrame:
    """Birden çok değişken için düzey ve birinci farkta IPS ve Fisher-ADF (tezlerdeki panel birim kök tablosu)."""
    from .panel import _xtset_al

    birim, zaman = _xtset_al(veri, birim, zaman)
    d = veri.sort_values([birim, zaman]).copy()
    satirlar = []
    for c in degiskenler:
        d[f"Δ{c}"] = d.groupby(birim)[c].diff()
        for ad, seviye in ((c, "Düzey"), (f"Δ{c}", "Birinci fark")):
            sat = {"Değişken": c, "Seviye": seviye}
            for t in ("ips", "fisher"):
                s = panel_birim_kok(d, ad, t, birim, zaman, gecikme=gecikme, deterministik=deterministik)
                isim, deger, p = s.ana_istatistik
                sat[f"{t.upper()} {isim}"] = deger
                sat[f"{t.upper()} p"] = p
            satirlar.append(sat)
    return pd.DataFrame(satirlar)


__all__ = ["panel_birim_kok", "panel_duraganlik", "PanelBirimKokSonucu"]
