"""ARDL sınır testi (Pesaran, Shin ve Smith, 2001): kısıtsız hata düzeltme modeli ve F sınırları.

>>> ek.ardl_sinir("np.log(ihracat) ~ np.log(reel_kur) + dunya_buyumesi", veri, gecikme=2)

k, uzun dönemdeki açıklayıcı sayısıdır (bağımlı değişken sayılmaz). Kritik değerler PSS tablosunun
statsmodels tarafından yeniden simüle edilmiş halidir. F istatistiği statsmodels ``UECM.bounds_test`` ile aynıdır;
statsmodels kritik değeri k+1 satırından okur, burada PSS'nin k tanımı kullanılır.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import patsy
from scipy import stats

from ._pss import crit_percentiles, crit_vals, large_p, small_p, stat_star
from ._yardimci import DIL, l, sayi, yildiz
from .rapor import KAYNAKLAR, Rapor, Tablo, _Adlar, _p, _p_hucre
from .zaman import _olcut, _zaman_boslugu_var

DURUM_ACIKLAMA = {
    1: ("no deterministic term; the test covers the level coefficients", "deterministik terim yok; sınama düzey katsayılarını kapsar"),
    2: ("constant is in the model and in the test", "sabit modelde ve sınamadadır"),
    3: ("constant is in the model; the test covers only the level coefficients", "sabit modeldedir; sınama yalnız düzey katsayılarındadır"),
    4: (
        "constant and trend are in the model; the test covers the trend and the level coefficients",
        "sabit ve trend modeldedir; sınama trend ile düzey katsayılarını kapsar",
    ),
    5: (
        "constant and trend are in the model; the test covers only the level coefficients",
        "sabit ve trend modeldedir; sınama yalnız düzey katsayılarındadır",
    ),
}


def _hata(m: str):
    from .regresyon import EkonometriHatasi

    return EkonometriHatasi(m)


def _pss_p(istatistik: float, k: int, durum: int, i1: bool) -> float:
    if istatistik <= 0:
        return 1.0
    anahtar = (k, durum, i1)
    esik = stat_star[anahtar]
    kat = small_p[anahtar] if istatistik > esik else large_p[anahtar]
    x = np.log(istatistik)
    return float(1 - stats.norm.cdf(sum(c * x**i for i, c in enumerate(kat))))


def _serileri_al(formul: str, veri: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str, list[str], pd.Index]:
    from .regresyon import bagimli_gorunumu

    y_df, X_df = patsy.dmatrices(formul, veri, return_type="dataframe")
    if _zaman_boslugu_var(y_df.index):
        raise _hata("Örneklemde zaman boşluğu var. ARDL için kesintisiz bir seri kullan.")
    xs = X_df.drop(columns=["Intercept"], errors="ignore")
    if xs.shape[1] == 0:
        raise _hata("ARDL sınır testi için en az bir açıklayıcı değişken gerekli.")
    if xs.shape[1] > 10:
        raise _hata("Pesaran-Shin-Smith tabloları en fazla 10 açıklayıcı içindir.")
    # geçici bir sonuç gibi görünen ad: gorunen_ad terim listesi ister; patsy sütun adı yeterli
    from .regresyon import _log_mu

    y_ad = bagimli_gorunumu(y_df.columns[0])
    x_adlar = []
    for c in xs.columns:
        kaynak = _log_mu(c)
        x_adlar.append(f"ln({kaynak})" if kaynak else c)
    return y_df.iloc[:, 0].to_numpy(float), xs.to_numpy(float), y_ad, x_adlar, y_df.index


def _tasarim(y: np.ndarray, X: np.ndarray, p: int, es_zamanli: bool, durum: int, bas: int) -> tuple[pd.DataFrame, list[str]]:
    """UECM satırları t = bas .. n-1. bas, modelin kendi gecikmesinden küçük olamaz."""
    if p < 1:
        raise _hata("ARDL gecikmesi en az 1 olmalı.")
    if bas < p:
        raise _hata("İç hata: ARDL örneklemi gecikmeden kısa.")
    n = len(y)
    t = np.arange(bas, n)
    if len(t) < 8:
        raise _hata(f"Gecikme {p} için gözlem yetersiz ({len(t)} kullanılabilir gözlem).")
    veri = pd.DataFrame({"dy": y[t] - y[t - 1], "y_l1": y[t - 1]})
    duzey = ["y_l1"]
    for j in range(X.shape[1]):
        ad = f"x{j}_l1"
        veri[ad] = X[t - 1, j]
        duzey.append(ad)
    for i in range(1, p):
        veri[f"dy_l{i}"] = y[t - i] - y[t - i - 1]
    for j in range(X.shape[1]):
        for lag in range(0 if es_zamanli else 1, p):
            veri[f"dx{j}_l{lag}"] = X[t - lag, j] - X[t - lag - 1, j]
    if durum in (4, 5):
        veri.insert(1, "trend", np.arange(1, n + 1)[t])
    return veri, duzey


def _gorunen(y_ad: str, x_adlar: list[str], p: int, es_zamanli: bool) -> dict[str, str]:
    g = {"y_l1": f"{y_ad}(-1)"}
    for j, ad in enumerate(x_adlar):
        g[f"x{j}_l1"] = f"{ad}(-1)"
    for i in range(1, p):
        g[f"dy_l{i}"] = f"Δ{y_ad}(-{i})"
    for j, ad in enumerate(x_adlar):
        for lag in range(0 if es_zamanli else 1, p):
            g[f"dx{j}_l{lag}"] = f"Δ{ad}" if lag == 0 else f"Δ{ad}(-{lag})"
    g["trend"] = "Trend"
    return g


def _model_kur(veri: pd.DataFrame, durum: int):
    from .regresyon import ekk

    parcalar = [c for c in veri.columns if c != "dy"]
    if durum in (4, 5):
        parcalar = ["trend"] + [c for c in parcalar if c != "trend"]
    formul = "dy ~ " + " + ".join(parcalar) + (" - 1" if durum == 1 else "")
    return ekk(formul, veri)


def _wald_f(model, adlar: list[str]) -> float:
    idx = [list(model.katsayilar.index).index(a) for a in adlar]
    q, k = len(idx), len(model.katsayilar)
    R = np.zeros((q, k))
    R[np.arange(q), idx] = 1
    b = model.katsayilar.to_numpy(float)
    V = model.kovaryans_matrisi.to_numpy(float)
    fark = R @ b
    orta = fark @ np.linalg.inv(R @ V @ R.T) @ fark
    return float(orta / q)


@dataclass
class ARDLSinirSonucu:
    """ARDL sınır testi. ``F`` Pesaran-Shin-Smith istatistiğidir; ``karar`` 5% sınırına göredir."""

    model: object
    F: float
    k: int
    durum: int
    gecikme: int
    es_zamanli: bool
    secim: str
    y_ad: str
    x_adlar: list
    gorunen: dict
    duzey: list
    alt: dict
    ust: dict
    p_alt: float
    p_ust: float
    karar: str
    _etiketler: dict

    @property
    def hiz(self) -> float:
        return float(self.model.katsayilar["y_l1"])

    def uzun_donem(self) -> dict[str, float]:
        """Açıklayıcıların uzun dönem çarpanı: −β_x / β_y."""
        by = self.hiz
        return {ad: float(-self.model.katsayilar[f"x{j}_l1"] / by) for j, ad in enumerate(self.x_adlar)}

    def _karar_cumle(self) -> str:
        f = sayi(self.F, 3)
        alt, ust = sayi(self.alt[5], 3), sayi(self.ust[5], 3)
        if self.karar == "esbutunlesik":
            govde = l(
                f"F istatistiği ({f}), %5 üst sınırı ({ust}) aşmaktadır. Düzey değişkenleri arasında uzun dönemli bir ilişki vardır.",
                f"The F statistic ({f}) exceeds the 5% upper bound ({ust}). The level variables are cointegrated.",
            )
        elif self.karar == "yok":
            govde = l(
                f"F istatistiği ({f}), %5 alt sınırının ({alt}) altındadır. Uzun dönemli bir ilişkiye dair kanıt bulunamamıştır.",
                f"The F statistic ({f}) is below the 5% lower bound ({alt}). There is no evidence of a long-run relationship.",
            )
        else:
            govde = l(
                f"F istatistiği ({f}), %5 alt ({alt}) ve üst ({ust}) sınırlarının arasındadır. Sınır testi belirsizdir.",
                f"The F statistic ({f}) lies between the 5% lower ({alt}) and upper ({ust}) bounds. The bounds test is inconclusive.",
            )
        return govde

    def __str__(self) -> str:
        isaret = {"esbutunlesik": "✅", "yok": "⚠️", "belirsiz": "⚠️"}[self.karar]
        baslik = l("ARDL sınır testi", "ARDL bounds test")
        return (
            f"{isaret} {baslik} — ARDL({self.gecikme}) — {l('durum', 'case')} {self.durum}, k = {self.k}, "
            f"{l('gecikme', 'lag')} = {self.gecikme} ({self.secim})\n"
            f"   F = {sayi(self.F, 3)}   I(0) %5 = {sayi(self.alt[5], 3)}   I(1) %5 = {sayi(self.ust[5], 3)}\n"
            f"   {self._karar_cumle()}"
        )

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
        if dil not in ("tr", "en") or bicim not in ("apa", "izgara"):
            raise _hata("dil 'tr' ya da 'en'; bicim 'apa' ya da 'izgara' olmalı.")
        token = DIL.set(dil)
        try:
            return self._rapor(dosya, baslik, etiketler, tablo_no, dil, bicim)
        finally:
            DIL.reset(token)

    def _rapor(self, dosya, baslik, etiketler, tablo_no, dil, bicim) -> Rapor:
        ad = _Adlar({**self._etiketler, **(etiketler or {})})
        baslik = baslik or l("ARDL Sınır Testi", "ARDL Bounds Test")
        uzun = self.uzun_donem()
        uzun_metin = "; ".join(f"{ad.yalin(a)} = {sayi(v, 4)}" for a, v in uzun.items())
        acik = DURUM_ACIKLAMA[self.durum][0 if dil == "en" else 1]
        if self.secim == "sabit":
            gecikme = l(f"Gecikme uzunluğu {self.gecikme} olarak belirlenmiştir.", f"The lag length is {self.gecikme}.")
        else:
            gecikme = l(
                f"Gecikme uzunluğu, ortak örneklemde {self.secim} ölçütüyle {self.gecikme} olarak seçilmiştir.",
                f"The lag length {self.gecikme} was selected by {self.secim} on a common sample.",
            )
        c = l(
            "Değişkenler arasındaki düzey ilişkisi Pesaran, Shin ve Smith (2001) sınır testi ile incelenmiştir. "
            f"Kısıtsız hata düzeltme modeli ARDL({self.gecikme}) düzenindedir; durum {self.durum} ({acik}). {gecikme} "
            f"Sınama, gecikmeli düzey katsayılarının ortak olarak sıfır olduğu hipotezine ait F istatistiğini I(0) ve I(1) "
            f"kritik sınırlarıyla karşılaştırır. k = {self.k}. {self._karar_cumle()} "
            f"Düzey katsayılarından hesaplanan uzun dönem çarpanları: {uzun_metin}. "
            f"Uyarlanma katsayısı {ad.metin(self.y_ad)}(-1) = {sayi(self.hiz, 4)} ({_p(float(self.model.p_degerleri['y_l1']))}).",
            "The level relationship was tested with the bounds test of Pesaran, Shin, and Smith (2001). "
            f"The unrestricted error-correction model is an ARDL({self.gecikme}); case {self.durum} ({acik}). {gecikme} "
            f"The test compares the F statistic for the joint hypothesis that the lagged level coefficients are zero "
            f"with the I(0) and I(1) critical bounds. k = {self.k}. {self._karar_cumle()} "
            f"Long-run multipliers implied by the level coefficients: {uzun_metin}. "
            f"The adjustment coefficient on {ad.metin(self.y_ad)}(-1) is {sayi(self.hiz, 4)} ({_p(float(self.model.p_degerleri['y_l1']))}).",
        )
        sinir = [
            [l("%10", "10%"), sayi(self.alt[10], 3), sayi(self.ust[10], 3)],
            [l("%5", "5%"), sayi(self.alt[5], 3), sayi(self.ust[5], 3)],
            [l("%1", "1%"), sayi(self.alt[1], 3), sayi(self.ust[1], 3)],
        ]
        t1 = Tablo(
            tablo_no,
            l("ARDL sınır testi", "ARDL bounds test"),
            [l("Anlamlılık", "Significance"), "I(0)", "I(1)"],
            sinir,
            l(
                f"F = {sayi(self.F, 3)}. H₀: düzey katsayıları ortak olarak sıfırdır. k = {self.k}; durum = {self.durum}. "
                "I(0) açıklayıcıların durağan, I(1) birim köklü olduğu sınırdır.",
                f"F = {sayi(self.F, 3)}. H₀: the level coefficients are jointly zero. k = {self.k}; case = {self.durum}. "
                "I(0) is the stationary bound and I(1) is the unit-root bound.",
            ),
        )
        m = self.model
        satirlar = []
        for sutun in m.katsayilar.index:
            gorunen = l("Sabit", "Constant") if sutun == "Intercept" else self.gorunen.get(sutun, sutun)
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
        t2 = Tablo(
            tablo_no + 1,
            l("Kısıtsız hata düzeltme modeli", "Unrestricted error-correction model"),
            [l("Değişken", "Variable"), l("Katsayı", "Coefficient"), l("Std. hata", "Std. error"), "t", "p"],
            satirlar,
            l(
                f"Bağımlı değişken: Δ{self.y_ad}. N = {m.n}; R² = {sayi(m.r2, 3)}.",
                f"Dependent variable: Δ{self.y_ad}. N = {m.n}; R² = {sayi(m.r2, 3)}.",
            ),
        )
        r = Rapor(
            baslik,
            [
                ("paragraf", c),
                ("tablo", t1),
                ("tablo", t2),
                ("baslik", l("Kaynakça", "References")),
                ("liste", [KAYNAKLAR["pesaran"]]),
            ],
            bicim=bicim,
            dil=dil,
        )
        if dosya is not None:
            r.kaydet(dosya)
        return r


def _kritik(k: int, durum: int) -> tuple[dict[int, float], dict[int, float]]:
    # dizin sırası: %10, %5, %1, %0,1 (90, 95, 99, 99.9 yüzdelikleri)
    anlam = {90: 10, 95: 5, 99: 1, 99.9: 0.1}
    alt = {anlam[y]: float(v) for y, v in zip(crit_percentiles, crit_vals[(k, durum, False)])}
    ust = {anlam[y]: float(v) for y, v in zip(crit_percentiles, crit_vals[(k, durum, True)])}
    return alt, ust


def ardl_sinir(
    formul: str,
    veri: pd.DataFrame,
    gecikme: int | None = None,
    en_fazla_gecikme: int | None = None,
    bilgi_olcutu: str = "aic",
    durum: int = 3,
    es_zamanli: bool = True,
) -> ARDLSinirSonucu:
    """Pesaran, Shin ve Smith (2001) ARDL sınır testi.

    Varsayılan durum 3: sabit modeldedir, F testi yalnız gecikmeli düzey katsayılarını sınar.
    ``gecikme`` tüm değişkenler için ortaktır (ARDL(p, p, ..., p)). Verilmezse ortak örneklemde AIC/BIC ile seçilir.
    ``es_zamanli=False`` açıklayıcıların cari farkını dışarıda bırakır.

    >>> ek.ardl_sinir("y ~ x + z", veri, gecikme=2, durum=3)
    """
    if durum not in DURUM_ACIKLAMA:
        raise _hata("durum 1, 2, 3, 4 ya da 5 olmalı.")
    if not isinstance(es_zamanli, bool):
        raise _hata("es_zamanli True ya da False olmalı.")
    olcut = _olcut(bilgi_olcutu)
    y, X, y_ad, x_adlar, _indeks = _serileri_al(formul, veri)
    n, nx = len(y), X.shape[1]

    def aday(p: int, bas: int):
        cerceve, duzey = _tasarim(y, X, p, es_zamanli, durum, bas)
        model = _model_kur(cerceve, durum)
        return model, duzey

    def sigar(p: int) -> bool:
        # sabit/trend + düzey (1+nx) + Δy gecikmeleri (p-1) + her x için p ya da p-1 fark
        kisa = nx * (p if es_zamanli else p - 1)
        det = 0 if durum == 1 else (2 if durum in (4, 5) else 1)
        return n - p > det + 1 + nx + (p - 1) + kisa + 4

    if gecikme is not None:
        if not sigar(int(gecikme)):
            raise _hata("Gecikme bu seri için çok büyük.")
        p, secim = int(gecikme), "sabit"
    else:
        pmax = 4 if en_fazla_gecikme is None else int(en_fazla_gecikme)
        if pmax < 1:
            raise _hata("en_fazla_gecikme en az 1 olmalı.")
        while pmax > 1 and not sigar(pmax):
            pmax -= 1
        if not sigar(pmax):
            raise _hata("ARDL sınır testi için seri çok kısa.")
        secenek = []
        for q in range(1, pmax + 1):
            model, _ = aday(q, pmax)
            secenek.append((model.aic if olcut == "aic" else model.bic, q))
        p, secim = min(secenek)[1], olcut.upper()

    model, duzey = aday(p, p)
    kisit = list(duzey)
    if durum == 2:
        kisit = ["Intercept"] + kisit
    elif durum == 4:
        kisit = ["trend"] + kisit
    F = _wald_f(model, kisit)
    alt, ust = _kritik(nx, durum)
    if F > ust[5]:
        karar = "esbutunlesik"
    elif F < alt[5]:
        karar = "yok"
    else:
        karar = "belirsiz"
    return ARDLSinirSonucu(
        model,
        F,
        nx,
        durum,
        p,
        es_zamanli,
        secim,
        y_ad,
        x_adlar,
        _gorunen(y_ad, x_adlar, p, es_zamanli),
        duzey,
        alt,
        ust,
        _pss_p(F, nx, durum, False),
        _pss_p(F, nx, durum, True),
        karar,
        dict(veri.attrs.get("etiketler", {})),
    )


__all__ = ["ardl_sinir", "ARDLSinirSonucu"]
