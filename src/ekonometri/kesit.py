"""Panelde yatay kesit bağımlılığı ve ikinci nesil birim kök testi.

>>> ek.yatay_kesit_bagimliligi(veri, "ln_gsyh")              # Stata xtcd: değişkenin kendisinde
>>> ek.yatay_kesit_bagimliligi(veri, model)                  # Stata xtcsd, pesaran: model kalıntılarında
>>> ek.cips(veri, "ln_gsyh", gecikme=1)                      # Pesaran (2007) CIPS (Stata pescadf / xtcips)

Yatay kesit bağımlılığı testleri (H0: birimler arasında korelasyon yok), birim çiftleri i < j için ρ_ij, ortak dönemlerdeki
Pearson korelasyonu ve T_ij ortak dönem sayısı olmak üzere:
* Breusch-Pagan (1980) LM = Σ T_ij ρ_ij² ~ χ²(N(N−1)/2)  — büyük N'de güvenilmez
* Pesaran (2004) ölçekli LM = √(1/(N(N−1))) Σ (T_ij ρ_ij² − 1) ~ N(0, 1)
* Pesaran (2004, 2015) CD = √(2/(N(N−1))) Σ √T_ij ρ_ij ~ N(0, 1)  — büyük N, küçük T'de de geçerli
* Baltagi, Feng ve Kang (2012) yanlılığı düzeltilmiş ölçekli LM (yalnız sabit etkiler modeli kalıntılarında)

CIPS: her birim için kesit ortalamalarıyla genişletilmiş ADF (CADF) regresyonu
    Δy_it = a_i + b_i y_i,t−1 + c_i ȳ_t−1 + Σ_{j=0..p} d_ij Δȳ_t−j + Σ_{j=1..p} δ_ij Δy_i,t−j (+ trend) + e_it
CADF_i = b_i'nin t değeri; CIPS = ortalama CADF_i. Kritik değerler Pesaran (2007) Tablo II (N ve T'ye göre doğrusal ara değer;
T, CADF regresyonundaki dönem sayısıdır).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ._cips_tablo import CIPS_KRITIK, CIPS_N, CIPS_T
from ._yardimci import DIL, l, p_cumle, p_yaz, sayi
from .regresyon import EkonometriHatasi


def _hata(m: str) -> EkonometriHatasi:
    return EkonometriHatasi(m)


def _genis(veri: pd.DataFrame, deger: pd.Series, birim: str, zaman: str) -> pd.DataFrame:
    """Uzun panel → geniş tablo (satır: zaman, sütun: birim)."""
    d = pd.DataFrame({"b": veri.loc[deger.index, birim].values, "z": veri.loc[deger.index, zaman].values, "x": deger.values}).dropna()
    if d.duplicated(["b", "z"]).any():
        raise _hata("Aynı birim-zaman çifti birden fazla kez geçiyor; panel yapısını (birim, zaman) kontrol et.")
    return d.pivot(index="z", columns="b", values="x").sort_index()


# --------------------------------------------------------------------------- yatay kesit bağımlılığı
@dataclass
class YatayKesitSonucu:
    kaynak: str  # değişken adı ya da "model kalıntıları"
    N: int
    T: int
    tablo: pd.DataFrame  # Test → İstatistik, sd, p
    ortalama_rho: float
    ortalama_mutlak_rho: float
    eksik_cift: int = 0

    @property
    def cd(self) -> float:
        return float(self.tablo.loc["Pesaran CD", "İstatistik"])

    @property
    def p(self) -> float:
        return float(self.tablo.loc["Pesaran CD", "p"])

    @property
    def bagimlilik_var(self) -> bool:
        return self.p < 0.05

    def karar(self) -> str:
        if self.bagimlilik_var:
            return (
                f"Pesaran CD testine göre yatay kesit bağımlılığı vardır (CD = {sayi(self.cd, 3)}, {p_cumle(self.p)}). "
                "Birinci nesil panel birim kök testleri (LLC, IPS, Fisher) yerine CIPS gibi ikinci nesil testler kullanılmalıdır."
            )
        return (
            f"Pesaran CD testine göre yatay kesit bağımlılığı bulunmamaktadır (CD = {sayi(self.cd, 3)}, {p_cumle(self.p)}); "
            "birinci nesil panel birim kök testleri kullanılabilir."
        )

    def __str__(self) -> str:
        s = [
            f"Yatay kesit bağımlılığı testleri — {self.kaynak} · N = {self.N}, T = {self.T}",
            "   H0: Birimler arasında korelasyon yoktur (yatay kesit bağımsızlığı).",
            f"   {'Test':<38}{'İstatistik':>12}{'sd':>8}{'p':>10}",
        ]
        for ad, r in self.tablo.iterrows():
            sd = "" if pd.isna(r["sd"]) else str(int(r["sd"]))
            s.append(f"   {ad:<38}{sayi(r['İstatistik'], 3):>12}{sd:>8}{p_yaz(r['p']):>10}")
        s.append(f"   Ortalama korelasyon ρ = {sayi(self.ortalama_rho, 3)}, ortalama |ρ| = {sayi(self.ortalama_mutlak_rho, 3)}")
        if self.eksik_cift:
            s.append(f"   Not: ortak dönemi 1 ya da daha az olan {self.eksik_cift} birim çifti hesaba katılmadı.")
        s.append(f"   {self.karar()}")
        return "\n".join(s)

    __repr__ = __str__

    def rapor(self, dosya=None, baslik=None, etiketler=None, tablo_no: int = 1, dil: str = "tr", bicim: str = "apa"):
        return yatay_kesit_raporu([self], dosya, baslik, etiketler, tablo_no, dil, bicim)


def yatay_kesit_raporu(sonuclar, dosya=None, baslik=None, etiketler=None, tablo_no: int = 1, dil: str = "tr", bicim: str = "apa"):
    """Birden çok değişken / model için tek yatay kesit bağımlılığı tablosu (LM, ölçekli LM, CD)."""
    from .rapor import KAYNAKLAR, Tablo, _Adlar, _p_hucre, rapor_ayari, test_raporu

    sonuclar = list(sonuclar)
    if not sonuclar:
        raise _hata("En az bir yatay kesit bağımlılığı sonucu ver.")
    token = rapor_ayari(dil, bicim)
    try:
        en = dil == "en"
        ad = _Adlar(etiketler)
        if en:
            p = [
                "Cross-sectional dependence was examined with the Breusch and Pagan (1980) LM, Pesaran (2004) scaled LM and Pesaran (2004, 2015) "
                "CD tests; the null hypothesis is cross-sectional independence. Since the LM tests are unreliable when N is large relative to T, "
                "the decision is based on the CD test."
            ]
        else:
            p = [
                "Birimler arasındaki yatay kesit bağımlılığı Breusch ve Pagan (1980) LM, Pesaran (2004) ölçekli LM ve Pesaran (2004, 2015) CD "
                "testleriyle incelenmiştir; temel hipotez yatay kesit bağımsızlığıdır. LM testleri N, T'ye göre büyük olduğunda güvenilir "
                "olmadığından karar CD testine göre verilmiştir."
            ]
        for s_ in sonuclar:
            a = l("model kalıntıları", "model residuals") if s_.kaynak == "model kalıntıları" else ad.metin(s_.kaynak)
            if en:
                p.append(
                    f"For {a}, CD = {sayi(s_.cd, 3)} ({p_cumle(s_.p)}): "
                    + (
                        "cross-sectional independence is rejected, so second-generation panel tests should be used."
                        if s_.bagimlilik_var
                        else "cross-sectional independence is not rejected."
                    )
                )
            else:
                p.append(
                    f"{a} için CD = {sayi(s_.cd, 3)} ({p_cumle(s_.p)}) olup "
                    + (
                        "yatay kesit bağımsızlığı reddedilmektedir; ikinci nesil panel testleri kullanılmalıdır."
                        if s_.bagimlilik_var
                        else "yatay kesit bağımsızlığı reddedilememektedir."
                    )
                )
        cevir = {
            "Breusch-Pagan LM": l("Breusch-Pagan LM", "Breusch-Pagan LM"),
            "Pesaran ölçekli LM": l("Pesaran ölçekli LM", "Pesaran scaled LM"),
            "Yanlılığı düzeltilmiş ölçekli LM": l("Yanlılığı düzeltilmiş ölçekli LM", "Bias-corrected scaled LM"),
            "Pesaran CD": "Pesaran CD",
        }
        satirlar = []
        for s_ in sonuclar:
            a = l("model kalıntıları", "model residuals") if s_.kaynak == "model kalıntıları" else ad.yalin(s_.kaynak)
            for test, r in s_.tablo.iterrows():
                satirlar.append([a, cevir[test], sayi(r["İstatistik"], 3), _p_hucre(r["p"])])
        t = Tablo(
            tablo_no,
            l("Yatay kesit bağımlılığı testi sonuçları", "Cross-sectional dependence test results"),
            [l("Değişken", "Variable"), "Test", l("İstatistik", "Statistic"), "p"],
            satirlar,
            l(
                "H₀: Yatay kesit bağımsızlığı. LM ~ χ²(N(N−1)/2); ölçekli LM ve CD ~ N(0, 1). Yanlılığı düzeltilmiş LM (Baltagi vd., 2012) "
                "yalnız sabit etkiler kalıntılarında hesaplanır.",
                "H₀: cross-sectional independence. LM ~ χ²(N(N−1)/2); scaled LM and CD ~ N(0, 1). The bias-corrected LM (Baltagi et al., 2012) "
                "is computed for fixed-effects residuals only.",
            ),
        )
        kaynak = [KAYNAKLAR["bp80"], KAYNAKLAR["pesaran_cd"], KAYNAKLAR["pesaran2015"]]
        if any("Yanlılığı düzeltilmiş ölçekli LM" in s_.tablo.index for s_ in sonuclar):
            kaynak.append(KAYNAKLAR["bfk2012"])
        return test_raporu(
            baslik or l("Yatay Kesit Bağımlılığı Testleri", "Cross-sectional dependence tests"), p, t, kaynak, dosya, dil, bicim
        )
    finally:
        DIL.reset(token)


def yatay_kesit_bagimliligi(veri: pd.DataFrame, degisken, birim: str | None = None, zaman: str | None = None) -> YatayKesitSonucu:
    """Yatay kesit bağımlılığı testleri: Breusch-Pagan LM, Pesaran ölçekli LM, Pesaran CD (+ sabit etkilerde yanlılığı düzeltilmiş LM).

    degisken: sütun adı (değişkenin kendisi; Stata ``xtcd``) ya da ``ek.panel(...)`` sonucu (kalıntılar; Stata ``xtcsd, pesaran``).
    Model verildiğinde ``veri``, modelin tahmin edildiği veri olmalıdır (birim ve zaman bilgisi buradan alınır).
    """
    from .panel import PanelSonuc, _xtset_al

    birim, zaman = _xtset_al(veri, birim, zaman)
    duzeltme = False
    if isinstance(degisken, PanelSonuc):
        deger = degisken.kalintilar
        if not deger.index.isin(veri.index).all():
            raise _hata("Model kalıntıları verinin satırlarıyla eşleşmiyor; modelin tahmin edildiği veriyi ver.")
        kaynak = "model kalıntıları"  # rapor dilinde çevrilir
        duzeltme = degisken.yontem == "sabit"
    else:
        if degisken not in veri.columns:
            raise _hata(f"Veride olmayan değişken: {degisken!r}")
        deger = veri[degisken].astype(float)
        kaynak = degisken
    G = _genis(veri, deger, birim, zaman)
    A = G.to_numpy(float)
    N = A.shape[1]
    if N < 3:
        raise _hata("Yatay kesit bağımlılığı testi için en az 3 birim gerekli.")
    var = ~np.isnan(A)
    rho, tij = [], []
    eksik = 0
    for i in range(N):
        for j in range(i):
            m = var[:, i] & var[:, j]
            t = int(m.sum())
            if t <= 1:
                eksik += 1
                continue
            a, b = A[m, i], A[m, j]
            if a.std() == 0 or b.std() == 0:
                raise _hata("Bazı birimlerde seri zaman içinde sabit; korelasyon hesaplanamıyor.")
            rho.append(float(np.corrcoef(a, b)[0, 1]))
            tij.append(t)
    rho, tij = np.array(rho), np.array(tij, dtype=float)
    k = len(rho)
    lm = float(np.sum(tij * rho**2))
    sclm = float(np.sqrt(1 / (2 * k)) * np.sum(tij * rho**2 - 1))
    cd = float(np.sqrt(1 / k) * np.sum(np.sqrt(tij) * rho))
    satirlar = {
        "Breusch-Pagan LM": (lm, k, float(stats.chi2.sf(lm, k))),
        "Pesaran ölçekli LM": (sclm, np.nan, float(2 * stats.norm.sf(abs(sclm)))),
    }
    if duzeltme:
        bc = sclm - N / (2 * (tij.max() - 1))
        satirlar["Yanlılığı düzeltilmiş ölçekli LM"] = (bc, np.nan, float(2 * stats.norm.sf(abs(bc))))
    satirlar["Pesaran CD"] = (cd, np.nan, float(2 * stats.norm.sf(abs(cd))))
    tablo = pd.DataFrame(satirlar, index=["İstatistik", "sd", "p"]).T
    return YatayKesitSonucu(kaynak, N, int(var.sum(axis=0).max()), tablo, float(rho.mean()), float(np.abs(rho).mean()), eksik)


# --------------------------------------------------------------------------- CIPS
_KESIK_SINIR = {"yok": (-6.12, 4.16), "sabit": (-6.19, 2.61), "trend": (-6.42, 1.70)}


@dataclass
class CIPSSonucu:
    degisken: str
    deterministik: str
    gecikme: int
    N: int
    T: int  # CADF regresyonundaki dönem sayısı (kritik değer tablosu bununla okunur)
    istatistik: float
    kritik: dict
    kesik: bool
    cadf: pd.Series = field(repr=False, default=None)  # birim → CADF t istatistiği

    @property
    def birim_kok_reddedildi(self) -> bool:
        return self.istatistik < self.kritik["%5"]

    def karar(self) -> str:
        for duzey in ("%1", "%5", "%10"):
            if self.istatistik < self.kritik[duzey]:
                return (
                    f"Birim kök boş hipotezi {duzey} düzeyinde reddedildi (CIPS = {sayi(self.istatistik, 3)} < {sayi(self.kritik[duzey], 2)}): "
                    f"yatay kesit bağımlılığı dikkate alındığında {self.degisken} durağandır."
                )
        return (
            f"Birim kök boş hipotezi reddedilemedi (CIPS = {sayi(self.istatistik, 3)} > %10 kritik değer {sayi(self.kritik['%10'], 2)}): "
            f"{self.degisken} birim köklüdür; farkı alınarak yeniden test edilmelidir."
        )

    def __str__(self) -> str:
        det = {"yok": "yok", "sabit": "sabit", "trend": "sabit + trend"}[self.deterministik]
        return "\n".join(
            [
                f"Pesaran (2007) CIPS panel birim kök testi — {self.degisken}" + (" (kesik CADF)" if self.kesik else ""),
                "   H0: Tüm birimlerde birim kök var · H1: Birimlerin bir kısmı durağan (yatay kesit bağımlılığına dayanıklı)",
                f"   N = {self.N} · T = {self.T} · deterministik: {det} · CADF gecikmesi: {self.gecikme}",
                f"   CIPS = {sayi(self.istatistik, 4)}   kritik değerler: "
                + "  ".join(f"{k}: {sayi(v, 2)}" for k, v in self.kritik.items()),
                f"   {self.karar()}",
            ]
        )

    __repr__ = __str__

    def rapor(self, dosya=None, baslik=None, etiketler=None, tablo_no: int = 1, dil: str = "tr", bicim: str = "apa"):
        return cips_raporu([self], dosya, baslik, etiketler, tablo_no, dil, bicim)


def cips_raporu(sonuclar, dosya=None, baslik=None, etiketler=None, tablo_no: int = 1, dil: str = "tr", bicim: str = "apa"):
    """Birden çok değişken (düzey / fark) için tek CIPS tablosu ve metni."""
    from .rapor import KAYNAKLAR, Tablo, _Adlar, rapor_ayari, test_raporu

    sonuclar = list(sonuclar)
    if not sonuclar:
        raise _hata("En az bir CIPS sonucu ver.")
    token = rapor_ayari(dil, bicim)
    try:
        en = dil == "en"
        ad = _Adlar(etiketler)
        if en:
            p = [
                "Because the units may be cross-sectionally dependent, stationarity was examined with Pesaran's (2007) CIPS test, which averages "
                "cross-sectionally augmented ADF (CADF) statistics. The null hypothesis is that all units contain a unit root; critical values "
                "are taken from Pesaran (2007, Table II)."
            ]
        else:
            p = [
                "Birimler arasında yatay kesit bağımlılığı olabileceğinden durağanlık, kesit ortalamalarıyla genişletilmiş ADF (CADF) "
                "istatistiklerinin ortalaması olan Pesaran (2007) CIPS testiyle incelenmiştir. Temel hipotez tüm birimlerde birim kök "
                "bulunduğudur; kritik değerler Pesaran (2007, Tablo II)'den alınmıştır."
            ]
        for s_ in sonuclar:
            if en:
                p.append(
                    f"For {ad.metin(s_.degisken)} CIPS = {sayi(s_.istatistik, 3)} against a 5% critical value of {sayi(s_.kritik['%5'], 2)}; the "
                    "unit-root null is " + ("rejected." if s_.birim_kok_reddedildi else "not rejected.")
                )
            else:
                p.append(
                    f"{ad.metin(s_.degisken)} için CIPS = {sayi(s_.istatistik, 3)} olup %5 kritik değeri {sayi(s_.kritik['%5'], 2)} ile "
                    "karşılaştırıldığında birim kök boş hipotezi "
                    + ("reddedilmektedir." if s_.birim_kok_reddedildi else "reddedilememektedir.")
                )
        det_ad = {"yok": l("yok", "none"), "sabit": l("sabit", "constant"), "trend": l("sabit + trend", "constant + trend")}
        satirlar = [
            [
                ad.yalin(s_.degisken),
                det_ad[s_.deterministik],
                str(s_.gecikme),
                sayi(s_.istatistik, 3) + ("ᵏ" if s_.kesik else ""),
                sayi(s_.kritik["%1"], 2),
                sayi(s_.kritik["%5"], 2),
                sayi(s_.kritik["%10"], 2),
                l("Reddedildi", "Rejected") if s_.birim_kok_reddedildi else l("Reddedilemedi", "Not rejected"),
            ]
            for s_ in sonuclar
        ]
        t = Tablo(
            tablo_no,
            l("CIPS panel birim kök testi sonuçları", "CIPS panel unit root test results"),
            [
                l("Değişken", "Variable"),
                l("Deterministik", "Deterministics"),
                l("Gecikme", "Lags"),
                "CIPS",
                "%1",
                "%5",
                "%10",
                l("H₀ kararı (%5)", "Decision (5%)"),
            ],
            satirlar,
            l(
                f"H₀: Tüm birimlerde birim kök vardır. N = {sonuclar[0].N}, T = {sonuclar[0].T}. Kritik değerler Pesaran (2007) Tablo II'den N ve "
                "T'ye göre doğrusal ara değerle. ᵏ: kesik CADF istatistikleri (CIPS*).",
                f"H₀: all units contain a unit root. N = {sonuclar[0].N}, T = {sonuclar[0].T}. Critical values from Pesaran (2007), Table II, "
                "linearly interpolated in N and T. ᵏ: truncated CADF statistics (CIPS*).",
            ),
        )
        return test_raporu(
            baslik or l("CIPS Panel Birim Kök Testi", "CIPS panel unit root test"), p, t, [KAYNAKLAR["pesaran2007"]], dosya, dil, bicim
        )
    finally:
        DIL.reset(token)


def _cips_kritik(N: int, T: int, det: str, kesik: bool) -> dict:
    tablo = np.asarray(CIPS_KRITIK[kesik][det], dtype=float)
    Nn, Tt = np.asarray(CIPS_N, float), np.asarray(CIPS_T, float)
    if not (Nn[0] <= N <= Nn[-1]) or not (Tt[0] <= T <= Tt[-1]):
        warnings.warn(
            f"N = {N}, T = {T} Pesaran (2007) tablosunun aralığı dışında (10–200); en yakın kenar değerleri kullanıldı.", stacklevel=3
        )
    n_, t_ = float(np.clip(N, Nn[0], Nn[-1])), float(np.clip(T, Tt[0], Tt[-1]))
    sonuc = {}
    for i, ad in enumerate(("%1", "%5", "%10")):
        satir = np.array([np.interp(t_, Tt, tablo[i, r]) for r in range(len(Nn))])  # önce T'de, sonra N'de doğrusal
        sonuc[ad] = round(float(np.interp(n_, Nn, satir)), 4)
    return sonuc


def cips(
    veri: pd.DataFrame,
    degisken: str,
    birim: str | None = None,
    zaman: str | None = None,
    gecikme: int = 1,
    deterministik: str = "sabit",
    kesik: bool = False,
) -> CIPSSonucu:
    """Pesaran (2007) CIPS testi (Stata ``pescadf`` / ``xtcips``). Dengeli panel gerekir.

    gecikme       : CADF regresyonundaki gecikme sayısı p (Δy_i ve Δȳ gecikmeleri)
    deterministik : "sabit" (varsayılan), "trend" (sabit + trend) ya da "yok"
    kesik         : True ise CADF istatistikleri Pesaran (2007) sınırlarında kesilir (CIPS*; küçük T'de önerilir)
    """
    from .panel import _xtset_al

    if deterministik not in _KESIK_SINIR:
        raise _hata("deterministik: 'sabit', 'trend' ya da 'yok' olmalı.")
    if int(gecikme) < 0:
        raise _hata("Gecikme negatif olamaz.")
    p = int(gecikme)
    birim, zaman = _xtset_al(veri, birim, zaman)
    if degisken not in veri.columns:
        raise _hata(f"Veride olmayan değişken: {degisken!r}")
    G = _genis(veri, veri[degisken].astype(float), birim, zaman)
    if G.isna().any().any():
        raise _hata("CIPS dengeli panel ister (her birimde tüm dönemler dolu olmalı).")
    Y = G.to_numpy(float)  # T_ham × N
    T_ham, N = Y.shape
    T = T_ham - p - 1  # CADF regresyonundaki dönem sayısı
    m = 2 + 2 * p + 1 + (deterministik != "yok") + (deterministik == "trend")
    if T <= m:  # en az 1 serbestlik derecesi (R plm ile aynı sınır)
        raise _hata(f"Zaman boyutu (T = {T_ham}) {p} gecikmeli CADF için çok kısa.")
    dY = np.diff(Y, axis=0, prepend=np.nan)
    ort = Y.mean(axis=1)
    dort = np.diff(ort, prepend=np.nan)
    satir = np.arange(p + 1, T_ham)
    ortak = [dort[satir], ort[satir - 1]] + [dort[satir - j] for j in range(1, p + 1)]
    if deterministik != "yok":
        ortak.append(np.ones(T))
    if deterministik == "trend":
        ortak.append(np.arange(1, T + 1, dtype=float))
    t_ist = []
    for i in range(N):
        X = np.column_stack([Y[satir - 1, i]] + [dY[satir - j, i] for j in range(1, p + 1)] + ortak)
        yy = dY[satir, i]
        XtX_inv = np.linalg.pinv(X.T @ X)
        b = XtX_inv @ X.T @ yy
        e = yy - X @ b
        s2 = float(e @ e) / (T - np.linalg.matrix_rank(X))
        t_ist.append(float(b[0] / np.sqrt(s2 * XtX_inv[0, 0])))
    t_ist = np.array(t_ist)
    if kesik:
        alt, ust = _KESIK_SINIR[deterministik]
        t_ist = np.clip(t_ist, alt, ust)
    return CIPSSonucu(
        degisken,
        deterministik,
        p,
        N,
        T,
        float(t_ist.mean()),
        _cips_kritik(N, T, deterministik, kesik),
        kesik,
        pd.Series(t_ist, index=G.columns),
    )


__all__ = ["yatay_kesit_bagimliligi", "yatay_kesit_raporu", "YatayKesitSonucu", "cips", "cips_raporu", "CIPSSonucu"]
