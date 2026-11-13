"""Vektör otoregresyon (VAR): tahmin, gecikme seçimi (Stata ``varsoc``), kararlılık (``varstable``) ve etki-tepki (``irf``).

>>> v = ek.var(veri, ["ln_ihracat", "ln_kur", "buyume"], gecikme=2)
>>> ek.var_gecikme_secimi(veri, ["ln_ihracat", "ln_kur"], en_fazla_gecikme=4)   # varsoc
>>> v.kararlilik()            # varstable
>>> v.etki_tepki(10)          # irf (Cholesky ortogonal)
>>> v.etki_tepki_grafigi(10)  # irf graph oirf

Hesaplar Stata ``var`` ile aynı tanımları kullanır: her denklem EKK ile tahmin edilir, hata kovaryansı en çok olabilirlik
tahminidir (Σ̂ = E'E / T), standart hatalar z istatistiğiyle raporlanır. Bilgi ölçütleri Lütkepohl (2005) tanımlarıdır.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ._yardimci import p_cumle, p_yaz, sayi
from .regresyon import EkonometriHatasi


def _hazirla(veri: pd.DataFrame, sutunlar: list[str]) -> np.ndarray:
    eksik = [c for c in sutunlar if c not in veri.columns]
    if eksik:
        raise EkonometriHatasi(f"Veride olmayan değişken(ler): {eksik}")
    if len(sutunlar) < 2:
        raise EkonometriHatasi("VAR için en az iki değişken gerekli.")
    d = veri[sutunlar].astype(float)
    if d.isna().any().any():
        d = d.dropna()
    return d.to_numpy()


def _tasarim(Y: np.ndarray, p: int, bas: int, sabit: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Bağımlı blok Y[bas:], açıklayıcılar [Y_{t-1}, …, Y_{t-p}, 1]."""
    T = len(Y)
    yy = Y[bas:]
    parca = [Y[bas - i : T - i] for i in range(1, p + 1)]
    if sabit:
        parca.append(np.ones((T - bas, 1)))
    X = np.hstack(parca) if parca else np.empty((T - bas, 0))
    return yy, X


def _ols(yy: np.ndarray, X: np.ndarray):
    if X.shape[1] == 0:
        return np.empty((0, yy.shape[1])), yy
    B = np.linalg.lstsq(X, yy, rcond=None)[0]
    return B, yy - X @ B


def _logdet(S: np.ndarray) -> float:
    isaret, ld = np.linalg.slogdet(S)
    return float(ld) if isaret > 0 else float("inf")


@dataclass
class VARSonucu:
    sutunlar: list[str]
    gecikme: int
    katsayilar: pd.DataFrame  # satır: açıklayıcı, sütun: denklem
    standart_hatalar: pd.DataFrame
    sigma: np.ndarray  # ML hata kovaryansı
    T: int
    log_olabilirlik: float
    aic: float
    hqic: float
    sbic: float
    _B: np.ndarray = field(repr=False, default=None)

    @property
    def z_degerleri(self) -> pd.DataFrame:
        return self.katsayilar / self.standart_hatalar

    @property
    def p_degerleri(self) -> pd.DataFrame:
        return pd.DataFrame(
            2 * stats.norm.sf(np.abs(self.z_degerleri.values)), index=self.katsayilar.index, columns=self.katsayilar.columns
        )

    def _A(self) -> list[np.ndarray]:
        K = len(self.sutunlar)
        return [self._B[i * K : (i + 1) * K].T for i in range(self.gecikme)]

    def kararlilik(self) -> pd.DataFrame:
        """Eşlik (companion) matrisinin özdeğerleri; hepsi birim çemberin içindeyse VAR kararlıdır (Stata ``varstable``)."""
        K, p = len(self.sutunlar), self.gecikme
        C = np.zeros((K * p, K * p))
        C[:K] = np.hstack(self._A())
        if p > 1:
            C[K:, :-K] = np.eye(K * (p - 1))
        oz = np.linalg.eigvals(C)
        t = pd.DataFrame({"Özdeğer": oz, "Modül": np.abs(oz)}).sort_values("Modül", ascending=False).reset_index(drop=True)
        t.attrs["kararli"] = bool((t["Modül"] < 1).all())
        return t

    def etki_tepki(self, ufuk: int = 10, ortogonal: bool = True) -> dict[tuple[str, str], np.ndarray]:
        """Etki-tepki fonksiyonları: {(etki, tepki): dizi[0..ufuk]}. ortogonal=True: Cholesky (Stata ``oirf``; değişken sırası önemlidir)."""
        K = len(self.sutunlar)
        A = self._A()
        Psi = [np.eye(K)]
        for h in range(1, ufuk + 1):
            Psi.append(sum(Psi[h - j] @ A[j - 1] for j in range(1, min(h, self.gecikme) + 1)))
        P = np.linalg.cholesky(self.sigma) if ortogonal else np.eye(K)
        sonuc = {}
        for i, etki in enumerate(self.sutunlar):
            for j, tepki in enumerate(self.sutunlar):
                sonuc[(etki, tepki)] = np.array([(Psi[h] @ P)[j, i] for h in range(ufuk + 1)])
        return sonuc

    def etki_tepki_grafigi(self, ufuk: int = 10, ortogonal: bool = True, baslik: str | None = None):
        """Tüm etki → tepki çiftleri için ızgara grafik (Stata ``irf graph oirf``)."""
        from .grafik import _bitir, _plt, _virgul

        plt = _plt()
        from matplotlib.ticker import FuncFormatter

        K = len(self.sutunlar)
        it = self.etki_tepki(ufuk, ortogonal)
        fig, eksenler = plt.subplots(K, K, figsize=(2.6 * K + 1, 2.1 * K + 0.6), squeeze=False, sharex=True)
        for i, etki in enumerate(self.sutunlar):
            for j, tepki in enumerate(self.sutunlar):
                ax = eksenler[j][i]
                ax.plot(range(ufuk + 1), it[(etki, tepki)], color="#1f4e79", linewidth=1.4)
                ax.axhline(0, color="grey", linewidth=0.7)
                ax.set_title(f"{etki} → {tepki}", fontsize=8)
                ax.yaxis.set_major_formatter(FuncFormatter(_virgul))
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(labelsize=7)
        fig.suptitle(baslik or ("Ortogonal etki-tepki fonksiyonları (Cholesky)" if ortogonal else "Etki-tepki fonksiyonları"), fontsize=10)
        return _bitir(fig, baslik or "Etki-tepki fonksiyonları")

    def yorumla(self):
        """Türkçe yorum: kararlılık, bilgi ölçütleri ve her denklemde %5'te anlamlı gecikmeler."""
        from .yorum import Yorum

        kr = self.kararlilik()
        enb = float(kr["Modül"].iloc[0])
        y = Yorum(f"VAR({self.gecikme}) modelinin yorumu — {', '.join(self.sutunlar)}")
        y.model_satirlari.append(
            f"Eşlik matrisinin en büyük özdeğer modülü {sayi(enb, 4)}; "
            + (
                "tüm özdeğerler birim çemberin içinde olduğundan VAR kararlıdır, etki-tepki fonksiyonları sönümlenir."
                if kr.attrs["kararli"]
                else "birim çemberin dışında / üzerinde özdeğer var: VAR kararsızdır, etki-tepki sonuçları güvenilir değildir "
                "(serilerin farkını almayı ya da eşbütünleşme varsa VECM'i düşün)."
            )
        )
        y.model_satirlari.append(
            f"T = {self.T}, log-olabilirlik = {sayi(self.log_olabilirlik, 2)}, AIC = {sayi(self.aic, 4)}, SBIC = {sayi(self.sbic, 4)}."
        )
        pv = self.p_degerleri
        for d in self.sutunlar:
            anlamli = [r for r in self.katsayilar.index if r != "Sabit" and pv.loc[r, d] < 0.05]
            if anlamli:
                y.katsayi_satirlari.append(
                    f"{d} denkleminde %5 düzeyinde anlamlı gecikmeler: "
                    + "; ".join(f"{r} ({sayi(self.katsayilar.loc[r, d], 4)}, {p_cumle(pv.loc[r, d])})" for r in anlamli)
                    + "."
                )
            else:
                y.katsayi_satirlari.append(f"{d} denkleminde %5 düzeyinde anlamlı gecikme yok.")
        y.notlar.append(
            "VAR katsayıları tek tek yorumlanmaz; değişkenler arası ilişki Granger nedensellik (vargranger / Toda-Yamamoto) ve "
            "etki-tepki fonksiyonlarıyla (irf) anlatılır. Cholesky etki-tepkileri değişken sırasına bağlıdır."
        )
        return y

    def rapor(
        self,
        dosya=None,
        baslik: str | None = None,
        etiketler: dict | None = None,
        tablo_no: int = 1,
        dil: str = "tr",
        bicim: str = "apa",
        etki_tepki: int | None = 10,
    ):
        """VAR tez raporu: yöntem metni, katsayı tablosu (SH parantez içinde), kararlılık ve isteğe bağlı etki-tepki şekli."""
        from ._yardimci import DIL, l, tablo_sayi, yildiz
        from .rapor import KAYNAKLAR, Rapor, Tablo, _Adlar, rapor_ayari

        token = rapor_ayari(dil, bicim)
        try:
            en = dil == "en"
            ad = _Adlar(etiketler)
            kr = self.kararlilik()
            enb = float(kr["Modül"].iloc[0])
            degiskenler = ", ".join(ad.metin(c) for c in self.sutunlar)
            if en:
                yontem = (
                    f"The dynamic relationships among {degiskenler} were analysed with a vector autoregression (VAR) model (Sims, 1980). "
                    f"Each equation of the VAR({self.gecikme}) was estimated by OLS over T = {self.T} observations; the error covariance "
                    f"matrix is the maximum-likelihood estimate and standard errors are reported with z statistics (Lütkepohl, 2005). "
                    f"Information criteria: AIC = {sayi(self.aic, 3)}, HQIC = {sayi(self.hqic, 3)}, SBIC = {sayi(self.sbic, 3)}."
                )
                istikrar = f"The largest modulus of the companion-matrix eigenvalues is {sayi(enb, 3)}; " + (
                    "all eigenvalues lie inside the unit circle, so the VAR satisfies the stability condition."
                    if kr.attrs["kararli"]
                    else "the VAR does not satisfy the stability condition."
                )
            else:
                yontem = (
                    f"{degiskenler} değişkenleri arasındaki dinamik ilişkiler vektör otoregresyon (VAR) modeliyle incelenmiştir (Sims, 1980). "
                    f"VAR({self.gecikme}) modelinin her denklemi T = {self.T} gözlemle EKK yöntemiyle tahmin edilmiş; hata kovaryans matrisi "
                    f"en çok olabilirlik tahminidir, standart hatalar z istatistiğiyle raporlanmıştır (Lütkepohl, 2005). Bilgi ölçütleri: "
                    f"AIC = {sayi(self.aic, 3)}, HQIC = {sayi(self.hqic, 3)}, SBIC = {sayi(self.sbic, 3)}."
                )
                istikrar = f"Eşlik matrisi özdeğerlerinin en büyük modülü {sayi(enb, 3)} olup " + (
                    "tüm özdeğerler birim çemberin içinde yer aldığından model kararlılık koşulunu sağlamaktadır."
                    if kr.attrs["kararli"]
                    else "model kararlılık koşulunu sağlamamaktadır; etki-tepki sonuçları temkinle yorumlanmalıdır."
                )
            pv = self.p_degerleri
            satirlar = []
            for r in self.katsayilar.index:
                gor = l("Sabit", "Constant") if r == "Sabit" else (r.split(".", 1)[0] + "." + ad.yalin(r.split(".", 1)[1]))
                satirlar.append([gor] + [f"{tablo_sayi(self.katsayilar.loc[r, d])}{yildiz(pv.loc[r, d])}" for d in self.sutunlar])
                satirlar.append([""] + [f"({tablo_sayi(self.standart_hatalar.loc[r, d])})" for d in self.sutunlar])
            t1 = Tablo(
                tablo_no,
                l(f"VAR({self.gecikme}) tahmin sonuçları", f"VAR({self.gecikme}) estimates"),
                [l("Açıklayıcı", "Regressor")] + [ad.yalin(d) for d in self.sutunlar],
                satirlar,
                l(
                    f"Sütunlar denklemlerdir. Parantez içinde standart hatalar. *** p < 0,01, ** p < 0,05, * p < 0,1. T = {self.T}.",
                    f"Columns are equations. Standard errors in parentheses. *** p < 0.01, ** p < 0.05, * p < 0.1. T = {self.T}.",
                ),
            )
            t2 = Tablo(
                tablo_no + 1,
                l("VAR kararlılık testi: eşlik matrisinin özdeğerleri", "VAR stability: eigenvalues of the companion matrix"),
                [l("Özdeğer", "Eigenvalue"), l("Modül", "Modulus")],
                [
                    [
                        f"{sayi(o.real, 4)}{'+' if o.imag >= 0 else '−'}{sayi(abs(o.imag), 4)}i"
                        if abs(o.imag) > 1e-12
                        else sayi(o.real, 4),
                        sayi(m_, 4),
                    ]
                    for o, m_ in zip(kr["Özdeğer"], kr["Modül"])
                ],
                l("Tüm modüller 1'den küçükse VAR kararlıdır.", "The VAR is stable if all moduli are below 1."),
            )
            bloklar = [
                ("baslik", l("Yöntem", "Method")),
                ("paragraf", yontem),
                ("baslik", l("Bulgular", "Results")),
                ("tablo", t1),
                ("paragraf", istikrar),
                ("tablo", t2),
            ]
            r = Rapor(baslik or l("VAR Modeli Sonuçları", "VAR model results"), bloklar, bicim=bicim, dil=dil)
            if etki_tepki:
                r.bloklar.append(
                    (
                        "paragraf",
                        l(
                            f"Şekilde {etki_tepki} dönemlik ortogonal (Cholesky) etki-tepki fonksiyonları verilmiştir; değişken sırası "
                            f"{', '.join(ad.yalin(c) for c in self.sutunlar)} şeklindedir.",
                            f"The figure shows orthogonalised (Cholesky) impulse responses over {etki_tepki} periods; the variable ordering is "
                            f"{', '.join(ad.yalin(c) for c in self.sutunlar)}.",
                        ),
                    )
                )
                r.sekil_ekle(
                    self.etki_tepki_grafigi(etki_tepki), l("Ortogonal etki-tepki fonksiyonları", "Orthogonalised impulse responses")
                )
            r.bloklar += [("baslik", l("Kaynakça", "References")), ("liste", sorted([KAYNAKLAR["sims"], KAYNAKLAR["lutkepohl"]]))]
            if dosya is not None:
                r.kaydet(dosya)
            return r
        finally:
            DIL.reset(token)

    def __str__(self) -> str:
        z, pv = self.z_degerleri, self.p_degerleri
        satir = [
            f"VAR({self.gecikme}) — değişkenler: {', '.join(self.sutunlar)} · T = {self.T}",
            f"Log-olabilirlik = {sayi(self.log_olabilirlik, 2)} · AIC = {sayi(self.aic, 4)} · HQIC = {sayi(self.hqic, 4)} · SBIC = {sayi(self.sbic, 4)}",
        ]
        for d in self.sutunlar:
            satir.append(f"\n▸ Denklem: {d}")
            for r in self.katsayilar.index:
                satir.append(
                    f"   {r:<22}{sayi(self.katsayilar.loc[r, d], 4):>12}{sayi(self.standart_hatalar.loc[r, d], 4):>12}"
                    f"{sayi(z.loc[r, d], 2):>9}{p_yaz(pv.loc[r, d]):>10}"
                )
        kr = self.kararlilik()
        satir.append(
            f"\nKararlılık: en büyük özdeğer modülü = {sayi(float(kr['Modül'].iloc[0]), 4)} → {'kararlı' if kr.attrs['kararli'] else 'KARARSIZ'}"
        )
        return "\n".join(satir)

    __repr__ = __str__


def var(veri: pd.DataFrame, sutunlar: list[str], gecikme: int = 1, sabit: bool = True) -> VARSonucu:
    """VAR(p) tahmini (Stata ``var y1 y2, lags(1/p)``). Veri zaman sırasına göre sıralı olmalıdır."""
    if gecikme < 1:
        raise EkonometriHatasi("Gecikme en az 1 olmalı.")
    Y = _hazirla(veri, sutunlar)
    yy, X = _tasarim(Y, gecikme, gecikme, sabit)
    T, K = yy.shape
    if T <= X.shape[1]:
        raise EkonometriHatasi("Gözlem sayısı parametre sayısına göre yetersiz.")
    B, E = _ols(yy, X)
    sigma = E.T @ E / T
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.outer(np.diag(XtX_inv), np.diag(sigma)))
    adlar = [f"L{i}.{c}" if i > 1 else f"L.{c}" for i in range(1, gecikme + 1) for c in sutunlar] + (["Sabit"] if sabit else [])
    ll = -T / 2 * (K * (1 + np.log(2 * np.pi)) + _logdet(sigma))
    t_p = K * X.shape[1]
    return VARSonucu(
        list(sutunlar),
        gecikme,
        pd.DataFrame(B, index=adlar, columns=sutunlar),
        pd.DataFrame(se, index=adlar, columns=sutunlar),
        sigma,
        T,
        float(ll),
        float(-2 * ll / T + 2 * t_p / T),
        float(-2 * ll / T + 2 * np.log(np.log(T)) / T * t_p),
        float(-2 * ll / T + np.log(T) / T * t_p),
        B,
    )


def var_gecikme_secimi(veri: pd.DataFrame, sutunlar: list[str], en_fazla_gecikme: int = 4, sabit: bool = True) -> pd.DataFrame:
    """Gecikme seçim tablosu (Stata ``varsoc``): LL, LR testi, FPE, AIC, HQIC, SBIC; her ölçütün seçtiği gecikme ``*`` ile.

    Tüm gecikmeler aynı (ortak) örneklemle karşılaştırılır: T = n − en_fazla_gecikme.
    """
    Y = _hazirla(veri, sutunlar)
    K = Y.shape[1]
    satirlar = []
    ll_once = None
    for p in range(0, en_fazla_gecikme + 1):
        yy, X = _tasarim(Y, p, en_fazla_gecikme, sabit)
        T = len(yy)
        _, E = _ols(yy, X)
        S = E.T @ E / T
        ll = -T / 2 * (K * (1 + np.log(2 * np.pi)) + _logdet(S))
        m = K * p + (1 if sabit else 0)
        t_p = K * m
        satir = {"Gecikme": p, "LL": ll}
        if ll_once is None:
            satir.update({"LR": np.nan, "sd": np.nan, "p": np.nan})
        else:
            lr = 2 * (ll - ll_once)
            satir.update({"LR": lr, "sd": K * K, "p": float(stats.chi2.sf(lr, K * K))})
        satir["FPE"] = float(np.linalg.det(E.T @ E / T) * ((T + m) / (T - m)) ** K)
        satir["AIC"] = -2 * ll / T + 2 * t_p / T
        satir["HQIC"] = -2 * ll / T + 2 * np.log(np.log(T)) / T * t_p
        satir["SBIC"] = -2 * ll / T + np.log(T) / T * t_p
        satirlar.append(satir)
        ll_once = ll
    t = pd.DataFrame(satirlar).set_index("Gecikme")
    secim = {c: int(t[c].idxmin()) for c in ("FPE", "AIC", "HQIC", "SBIC")}
    # LR: en büyük gecikmeden geriye, %5'te anlamlı ilk gecikme (Stata varsoc kuralı)
    anlamli = [g for g in t.index[1:] if t.loc[g, "p"] < 0.05]
    secim["LR"] = max(anlamli) if anlamli else 0
    t.attrs["secim"] = secim
    t.attrs["T"] = len(Y) - en_fazla_gecikme
    return t


__all__ = ["var", "var_gecikme_secimi", "VARSonucu"]
