"""Panel veri: havuzlanmış EKK, sabit etkiler (içsel/within), rassal etkiler (Swamy-Arora) ve model seçim testleri.

>>> veri = ek.veri.yukle("firma")
>>> fe = ek.panel("np.log(satis) ~ np.log(calisan) + np.log(sermaye) + arge", veri, birim="firma", zaman="yil", yontem="sabit")
>>> analiz = ek.panel_analizi("np.log(satis) ~ np.log(calisan) + np.log(sermaye) + arge", veri, birim="firma", zaman="yil")
>>> analiz                       # üç model + F, Breusch-Pagan LM ve Hausman testleri + önerilen model
>>> analiz.rapor("panel.docx")   # tez bölümü
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import patsy
from scipy import stats

from ._yardimci import DIL, benzersiz_indeks, l, p_cumle, p_yaz, sayi, tablo_sayi, yildiz
from .regresyon import EkonometriHatasi, _terimleri_coz, bagimli_gorunumu, gorunen_ad

YONTEMLER = {"havuz": "Havuzlanmış EKK", "sabit": "Sabit etkiler", "rassal": "Rassal etkiler"}
_YONTEM_EN = {"havuz": "Pooled OLS", "sabit": "Fixed effects", "rassal": "Random effects"}


def _yontem_ad(kod: str) -> str:
    return l(YONTEMLER[kod], _YONTEM_EN[kod])


def _se_metin(acik: str) -> str:
    if DIL.get() != "en":
        return acik
    if acik == "klasik":
        return "classical"
    onek = "birimlere göre kümelenmiş ("
    if acik.startswith(onek) and acik.endswith(")"):
        ic = acik[len(onek) : -1].replace(
            " küme; değişen varyans ve birim içi otokorelasyona dayanıklı",
            " clusters; robust to heteroskedasticity and within-unit autocorrelation",
        )
        return f"clustered by unit ({ic})"
    return acik


@dataclass
class PanelSonuc:
    """Panel veri modeli sonuçları. ``ek.panel(...)`` tarafından döndürülür."""

    yontem: str
    formul: str
    bagimli: str
    katsayilar: pd.Series
    standart_hatalar: pd.Series
    t_degerleri: pd.Series
    p_degerleri: pd.Series
    kovaryans_matrisi: pd.DataFrame
    kovaryans_turu: str
    kovaryans_aciklama: str
    n: int
    k: int
    birim_sayisi: int
    zaman_sayisi: int
    sd: int
    sse: float
    r2: float  # modelin kendi R²'si: sabit etkilerde 'içsel' (within), diğerlerinde dönüştürülmüş regresyonun R²'si
    r2_ic: float
    r2_arasi: float
    r2_genel: float
    f_istatistigi: float
    f_p_degeri: float
    f_serbestlik: tuple[int, int]
    kalintilar: pd.Series
    zaman_etkisi: bool
    dengeli: bool
    cikarilanlar: list = field(default_factory=list)  # sabit etkilerde zamanla değişmediği için çıkarılan terimler
    sigma2_u: float | None = None
    sigma2_e: float | None = None
    theta: float | None = None
    X: pd.DataFrame = field(repr=False, default=None)
    _terimler: list = field(repr=False, default_factory=list)
    _birimler: dict = field(repr=False, default_factory=dict)
    _veri: pd.DataFrame = field(repr=False, default=None)
    _zaman_sutunlari: list = field(repr=False, default_factory=list)
    _cikarilan_degiskenler: list = field(repr=False, default_factory=list)

    cikarim_sd: int | None = None  # t testleri için payda sd (kümelenmişte G − 1); rassal etkilerde None → z
    _etiketler: dict = field(repr=False, default_factory=dict)

    @property
    def ist_adi(self) -> str:
        """Stata gibi: rassal etkilerde z, diğerlerinde t."""
        return "z" if self.yontem == "rassal" else "t"

    @property
    def genel_test(self) -> str:
        """Genel anlamlılık testinin metni: F(q, sd) ya da Wald χ²(q)."""
        if self.yontem == "rassal":
            return f"Wald χ²({self.f_serbestlik[0]}) = {sayi(self.f_istatistigi, 2)}"
        return f"F({self.f_serbestlik[0]}, {self.f_serbestlik[1]}) = {sayi(self.f_istatistigi, 2)}"

    @property
    def model_turu(self) -> str:
        return YONTEMLER[self.yontem]

    @property
    def duzeltilmis_r2(self) -> float:
        return 1 - (1 - self.r2) * (self.n - 1) / self.sd

    def _uyum(self) -> dict[str, str]:
        return {
            "Gözlem sayısı": sayi(self.n, 0),
            "Birim sayısı": sayi(self.birim_sayisi, 0),
            "R² (içsel)": sayi(self.r2_ic, 3),
            "R² (birimler arası)": sayi(self.r2_arasi, 3),
            "R² (genel)": sayi(self.r2_genel, 3),
            "Birim etkileri": l(
                {"havuz": "Yok", "sabit": "Sabit", "rassal": "Rassal"}[self.yontem],
                {"havuz": "None", "sabit": "Fixed", "rassal": "Random"}[self.yontem],
            ),
            "Zaman etkileri": l("Var", "Yes") if self.zaman_etkisi else l("Yok", "No"),
        }

    def guven_araliklari(self, guven: float = 0.95) -> pd.DataFrame:
        t = stats.norm.ppf(0.5 + guven / 2) if self.yontem == "rassal" else stats.t.ppf(0.5 + guven / 2, self.cikarim_sd or self.sd)
        return pd.DataFrame({"alt": self.katsayilar - t * self.standart_hatalar, "üst": self.katsayilar + t * self.standart_hatalar})

    def ozet(self) -> str:
        g = 78
        s = [
            "═" * g,
            f" Panel Veri — {self.model_turu} · bağımlı değişken: {bagimli_gorunumu(self.bagimli)}",
            "═" * g,
            f" Gözlem: {self.n:<8} Birim: {self.birim_sayisi:<6} Dönem: {self.zaman_sayisi:<5} Panel: {'dengeli' if self.dengeli else 'dengesiz'}",
            f" R² içsel: {sayi(self.r2_ic, 4):<9} arası: {sayi(self.r2_arasi, 4):<9} genel: {sayi(self.r2_genel, 4):<9} "
            f"{self.genel_test} (p {p_yaz(self.f_p_degeri)})",
            f" Standart hatalar: {self.kovaryans_aciklama}" + (" · zaman etkileri dahil" if self.zaman_etkisi else ""),
        ]
        if self.yontem == "rassal":
            s.append(f" σ²(birim) = {sayi(self.sigma2_u, 5)} · σ²(hata) = {sayi(self.sigma2_e, 5)} · θ = {sayi(self.theta, 4)}")
        s += ["─" * g, f" {'Değişken':<28}{'Katsayı':>12}{'Std. hata':>12}{self.ist_adi:>9}{'p':>10}", "─" * g]
        for ad in self.katsayilar.index:
            if ad in self._zaman_sutunlari:
                continue
            gor = gorunen_ad(self, ad)
            gor = gor if len(gor) <= 27 else gor[:26] + "…"
            p = self.p_degerleri[ad]
            s.append(
                f" {gor:<28}{tablo_sayi(self.katsayilar[ad]):>12}{tablo_sayi(self.standart_hatalar[ad]):>12}"
                f"{sayi(self.t_degerleri[ad], 2):>9}{p_yaz(p):>10} {yildiz(p)}"
            )
        s.append("─" * g)
        if self.cikarilanlar:
            s.append(f" Zamanla değişmediği için sabit etkiler modelinden çıkarılan: {', '.join(self.cikarilanlar)}")
        s.append("═" * g)
        return "\n".join(s)

    def __str__(self) -> str:
        return self.ozet()

    __repr__ = __str__

    def rapor(
        self,
        dosya: str | None = None,
        baslik: str = "Bulgular",
        etiketler: dict | None = None,
        tablo_no: int = 1,
        dil: str = "tr",
        bicim: str = "apa",
    ):
        """Tek panel modeli için tez raporu (Yöntem + tanımlayıcı istatistikler + sonuç tablosu + yorum).

        Model seçim testleriyle (F, LM, Hausman) tam bölüm için: ``ek.panel_analizi(...).rapor()``.
        """
        from . import __version__
        from .rapor import KAYNAKLAR, Rapor, Tablo, _Adlar, _p, _p_hucre, _tanimlayici, katsayi_paragraflari, rapor_ayari

        if dil == "en" and baslik == "Bulgular":
            baslik = "Findings"
        token = rapor_ayari(dil, bicim)
        try:
            ad = _Adlar({**self._etiketler, **(etiketler or {})})
            r = Rapor(baslik, bicim=bicim, dil=dil)
            denge = l("dengeli" if self.dengeli else "dengesiz", "balanced" if self.dengeli else "unbalanced")
            yontem_metni = l(
                {
                    "havuz": "havuzlanmış en küçük kareler (EKK) yöntemiyle",
                    "sabit": "sabit etkiler (içsel / within) tahmincisiyle",
                    "rassal": "rassal etkiler tahmincisiyle (Swamy ve Arora, 1972)",
                }[self.yontem],
                {
                    "havuz": "pooled ordinary least squares",
                    "sabit": "the fixed-effects (within) estimator",
                    "rassal": "the random-effects estimator (Swamy and Arora, 1972)",
                }[self.yontem],
            )
            y = l(
                f"Çalışmada {sayi(self.birim_sayisi, 0)} birim ve {sayi(self.zaman_sayisi, 0)} dönemden oluşan {denge} "
                f"panel veri seti ({sayi(self.n, 0)} gözlem) kullanılmış, model {yontem_metni} tahmin edilmiştir (Baltagi, 2021).",
                f"The analysis uses a {denge} panel of {sayi(self.birim_sayisi, 0)} units and {sayi(self.zaman_sayisi, 0)} periods "
                f"({sayi(self.n, 0)} observations). The model was estimated by {yontem_metni} (Baltagi, 2021).",
            )
            if self.zaman_etkisi:
                y += l(
                    " Ortak şokları ayıklamak için modele zaman etkileri (dönem kuklaları) eklenmiştir.",
                    " Time effects (period dummies) were added to absorb common shocks.",
                )
            y += l(
                f" Standart hatalar: {self.kovaryans_aciklama}. Hesaplar ekonometri-tr {__version__} ile yapılmıştır.",
                f" Standard errors: {_se_metin(self.kovaryans_aciklama)}. The calculations were done with ekonometri-tr {__version__}.",
            )
            r.bloklar += [("baslik", l("Yöntem", "Method")), ("paragraf", y)]
            b, no = _tanimlayici(self, ad, tablo_no)
            r.bloklar += b
            ga = self.guven_araliklari()
            satirlar = [
                [
                    ad.satir(t),
                    tablo_sayi(self.katsayilar[t.sutun]) + yildiz(self.p_degerleri[t.sutun]),
                    tablo_sayi(self.standart_hatalar[t.sutun]),
                    sayi(self.t_degerleri[t.sutun], 2),
                    _p_hucre(self.p_degerleri[t.sutun]),
                    f"[{tablo_sayi(ga.loc[t.sutun, 'alt'])}; {tablo_sayi(ga.loc[t.sutun, 'üst'])}]",
                ]
                for t in self._terimler
            ]
            ad_model = _yontem_ad(self.yontem)
            notu = l(
                f"N = {sayi(self.n, 0)}; birim sayısı = {sayi(self.birim_sayisi, 0)}; R² (içsel) = {sayi(self.r2_ic, 3)}; R² (birimler arası) = {sayi(self.r2_arasi, 3)}; "
                f"R² (genel) = {sayi(self.r2_genel, 3)}; {self.genel_test} ({_p(self.f_p_degeri)}). Standart hatalar: {self.kovaryans_aciklama}."
                + (" Zaman etkileri modele dahildir; katsayıları tabloda gösterilmemiştir." if self.zaman_etkisi else "")
                + " SH: standart hata; GA: güven aralığı. *** p < 0,01; ** p < 0,05; * p < 0,10.",
                f"N = {sayi(self.n, 0)}; units = {sayi(self.birim_sayisi, 0)}; R² (within) = {sayi(self.r2_ic, 3)}; R² (between) = {sayi(self.r2_arasi, 3)}; "
                f"R² (overall) = {sayi(self.r2_genel, 3)}; {self.genel_test} ({_p(self.f_p_degeri)}). Standard errors: {_se_metin(self.kovaryans_aciklama)}."
                + (" Time effects are included; their coefficients are not shown." if self.zaman_etkisi else "")
                + " SE: standard error; CI: confidence interval. *** p < 0.01; ** p < 0.05; * p < 0.10.",
            )
            genel = l(
                f"{self.model_turu} modelinin tahmin sonuçları Tablo {no} ile sunulmuştur. "
                + (
                    f"Açıklayıcı değişkenler birlikte istatistiksel olarak anlamlıdır ({self.genel_test}; {_p(self.f_p_degeri)})."
                    if self.f_p_degeri < 0.05
                    else f"Açıklayıcı değişkenler birlikte istatistiksel olarak anlamlı bulunmamıştır ({self.genel_test}; {_p(self.f_p_degeri)})."
                ),
                f"Estimates of the {ad_model} model are reported in Table {no}. "
                + (
                    f"The regressors are jointly significant ({self.genel_test}; {_p(self.f_p_degeri)})."
                    if self.f_p_degeri < 0.05
                    else f"The regressors are not jointly significant ({self.genel_test}; {_p(self.f_p_degeri)})."
                ),
            )
            r.bloklar += [
                ("baslik", l("Bulgular", "Findings")),
                ("paragraf", genel),
                (
                    "tablo",
                    Tablo(
                        no,
                        l(f"{self.model_turu} modeli tahmin sonuçları", f"{ad_model} estimates"),
                        [l("Değişken", "Variable"), "β", l("SH", "SE"), self.ist_adi, "p", l("%95 GA", "95% CI")],
                        satirlar,
                        notu,
                    ),
                ),
            ]
            r.bloklar += katsayi_paragraflari(self, ad)
            if self.yontem == "sabit":
                r.bloklar.append(
                    (
                        "paragraf",
                        l(
                            "Sabit etkiler katsayıları, birimlerin kendi içindeki zamana bağlı değişimlerinden elde edildiğinden birim içi (within) etkiler olarak yorumlanmalıdır.",
                            "Fixed-effects coefficients are identified from within-unit variation over time and should be read as within effects.",
                        ),
                    )
                )
            r.bloklar.append(
                (
                    "paragraf",
                    l(
                        "Bulgular değişkenler arasındaki ilişkiyi göstermektedir; nedensel bir yorum, araştırma tasarımının buna izin verdiği ölçüde yapılabilir.",
                        "The findings describe associations. A causal reading is warranted only to the extent that the research design supports it.",
                    ),
                )
            )
            r.bloklar += [
                ("baslik", l("Kaynakça", "References")),
                ("liste", sorted(KAYNAKLAR[x] for x in (["baltagi", "swamy"] if self.yontem == "rassal" else ["baltagi"]))),
            ]
            if dosya is not None:
                r.kaydet(dosya)
            return r
        finally:
            DIL.reset(token)

    def yorumla(self):
        """Katsayıların Türkçe yorumu (EKK yorum kurallarıyla) ve panel modeline özgü notlar."""
        from .yorum import yorumla

        y = yorumla(self)
        y.model_satirlari = [
            f"{self.model_turu} modeli: {sayi(self.birim_sayisi, 0)} birim, {sayi(self.zaman_sayisi, 0)} dönem, {sayi(self.n, 0)} gözlem ({'dengeli' if self.dengeli else 'dengesiz'} panel).",
            f"R² içsel = {sayi(self.r2_ic, 3)} (birim içi zamana bağlı değişimin açıklanan kısmı), birimler arası = {sayi(self.r2_arasi, 3)}, genel = {sayi(self.r2_genel, 3)}.",
            f"Standart hatalar: {self.kovaryans_aciklama}.",
        ]
        if self.yontem == "sabit":
            y.notlar.insert(
                0,
                "Sabit etkiler katsayıları, aynı birimin zaman içindeki değişimine dayanır (birim içi etki); zamanla değişmeyen özellikler tahmin edilemez.",
            )
        elif self.yontem == "rassal":
            y.notlar.insert(
                0,
                "Rassal etkiler, gözlenmeyen birim etkisinin açıklayıcı değişkenlerle ilişkisiz olduğunu varsayar; Hausman testiyle kontrol et: ek.panel_analizi(...).",
            )
        y.notlar = [n for n in y.notlar if "model.varsayimlar()" not in n]
        return y


# --------------------------------------------------------------------------- yardımcılar
def _xtset_al(veri, birim, zaman):
    """birim / zaman verilmemişse ek.xtset ile tanımlanan yapıyı kullan (Stata'daki gibi)."""
    if birim is None or zaman is None:
        xt = getattr(veri, "attrs", {}).get("xtset")
        if xt is None:
            raise EkonometriHatasi(
                "Birim ve zaman değişkenlerini ver (birim='firma', zaman='yil') ya da önce veri = ek.xtset(veri, 'firma', 'yil') çalıştır."
            )
        birim = birim or xt[0]
        zaman = zaman or xt[1]
    return birim, zaman


def _grup_ortalama(A: np.ndarray, kod: np.ndarray, G: int) -> np.ndarray:
    """Her gözlem için kendi biriminin ortalaması (satır sayısı korunur)."""
    sayim = np.bincount(kod, minlength=G).astype(float)
    if A.ndim == 1:
        return (np.bincount(kod, weights=A, minlength=G) / sayim)[kod]
    return np.column_stack([(np.bincount(kod, weights=A[:, j], minlength=G) / sayim)[kod] for j in range(A.shape[1])])


def _birim_ortalama(A: np.ndarray, kod: np.ndarray, G: int) -> np.ndarray:
    sayim = np.bincount(kod, minlength=G).astype(float)
    if A.ndim == 1:
        return np.bincount(kod, weights=A, minlength=G) / sayim
    return np.column_stack([np.bincount(kod, weights=A[:, j], minlength=G) / sayim for j in range(A.shape[1])])


def _r2(y, yhat) -> float:
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def _hazirla(formul, veri, birim, zaman, zaman_etkisi, derinlik):
    if not isinstance(veri, pd.DataFrame):
        raise EkonometriHatasi("veri bir pandas DataFrame olmalı.")
    veri = benzersiz_indeks(veri)
    for c, rol in ((birim, "birim"), (zaman, "zaman")):
        if c not in veri.columns:
            raise EkonometriHatasi(f"{rol} sütunu {c!r} veride yok. Mevcut sütunlar: {list(veri.columns)}")
    if veri.duplicated([birim, zaman]).any():
        ornek = veri.loc[veri.duplicated([birim, zaman], keep=False), [birim, zaman]].head(3).values.tolist()
        raise EkonometriHatasi(
            f"Aynı birim–zaman çifti birden fazla kez geçiyor (ör. {ornek}). Her birim için her dönemde tek gözlem olmalı."
        )
    try:
        env = patsy.EvalEnvironment.capture(derinlik).with_outer_namespace(
            {"np": np, "log": np.log, "ln": np.log, "exp": np.exp, "sqrt": np.sqrt}
        )
        y_df, X_df = patsy.dmatrices(formul, veri, return_type="dataframe", NA_action="drop", eval_env=env)
    except patsy.PatsyError as h:
        raise EkonometriHatasi(f"Formül okunamadı: {h}\nMevcut sütunlar: {list(veri.columns)}") from None
    if "Intercept" not in X_df.columns:
        raise EkonometriHatasi("Panel modellerinde sabit terim olmalı (formülden '- 1' / '+ 0' çıkar).")
    idx = y_df.index
    kod, birimler_ = pd.factorize(veri.loc[idx, birim], sort=True)
    zkod, zamanlar = pd.factorize(veri.loc[idx, zaman], sort=True)
    zaman_sut = []
    if zaman_etkisi:
        for j in range(1, len(zamanlar)):
            ad = f"{zaman}[{zamanlar[j]}]"
            X_df[ad] = (zkod == j).astype(float)
            zaman_sut.append(ad)
    kullanilan = [c for c in veri.columns if isinstance(c, str) and re.search(rf"(?<![\w.]){re.escape(c)}(?!\w)", formul)]
    return y_df, X_df, kod, len(birimler_), len(zamanlar), zaman_sut, veri.loc[idx, kullanilan]


def _kov(X, e, XtX_inv, tur, kod, G, sd, carpan="stata_ekk"):
    if tur == "klasik":
        return float(e @ e) / sd * XtX_inv, "klasik"
    if tur == "kume":
        from .regresyon import kume_kovaryansi

        V, _, G = kume_kovaryansi(X, e, XtX_inv, kod, carpan)
        return V, f"birimlere göre kümelenmiş ({G} küme; değişen varyans ve birim içi otokorelasyona dayanıklı)"
    raise EkonometriHatasi(f"Bilinmeyen kovaryans türü: {tur!r}. Seçenekler: 'klasik', 'kume' (birimlere göre kümelenmiş)")


def _kor2(a, b) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def _icsel_r2_ler(y, X, b, kod, G):
    """Stata xtreg tanımlarıyla içsel, birimler arası ve genel R² (korelasyon kareleri)."""
    yhat = X @ b
    y_dm = y - _grup_ortalama(y, kod, G)
    yh_dm = yhat - _grup_ortalama(yhat, kod, G)
    yb, Xb = _birim_ortalama(y, kod, G), _birim_ortalama(X, kod, G)
    return _kor2(y_dm, yh_dm), _kor2(yb, Xb @ b), _kor2(y, yhat)


# --------------------------------------------------------------------------- ana fonksiyon
def panel(
    formul: str,
    veri: pd.DataFrame,
    birim: str | None = None,
    zaman: str | None = None,
    yontem: str = "sabit",
    kovaryans: str = "klasik",
    zaman_etkisi: bool = False,
    birimler: dict[str, str] | None = None,
    _derinlik: int = 1,
) -> PanelSonuc:
    """Panel veri modeli.

    Parametreler
    ------------
    birim, zaman : str
        Birim (firma, il, ülke...) ve zaman (yıl, çeyrek...) sütunlarının adı.
    yontem : {"sabit", "rassal", "havuz"}
        Sabit etkiler (içsel / within), rassal etkiler (Swamy-Arora GLS) ya da havuzlanmış EKK.
    kovaryans : {"klasik", "kume"}
        "kume": birimlere göre kümelenmiş standart hata (değişen varyans ve birim içi otokorelasyona dayanıklı; önerilir).
    zaman_etkisi : bool
        True ise her dönem için kukla eklenir (iki yönlü model); ortak şokları ve trendleri ayıklar.
    """
    birim, zaman = _xtset_al(veri, birim, zaman)
    if yontem not in YONTEMLER:
        raise EkonometriHatasi(f"Bilinmeyen yöntem: {yontem!r}. Seçenekler: 'sabit', 'rassal', 'havuz'")
    if kovaryans not in ("klasik", "kume"):
        raise EkonometriHatasi(f"Bilinmeyen kovaryans türü: {kovaryans!r}. Seçenekler: 'klasik', 'kume'")
    y_df, X_df, kod, G, Tn, zaman_sut, kveri = _hazirla(formul, veri, birim, zaman, zaman_etkisi, _derinlik + 1)
    y = y_df.iloc[:, 0].values.astype(float)
    X_tum = X_df.values.astype(float)
    n = len(y)
    Ti = np.bincount(kod, minlength=G)
    if G < 2:
        raise EkonometriHatasi("Panel modeli için en az 2 birim gerekli.")
    # zamanla değişmeyen (birim içinde sabit) terimler
    X_ic = X_tum - _grup_ortalama(X_tum, kod, G)
    degismeyen = [
        c
        for j, c in enumerate(X_df.columns)
        if c != "Intercept" and np.allclose(X_ic[:, j], 0, atol=1e-10 * (1 + np.abs(X_tum[:, j]).max()))
    ]
    cikarilan: list[str] = []
    s2u = s2e = theta_ort = None

    if yontem == "havuz":
        isimler = list(X_df.columns)
        Z, yz = X_tum, y
        sd = n - Z.shape[1]
    elif yontem == "sabit":
        cikarilan = degismeyen
        isimler = [c for c in X_df.columns if c not in cikarilan]
        sec = [X_df.columns.get_loc(c) for c in isimler]
        Xs = X_tum[:, sec]
        Z = Xs - _grup_ortalama(Xs, kod, G) + Xs.mean(axis=0)
        yz = y - _grup_ortalama(y, kod, G) + y.mean()
        sd = n - Z.shape[1] - (G - 1)
    else:  # rassal — Swamy-Arora
        isimler = list(X_df.columns)
        y_ic = y - _grup_ortalama(y, kod, G)
        e_ic = y_ic - X_ic @ np.linalg.lstsq(X_ic, y_ic, rcond=None)[0]
        rank_ic = np.linalg.matrix_rank(X_ic / np.where(np.linalg.norm(X_tum, axis=0) == 0, 1, np.linalg.norm(X_tum, axis=0)))
        yb, Xb = _birim_ortalama(y, kod, G), _birim_ortalama(X_tum, kod, G)
        bb, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
        rank_ar = np.linalg.matrix_rank(Xb)
        sd_ic, sd_ar = n - G - rank_ic, G - rank_ar
        if sd_ic <= 0 or sd_ar <= 0:
            raise EkonometriHatasi("Rassal etkiler için gözlem/birim sayısı yetersiz (varyans bileşenleri tahmin edilemiyor).")
        s2e = float(e_ic @ e_ic) / sd_ic
        u = yb - Xb @ bb
        T_harm = G / np.sum(1.0 / Ti)
        s2u = max(0.0, float(u @ u) / sd_ar - s2e / T_harm)
        th = 1 - np.sqrt(s2e / (Ti * s2u + s2e))
        theta_ort = float(np.mean(th))
        Z = X_tum - th[kod][:, None] * Xb[kod]
        yz = y - th[kod] * yb[kod]
        sd = n - Z.shape[1]

    if np.linalg.matrix_rank(Z) < Z.shape[1]:
        raise EkonometriHatasi("Tam çoklu doğrusal bağlantı: bazı açıklayıcı değişkenler diğerlerinin doğrusal birleşimi.")
    XtX_inv = np.linalg.inv(Z.T @ Z)
    b = XtX_inv @ Z.T @ yz
    e = yz - Z @ b
    sse = float(e @ e)
    # Stata: xtreg re küme çarpanı G/(G−1) ve z; regress / xtreg fe çarpanı G/(G−1)·(n−1)/(n−k) ve G − 1 sd ile t
    V, kov_ac = _kov(Z, e, XtX_inv, kovaryans, kod, G, sd, carpan="ml" if yontem == "rassal" else "stata_ekk")
    se = np.sqrt(np.diag(V))
    tv = b / se
    cikarim_sd = G - 1 if kovaryans == "kume" and yontem != "rassal" else None
    if yontem == "rassal":
        pv = 2 * stats.norm.sf(np.abs(tv))
    else:
        pv = 2 * stats.t.sf(np.abs(tv), cikarim_sd or sd)

    # R² ölçüleri (orijinal X ve tahmin edilen katsayılarla)
    X_r = X_tum[:, [X_df.columns.get_loc(c) for c in isimler]]
    r2_ic, r2_ar, r2_gn = _icsel_r2_ler(y, X_r, b, kod, G)
    r2_kendi = (
        r2_ic
        if yontem == "sabit"
        else _r2(yz, Z @ b)
        if yontem == "havuz"
        else float(1 - sse / np.sum((yz - Z[:, [0]] @ np.linalg.lstsq(Z[:, [0]], yz, rcond=None)[0]) ** 2))
    )

    # genel anlamlılık: sabit dışındaki katsayılar (Wald / F)
    idx = [i for i, c in enumerate(isimler) if c != "Intercept"]
    q = len(idx)
    if q:
        bi = b[idx]
        W = float(bi @ np.linalg.inv(V[np.ix_(idx, idx)]) @ bi)
        if yontem == "rassal":  # Stata xtreg re: Wald χ²
            F, Fp = W, float(stats.chi2.sf(W, q))
        else:
            F = W / q
            Fp = float(stats.f.sf(F, q, cikarim_sd or sd))
    else:
        F, Fp = float("nan"), float("nan")

    seri = lambda v, ad: pd.Series(v, index=isimler, name=ad)  # noqa: E731
    di = X_df.design_info
    tum_terimler = _terimleri_coz(di)
    terimler = [t for t in tum_terimler if t.sutun in isimler]
    cikarilan_dg = [t.degisken for t in tum_terimler if t.sutun in cikarilan]
    sonuc = PanelSonuc(
        yontem=yontem,
        formul=formul,
        bagimli=y_df.columns[0],
        katsayilar=seri(b, "katsayı"),
        standart_hatalar=seri(se, "std. hata"),
        t_degerleri=seri(tv, "t"),
        p_degerleri=seri(pv, "p"),
        kovaryans_matrisi=pd.DataFrame(V, index=isimler, columns=isimler),
        kovaryans_turu=kovaryans,
        kovaryans_aciklama=kov_ac,
        n=n,
        k=len(isimler),
        birim_sayisi=G,
        zaman_sayisi=Tn,
        sd=sd,
        sse=sse,
        r2=float(r2_kendi),
        r2_ic=r2_ic,
        r2_arasi=r2_ar,
        r2_genel=r2_gn,
        f_istatistigi=F,
        f_p_degeri=Fp,
        f_serbestlik=(q,) if yontem == "rassal" else (q, cikarim_sd or sd),
        kalintilar=pd.Series(y - X_r @ b, index=y_df.index, name="kalıntı"),
        zaman_etkisi=zaman_etkisi,
        dengeli=bool(np.ptp(Ti) == 0),
        cikarilanlar=cikarilan,
        sigma2_u=s2u,
        sigma2_e=s2e,
        theta=theta_ort,
        X=pd.DataFrame(X_r, index=y_df.index, columns=isimler),
        _terimler=terimler,
        _birimler=dict(birimler or {}),
        _veri=kveri,
        cikarim_sd=cikarim_sd,
        _etiketler=dict(veri.attrs.get("etiketler", {})),
        _zaman_sutunlari=zaman_sut,
        _cikarilan_degiskenler=cikarilan_dg,
    )
    sonuc._kod, sonuc._y = kod, y  # testler için
    return sonuc


# --------------------------------------------------------------------------- model seçim testleri
@dataclass
class PanelTesti:
    ad: str
    karsilastirma: str
    h0: str
    istatistik: float
    serbestlik: tuple
    p_degeri: float
    karar: str
    secilen: str
    detay: dict = field(default_factory=dict)

    def __str__(self) -> str:
        sd = ", ".join(str(s) for s in self.serbestlik)
        return f"{self.ad} ({self.karsilastirma}): istatistik = {sayi(self.istatistik, 3)} [{sd}], {p_cumle(self.p_degeri)}\n   H0: {self.h0}\n   {self.karar}"

    __repr__ = __str__


def f_testi(havuz: PanelSonuc, sabit: PanelSonuc) -> PanelTesti:
    """Birim etkilerinin anlamlılığı (havuzlanmış EKK ↔ sabit etkiler). H0: tüm birim etkileri sıfırdır."""
    G = sabit.birim_sayisi
    if sabit.cikarilanlar:
        X = havuz.X.drop(columns=sabit.cikarilanlar).values
        e = havuz._y - X @ np.linalg.lstsq(X, havuz._y, rcond=None)[0]
        sse_h = float(e @ e)
    else:
        sse_h = havuz.sse
    F = ((sse_h - sabit.sse) / (G - 1)) / (sabit.sse / sabit.sd)
    p = float(stats.f.sf(F, G - 1, sabit.sd))
    ret = p < 0.05
    return PanelTesti(
        "F testi",
        "havuzlanmış EKK – sabit etkiler",
        "Birim etkileri yoktur (havuzlanmış EKK uygundur).",
        float(F),
        (G - 1, sabit.sd),
        p,
        "p < 0,05: H0 reddedildi — birim etkileri anlamlı; sabit etkiler havuzlanmış EKK'ye tercih edilir."
        if ret
        else "p ≥ 0,05: H0 reddedilemedi — birim etkileri anlamlı değil; havuzlanmış EKK yeterli.",
        "sabit" if ret else "havuz",
    )


def bp_lm_testi(havuz: PanelSonuc) -> PanelTesti:
    """Breusch-Pagan (1980) LM testi (dengesiz panel için Baltagi-Li biçimi). H0: birim etkilerinin varyansı sıfırdır."""
    e = havuz.kalintilar.values
    kod = havuz._kod
    G = havuz.birim_sayisi
    Ti = np.bincount(kod, minlength=G)
    n = len(e)
    toplam = np.bincount(kod, weights=e, minlength=G)
    LM = n**2 / (2 * (np.sum(Ti.astype(float) ** 2) - n)) * (np.sum(toplam**2) / float(e @ e) - 1) ** 2
    p = float(stats.chi2.sf(LM, 1))
    ret = p < 0.05
    return PanelTesti(
        "Breusch-Pagan LM testi",
        "havuzlanmış EKK – rassal etkiler",
        "Birim etkilerinin varyansı sıfırdır (havuzlanmış EKK uygundur).",
        float(LM),
        (1,),
        p,
        "p < 0,05: H0 reddedildi — rassal birim etkileri var; rassal etkiler havuzlanmış EKK'ye tercih edilir."
        if ret
        else "p ≥ 0,05: H0 reddedilemedi — rassal birim etkisine dair kanıt yok; havuzlanmış EKK yeterli.",
        "rassal" if ret else "havuz",
    )


def hausman(sabit: PanelSonuc, rassal: PanelSonuc, sigmamore: bool | None = None) -> PanelTesti:
    """Hausman (1978) testi (sabit etkiler ↔ rassal etkiler). H0: birim etkileri açıklayıcı değişkenlerle ilişkisizdir.

    sigmamore : None (varsayılan) — kovaryans farkı pozitif tanımlı değilse Stata 'sigmamore' düzeltmesi otomatik uygulanır;
        False — Stata 'hausman fe re' varsayılanı (düzeltme yok, gerekirse genelleştirilmiş ters); True — her zaman sigmamore.
    """
    ortak = [c for c in sabit.katsayilar.index if c != "Intercept" and c in rassal.katsayilar.index]
    if not ortak:
        raise EkonometriHatasi("Hausman testi için iki modelde ortak, zamanla değişen bir açıklayıcı değişken olmalı.")
    d = (sabit.katsayilar[ortak] - rassal.katsayilar[ortak]).values
    Vs = sabit.kovaryans_matrisi.loc[ortak, ortak].values
    Vr = rassal.kovaryans_matrisi.loc[ortak, ortak].values

    def pd_mi(M):
        o = np.linalg.eigvalsh(M)
        return bool(np.all(o > 1e-12 * max(1.0, np.abs(o).max())))

    Vd = Vs - Vr
    duzeltme = False
    otomatik = sigmamore is None and not pd_mi(Vd)
    if (sigmamore or otomatik) and sabit.kovaryans_turu == "klasik" and rassal.kovaryans_turu == "klasik":
        # Stata 'sigmamore': iki kovaryans da etkin tahmincinin (rassal etkiler) hata varyansıyla ölçeklenir
        Vd = Vs * (rassal.sse / rassal.sd) / (sabit.sse / sabit.sd) - Vr
        duzeltme = True
    pozitif = pd_mi(Vd)
    H = float(d @ np.linalg.pinv(Vd) @ d)
    sd = int(np.linalg.matrix_rank(Vd))
    p = float(stats.chi2.sf(H, sd))
    ret = p < 0.05
    karar = (
        "p < 0,05: H0 reddedildi — birim etkileri açıklayıcı değişkenlerle ilişkili; rassal etkiler tutarsız, sabit etkiler tercih edilir."
        if ret
        else "p ≥ 0,05: H0 reddedilemedi — rassal etkiler tutarlı ve daha etkin; rassal etkiler tercih edilir."
    )
    if duzeltme:
        karar += " (Kovaryans farkı pozitif tanımlı olmadığından iki kovaryans matrisi de rassal etkiler hata varyansıyla ölçeklendi — Stata 'sigmamore'.)"
    if not pozitif:
        karar += " (Uyarı: kovaryans farkı pozitif tanımlı değil; genelleştirilmiş ters kullanıldı, sonuç temkinli yorumlanmalı.)"
    return PanelTesti(
        "Hausman testi",
        "sabit etkiler – rassal etkiler",
        "Birim etkileri açıklayıcı değişkenlerle ilişkisizdir (rassal etkiler tutarlıdır).",
        H,
        (sd,),
        p,
        karar,
        "sabit" if ret else "rassal",
        {"sigmamore": duzeltme, "pozitif_tanimli": pozitif},
    )


def xttest0(rassal: PanelSonuc) -> PanelTesti:
    """Rassal etkiler için Breusch-Pagan LM testi, Stata ``xttest0`` biçiminde (tek yönlü χ̄²(01) p-değeri).

    İstatistik, rassal etkiler modelindeki açıklayıcılarla tahmin edilen havuzlanmış EKK kalıntılarından
    Breusch ve Pagan (1980) / Baltagi ve Li (1990) formülüyle hesaplanır. Var(u) ≥ 0 kısıtı nedeniyle test tek yönlüdür:
    kalıntılar birim içi negatif ilişki gösteriyorsa istatistik 0 alınır; p-değeri χ²(1) kuyruğunun yarısıdır.
    """
    if rassal.yontem != "rassal":
        raise EkonometriHatasi("xttest0 rassal etkiler modelinden sonra kullanılır (xtreg ..., re).")
    X = rassal.X.values
    y = rassal._y
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    kod, G = rassal._kod, rassal.birim_sayisi
    Ti = np.bincount(kod, minlength=G).astype(float)
    n = len(e)
    A = float(np.sum(np.bincount(kod, weights=e, minlength=G) ** 2) / (e @ e) - 1)
    LM = n**2 / (2 * (np.sum(Ti**2) - n)) * A**2
    if A < 0:
        LM = 0.0
    p = 1.0 if LM == 0 else float(0.5 * stats.chi2.sf(LM, 1))
    return PanelTesti(
        "Breusch-Pagan LM (xttest0)",
        "havuzlanmış EKK – rassal etkiler",
        "Var(u) = 0 (birim etkisi yoktur).",
        float(LM),
        ("χ̄²(01)",),
        p,
        "p < 0,05: H0 reddedildi — rassal birim etkileri var; rassal etkiler havuzlanmış EKK'ye tercih edilir."
        if p < 0.05
        else "p ≥ 0,05: H0 reddedilemedi — rassal birim etkisine dair kanıt yok.",
        "rassal" if p < 0.05 else "havuz",
    )


# --------------------------------------------------------------------------- tam analiz
@dataclass
class PanelAnalizi:
    """Üç panel modeli, model seçim testleri ve önerilen model."""

    havuz: PanelSonuc
    sabit: PanelSonuc
    rassal: PanelSonuc
    testler: list[PanelTesti]
    onerilen: str
    kovaryans: str
    nihai: PanelSonuc  # önerilen modelin (istenen kovaryansla) tahmini

    @property
    def modeller(self) -> dict[str, PanelSonuc]:
        return {"havuz": self.havuz, "sabit": self.sabit, "rassal": self.rassal}

    def karsilastirma(self):
        from .karsilastirma import karsilastir

        return karsilastir(self.havuz, self.sabit, self.rassal, adlar=["Havuzlanmış EKK", "Sabit etkiler", "Rassal etkiler"])

    def test_tablosu(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Test": t.ad,
                    "Karşılaştırma": t.karsilastirma,
                    "İstatistik": round(t.istatistik, 3),
                    "sd": ", ".join(map(str, t.serbestlik)),
                    "p": round(t.p_degeri, 4),
                    "Tercih": YONTEMLER[t.secilen],
                }
                for t in self.testler
            ]
        )

    def __str__(self) -> str:
        return (
            "📊 Panel veri analizi\n\n"
            + str(self.karsilastirma())
            + "\n\n🔎 Model seçim testleri\n"
            + "\n".join(str(t) for t in self.testler)
            + f"\n\n👉 Önerilen model: {YONTEMLER[self.onerilen]}"
            + (" (birimlere göre kümelenmiş standart hatalarla)" if self.kovaryans == "kume" else "")
        )

    __repr__ = __str__

    def _repr_html_(self) -> str:
        return self.rapor()._repr_html_()

    def rapor(
        self,
        dosya: str | None = None,
        baslik: str = "Panel Veri Analizi",
        etiketler: dict | None = None,
        tablo_no: int = 1,
        dil: str = "tr",
        bicim: str = "apa",
    ):
        """Yöntem + model seçimi + bulgular: tez bölümü olarak (.docx / .md / .html)."""
        from . import __version__
        from .rapor import rapor_ayari

        if dil == "en" and baslik == "Panel Veri Analizi":
            baslik = "Panel data analysis"
        token = rapor_ayari(dil, bicim)
        try:
            return self._rapor_govde(dosya, baslik, etiketler, tablo_no, dil, bicim, __version__)
        finally:
            DIL.reset(token)

    def _rapor_govde(self, dosya, baslik, etiketler, tablo_no, dil, bicim, surum):
        from .rapor import KAYNAKLAR, Rapor, Tablo, _Adlar, _p, _p_hucre, _tanimlayici, katsayi_paragraflari

        ad = _Adlar({**self.nihai._etiketler, **(etiketler or {})})
        m = self.nihai
        r = Rapor(baslik, bicim=bicim, dil=dil)
        en = dil == "en"
        if en:
            bas = (
                f"The analysis uses a {'balanced' if m.dengeli else 'unbalanced'} panel of {sayi(m.birim_sayisi, 0)} units and {sayi(m.zaman_sayisi, 0)} periods "
                f"({sayi(m.n, 0)} observations). The model was estimated by pooled OLS, fixed effects, and random effects (Swamy and Arora, 1972) (Baltagi, 2021). "
                "The F test and the Breusch and Pagan (1980) LM test were used to detect unit effects, and the Hausman (1978) test was used to check whether those effects are related to the regressors."
            )
            if m.zaman_etkisi:
                bas += " Time effects (period dummies) were added to absorb shocks common to every period."
            if self.kovaryans == "kume":
                bas += " In the preferred model, standard errors are clustered by unit, which is robust to heteroskedasticity and within-unit autocorrelation."
            bas += f" The calculations were done with ekonometri-tr {surum}."
        else:
            bas = (
                f"Çalışmada {sayi(m.birim_sayisi, 0)} birim ve {sayi(m.zaman_sayisi, 0)} dönemden oluşan {'dengeli' if m.dengeli else 'dengesiz'} panel veri seti "
                f"({sayi(m.n, 0)} gözlem) kullanılmıştır. Model havuzlanmış EKK, sabit etkiler ve rassal etkiler (Swamy ve Arora, 1972) yöntemleriyle tahmin edilmiştir (Baltagi, 2021). "
                "Uygun modelin seçiminde birim etkilerinin varlığı F testi ve Breusch ve Pagan (1980) LM testiyle, birim etkilerinin açıklayıcı değişkenlerle ilişkisi ise Hausman (1978) testiyle sınanmıştır."
            )
            if m.zaman_etkisi:
                bas += " Tüm dönemleri etkileyen ortak şokları ayıklamak için modellere zaman etkileri (dönem kuklaları) eklenmiştir."
            if self.kovaryans == "kume":
                bas += " Nihai modelde, değişen varyans ve birim içi otokorelasyona karşı birimlere göre kümelenmiş standart hatalar kullanılmıştır."
            bas += f" Hesaplar ekonometri-tr {surum} ile yapılmıştır."
        r.bloklar += [("baslik", l("Yöntem", "Method")), ("paragraf", bas)]

        b, no = _tanimlayici(m, ad, tablo_no)
        r.bloklar += b

        # model seçimi
        ft, lm, ht = self.testler
        if en:
            secim = [
                f"The model-selection tests are reported in Table {no}.",
                f"The F test comparing pooled OLS with fixed effects (F({ft.serbestlik[0]}, {ft.serbestlik[1]}) = {sayi(ft.istatistik, 2)}; {_p(ft.p_degeri)}) "
                + (
                    "indicates that the unit effects are significant."
                    if ft.secilen == "sabit"
                    else "indicates that the unit effects are not significant."
                ),
                f"The Breusch-Pagan LM test (χ²(1) = {sayi(lm.istatistik, 2)}; {_p(lm.p_degeri)}) "
                + ("indicates random unit effects." if lm.secilen == "rassal" else "does not provide evidence of random unit effects."),
                f"The Hausman test comparing fixed and random effects (χ²({ht.serbestlik[0]}) = {sayi(ht.istatistik, 2)}; {_p(ht.p_degeri)}) "
                + (
                    "indicates that the unit effects are related to the regressors, so the random-effects estimator is inconsistent."
                    if ht.secilen == "sabit"
                    else "does not reject the hypothesis that the unit effects are unrelated to the regressors."
                ),
                f"Taken together, the tests support the {_YONTEM_EN[self.onerilen].lower()} model.",
            ]
            test_ad = {
                "F testi": "F test",
                "Breusch-Pagan LM testi": "Breusch-Pagan LM test",
                "Hausman testi": "Hausman test",
            }
            test_kars = {
                "havuzlanmış EKK – sabit etkiler": "pooled OLS – fixed effects",
                "havuzlanmış EKK – rassal etkiler": "pooled OLS – random effects",
                "sabit etkiler – rassal etkiler": "fixed effects – random effects",
            }
            test_satir = [
                [
                    test_ad.get(t.ad, t.ad),
                    test_kars.get(t.karsilastirma, t.karsilastirma),
                    sayi(t.istatistik, 3),
                    _p_hucre(t.p_degeri),
                    _yontem_ad(t.secilen),
                ]
                for t in self.testler
            ]
            test_tablo = Tablo(
                no,
                "Panel model-selection tests",
                ["Test", "Comparison", "Statistic", "p", "Preferred"],
                test_satir,
                "F test H₀: no unit effects. Breusch-Pagan LM H₀: the variance of the unit effects is zero. Hausman H₀: the unit effects are unrelated to the regressors.",
                sol=2,
            )
        else:
            secim = [
                f"Model seçim testlerinin sonuçları Tablo {no} ile sunulmuştur.",
                f"Havuzlanmış EKK ile sabit etkiler modelini karşılaştıran F testi (F({ft.serbestlik[0]}, {ft.serbestlik[1]}) = {sayi(ft.istatistik, 2)}; {_p(ft.p_degeri)}) "
                + (
                    "birim etkilerinin anlamlı olduğunu göstermektedir."
                    if ft.secilen == "sabit"
                    else "birim etkilerinin anlamlı olmadığını göstermektedir."
                ),
                f"Breusch-Pagan LM testi (χ²(1) = {sayi(lm.istatistik, 2)}; {_p(lm.p_degeri)}) "
                + (
                    "rassal birim etkilerinin varlığına işaret etmektedir."
                    if lm.secilen == "rassal"
                    else "rassal birim etkilerine dair kanıt sunmamaktadır."
                ),
                f"Sabit ve rassal etkiler modellerini karşılaştıran Hausman testi (χ²({ht.serbestlik[0]}) = {sayi(ht.istatistik, 2)}; {_p(ht.p_degeri)}) ise "
                + (
                    "birim etkilerinin açıklayıcı değişkenlerle ilişkili olduğunu, dolayısıyla rassal etkiler tahmincisinin tutarsız olduğunu göstermektedir."
                    if ht.secilen == "sabit"
                    else "birim etkilerinin açıklayıcı değişkenlerle ilişkisiz olduğu hipotezinin reddedilemediğini göstermektedir."
                ),
                f"Test sonuçları birlikte değerlendirildiğinde {YONTEMLER[self.onerilen].lower()} modelinin uygun olduğuna karar verilmiştir.",
            ]
            test_tablo = Tablo(
                no,
                "Panel veri model seçim testleri",
                ["Test", "Karşılaştırma", "İstatistik", "p", "Tercih edilen"],
                [[t.ad, t.karsilastirma, sayi(t.istatistik, 3), _p_hucre(t.p_degeri), YONTEMLER[t.secilen]] for t in self.testler],
                "F testi H₀: birim etkileri yoktur. Breusch-Pagan LM H₀: birim etkilerinin varyansı sıfırdır. Hausman H₀: birim etkileri açıklayıcı değişkenlerle ilişkisizdir.",
                sol=2,
            )
        r.bloklar += [("baslik", l("Model Seçimi", "Model selection")), ("paragraf", " ".join(secim)), ("tablo", test_tablo)]
        no += 1

        # karşılaştırmalı tablo + nihai model metni
        k = self.karsilastirma()
        if en:
            k.adlar = ["Pooled OLS", "Fixed effects", "Random effects"]
        tab = k._tablo(etiketler, no)
        tab.baslik = "Panel model estimates" if en else "Panel veri modellerinin tahmin sonuçları"
        cik_ad = list(dict.fromkeys(self.sabit._cikarilan_degiskenler))
        if en:
            cik = (
                f" In the fixed-effects model, {', '.join(ad.metin(c) for c in cik_ad)}, which {'does' if len(cik_ad) == 1 else 'do'} not vary over time, "
                "could not be estimated because the unit effects absorb them."
                if cik_ad
                else ""
            )
            kume = " (with standard errors clustered by unit)" if self.kovaryans == "kume" else ""
            if m.f_p_degeri < 0.05:
                genel = (
                    f"Estimates from the three models are compared in Table {no}.{cik} "
                    f"The findings below interpret the preferred {_YONTEM_EN[self.onerilen].lower()} model{kume}. "
                    f"The regressors are jointly significant ({m.genel_test}; {_p(m.f_p_degeri)})"
                )
            else:
                genel = (
                    f"Estimates from the three models are compared in Table {no}.{cik} "
                    f"In the preferred model the regressors are not jointly significant ({_p(m.f_p_degeri)})"
                )
            genel += f"; within R² is {sayi(m.r2_ic, 3)}, between R² is {sayi(m.r2_arasi, 3)}, and overall R² is {sayi(m.r2_genel, 3)}."
        else:
            cik = (
                f" Sabit etkiler modelinde zamanla değişmeyen {', '.join(ad.metin(c) for c in cik_ad)} {'değişkeni' if len(cik_ad) == 1 else 'değişkenleri'} "
                "birim etkileri tarafından soğurulduğundan tahmin edilememiştir."
                if cik_ad
                else ""
            )
            genel = (
                f"Üç modelin tahmin sonuçları Tablo {no} ile karşılaştırmalı olarak sunulmuştur.{cik} "
                f"Aşağıda, seçilen {YONTEMLER[self.onerilen].lower()} modelinin bulguları yorumlanmaktadır"
                + (" (birimlere göre kümelenmiş standart hatalarla)" if self.kovaryans == "kume" else "")
                + f". Modelde açıklayıcı değişkenler birlikte anlamlıdır ({m.genel_test}; {_p(m.f_p_degeri)})"
                if m.f_p_degeri < 0.05
                else f"Üç modelin tahmin sonuçları Tablo {no} ile karşılaştırmalı olarak sunulmuştur.{cik} Seçilen modelde açıklayıcı değişkenler birlikte anlamlı bulunmamıştır ({_p(m.f_p_degeri)})"
            )
            genel += f"; içsel R² {sayi(m.r2_ic, 3)}, birimler arası R² {sayi(m.r2_arasi, 3)} ve genel R² {sayi(m.r2_genel, 3)} olarak hesaplanmıştır."
        r.bloklar += [("baslik", l("Bulgular", "Findings")), ("paragraf", genel), ("tablo", tab)]
        r.bloklar += katsayi_paragraflari(m, ad)
        if self.onerilen == "sabit":
            r.bloklar.append(
                (
                    "paragraf",
                    l(
                        "Sabit etkiler katsayıları, birimlerin kendi içindeki zamana bağlı değişimlerinden elde edildiğinden birim içi (within) etkiler olarak yorumlanmalıdır.",
                        "Fixed-effects coefficients are identified from within-unit variation over time and should be read as within effects.",
                    ),
                )
            )
        r.bloklar.append(
            (
                "paragraf",
                l(
                    "Bulgular değişkenler arasındaki ilişkiyi göstermektedir; nedensel bir yorum, araştırma tasarımının buna izin verdiği ölçüde yapılabilir.",
                    "The findings describe associations. A causal reading is warranted only to the extent that the research design supports it.",
                ),
            )
        )
        r.bloklar += [
            ("baslik", l("Kaynakça", "References")),
            ("liste", sorted(KAYNAKLAR[x] for x in ("baltagi", "bp80", "hausman", "swamy"))),
        ]
        if dosya is not None:
            r.kaydet(dosya)
        return r


def panel_analizi(
    formul: str,
    veri: pd.DataFrame,
    birim: str | None = None,
    zaman: str | None = None,
    kovaryans: str = "kume",
    zaman_etkisi: bool = False,
    birimler: dict[str, str] | None = None,
) -> PanelAnalizi:
    """Havuzlanmış EKK, sabit etkiler ve rassal etkiler modellerini tahmin eder; F, Breusch-Pagan LM ve Hausman testleriyle
    uygun modeli seçer. Seçim testleri klasik kovaryansla yapılır; seçilen model ``kovaryans`` ile (varsayılan: kümelenmiş) yeniden raporlanır.
    """
    birim, zaman = _xtset_al(veri, birim, zaman)
    ort = {"birim": birim, "zaman": zaman, "zaman_etkisi": zaman_etkisi, "birimler": birimler, "_derinlik": 2}
    h = panel(formul, veri, yontem="havuz", **ort)
    s = panel(formul, veri, yontem="sabit", **ort)
    r = panel(formul, veri, yontem="rassal", **ort)
    testler = [f_testi(h, s), bp_lm_testi(h), hausman(s, r)]
    ft, lm, ht = testler
    if ft.secilen == "havuz" and lm.secilen == "havuz":
        oneri = "havuz"
    elif ft.secilen == "sabit" and lm.secilen == "havuz":
        oneri = "sabit"
    elif ft.secilen == "havuz" and lm.secilen == "rassal":
        oneri = "rassal"
    else:
        oneri = ht.secilen
    if kovaryans == "klasik":
        modeller = {"havuz": h, "sabit": s, "rassal": r}
        nihai = modeller[oneri]
    else:
        h, s, r = (panel(formul, veri, yontem=y, kovaryans=kovaryans, **ort) for y in ("havuz", "sabit", "rassal"))
        nihai = {"havuz": h, "sabit": s, "rassal": r}[oneri]
    return PanelAnalizi(h, s, r, testler, oneri, kovaryans, nihai)


__all__ = ["panel", "panel_analizi", "PanelSonuc", "PanelAnalizi", "PanelTesti", "f_testi", "bp_lm_testi", "hausman", "xttest0"]
