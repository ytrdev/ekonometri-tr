"""En Küçük Kareler (EKK / OLS) regresyonu: tahmin, Türkçe özet tablo ve Türkçe yorum.

Hesaplamalar doğrudan matris cebiriyle yapılır (statsmodels'e bağımlı değildir);
doğruluğu test paketinde statsmodels ile karşılaştırılarak doğrulanır.

Örnek
-----
>>> import ekonometri as ek
>>> veri = ek.veri.yukle("ucret")
>>> model = ek.ekk("np.log(ucret) ~ egitim + tecrube + I(tecrube**2) + cinsiyet", veri)
>>> print(model.yorumla())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import patsy
from scipy import stats

from ._yardimci import anlamlilik_ifadesi, benzersiz_indeks, ekk_coz, p_cumle, p_yaz, sabit_var_mi, sayi, tablo_sayi, yildiz, yuzde

if TYPE_CHECKING:  # yalnızca tip ipuçları için
    from .rapor import Rapor
    from .testler import VarsayimRaporu
    from .yorum import Yorum

KOVARYANS_TURLERI = ("klasik", "HC0", "HC1", "HC2", "HC3", "HAC", "newey", "kume")


class EkonometriHatasi(ValueError):
    """Kullanıcıya Türkçe açıklama veren hata."""


# --------------------------------------------------------------------------- kovaryans
def _kovaryans(X: np.ndarray, e: np.ndarray, XtX_inv: np.ndarray, tur: str, gecikme: int | None) -> tuple[np.ndarray, str]:
    n, k = X.shape
    if tur == "klasik":
        s2 = float(e @ e) / (n - k)
        return s2 * XtX_inv, "klasik (homoskedastik)"
    if tur in ("HC0", "HC1", "HC2", "HC3"):
        h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)  # kaldıraç (hat) değerleri
        if tur in ("HC0", "HC1"):
            w = e**2
        elif tur == "HC2":
            w = e**2 / (1 - h)
        else:
            w = e**2 / (1 - h) ** 2
        orta = (X * w[:, None]).T @ X
        V = XtX_inv @ orta @ XtX_inv
        if tur == "HC1":
            V *= n / (n - k)
        return V, f"değişen varyansa dayanıklı ({tur}, White)"
    if tur == "HAC":
        L = gecikme if gecikme is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
        Xe = X * e[:, None]
        S = Xe.T @ Xe
        for gec in range(1, L + 1):
            w = 1 - gec / (L + 1)  # Bartlett çekirdeği
            G = Xe[gec:].T @ Xe[:-gec]
            S += w * (G + G.T)
        V = XtX_inv @ S @ XtX_inv
        return V, f"değişen varyans ve otokorelasyona dayanıklı (HAC / Newey-West, {L} gecikme)"
    if tur == "newey":  # Stata newey: Bartlett çekirdeği, çarpan n/(n−k); gecikme zorunlu
        if gecikme is None:
            raise EkonometriHatasi("Newey-West için gecikme sayısını ver: gecikme=2 (Stata: newey y x, lag(2)).")
        V, _ = _kovaryans(X, e, XtX_inv, "HAC", gecikme)
        return V * n / (n - k), f"Newey-West ({gecikme} gecikme)"
    raise EkonometriHatasi(f"Bilinmeyen kovaryans türü: {tur!r}. Seçenekler: {', '.join(KOVARYANS_TURLERI)}")


def kume_kovaryansi(
    X: np.ndarray, e: np.ndarray, XtX_inv: np.ndarray, kod: np.ndarray, carpan: str = "stata_ekk"
) -> tuple[np.ndarray, str, int]:
    """Kümelenmiş (cluster-robust) kovaryans. Stata çarpanı: G/(G−1)·(n−1)/(n−k) (EKK, xtreg fe) ya da G/(G−1) (ML, xtreg re)."""
    n, k = X.shape
    G = int(kod.max()) + 1
    if G < 2:
        raise EkonometriHatasi("Kümelenmiş standart hata için en az 2 küme gerekli.")
    Xe = X * e[:, None]
    toplam = np.column_stack([np.bincount(kod, weights=Xe[:, j], minlength=G) for j in range(k)])
    S = toplam.T @ toplam
    c = G / (G - 1) * ((n - 1) / (n - k) if carpan == "stata_ekk" else 1.0)
    return c * XtX_inv @ S @ XtX_inv, f"kümelenmiş, {G} küme", G


# --------------------------------------------------------------------------- terim çözümleme
_LOG_RE = re.compile(r"^(?:np\.)?(?:log|ln)\((.+)\)$")
_KARE_RE = re.compile(r"^I\((.+?)\s*\*\*\s*2\)$")
_KUKLA_RE = re.compile(r"^(?:C\()?(.+?)(?:\))?(?:,.*)?\[T\.(.+)\]$")


def _log_mu(ad: str) -> str | None:
    """'np.log(ucret)' -> 'ucret'; 'log_ucret' / 'ln_ucret' -> 'ucret'; değilse None."""
    m = _LOG_RE.match(ad.strip())
    if m:
        return m.group(1).strip()
    for on in ("log_", "ln_"):
        if ad.startswith(on):
            return ad[len(on) :]
    return None


@dataclass
class _Terim:
    sutun: str  # tasarım matrisindeki sütun adı
    tur: str  # sabit | surekli | log | kare | kukla | etkilesim | diger
    degisken: str  # yalın değişken adı
    seviye: str = ""  # kukla için kategori
    referans: str = ""  # kukla için referans kategori
    parcalar: list = field(default_factory=list)  # etkileşim için bileşen terimler


def _terimleri_coz(design_info: patsy.DesignInfo) -> list[_Terim]:
    referanslar: dict[str, str] = {}
    for faktor, bilgi in design_info.factor_infos.items():
        if bilgi.type == "categorical" and bilgi.categories:
            ad = faktor.name()
            yalin = re.sub(r"^C\((.+?)(?:,.*)?\)$", r"\1", ad).strip()
            referanslar[ad] = str(bilgi.categories[0])
            referanslar[yalin] = str(bilgi.categories[0])

    def tek(ad: str) -> _Terim:
        if ad == "Intercept":
            return _Terim(ad, "sabit", "sabit")
        m = _KUKLA_RE.match(ad)
        if m and "[T." in ad:
            dg = m.group(1).strip()
            return _Terim(ad, "kukla", dg, m.group(2), referanslar.get(dg, referanslar.get(f"C({dg})", "referans grup")))
        m = _KARE_RE.match(ad)
        if m:
            return _Terim(ad, "kare", m.group(1).strip())
        lg = _log_mu(ad)
        if lg:
            return _Terim(ad, "log", lg)
        if re.match(r"^[A-Za-z_çğıöşüÇĞİÖŞÜ][\wçğıöşüÇĞİÖŞÜ]*$", ad):
            return _Terim(ad, "surekli", ad)
        return _Terim(ad, "diger", ad)

    terimler = []
    for ad in design_info.column_names:
        bolumler = _etkilesim_bol(ad)
        if len(bolumler) > 1:
            parcalar = [tek(b) for b in bolumler]
            terimler.append(_Terim(ad, "etkilesim" if all(p.tur != "diger" for p in parcalar) else "diger", ad, parcalar=parcalar))
        else:
            terimler.append(tek(ad))
    return terimler


def _etkilesim_bol(ad: str) -> list[str]:
    """'egitim:cinsiyet[T.Kadın]' -> ['egitim', 'cinsiyet[T.Kadın]'] (köşeli/normal parantez içindeki ':' bölünmez)."""
    parcalar, derinlik, son = [], 0, 0
    for i, c in enumerate(ad):
        if c in "[(":
            derinlik += 1
        elif c in "])":
            derinlik -= 1
        elif c == ":" and derinlik == 0:
            parcalar.append(ad[son:i])
            son = i + 1
    return parcalar + [ad[son:]]


def _terim_gorunumu(t: _Terim) -> str:
    if t.tur == "sabit":
        return "Sabit"
    if t.tur == "kukla":
        return f"{t.degisken}: {t.seviye}"
    if t.tur == "kare":
        return f"{t.degisken}²"
    if t.tur == "log":
        return f"ln({t.degisken})"
    if t.tur == "etkilesim":
        return " × ".join(_terim_gorunumu(p) for p in t.parcalar)
    return t.sutun


def gorunen_ad(sonuc, sutun: str) -> str:
    """Sonuç tablolarında terimin okunur adı: 'C(cinsiyet)[T.Kadın]' → 'cinsiyet: Kadın', 'I(x ** 2)' → 'x²', 'L_x' → 'L.x'."""
    if sutun == "Intercept":
        return "Sabit"
    et = getattr(sonuc, "_etiketler", {}) or {}
    if re.fullmatch(r"[LDF]\d*(?:[LDF]\d*)*\.\S+", str(et.get(sutun, ""))):
        return et[sutun]
    t = next((t for t in getattr(sonuc, "_terimler", []) if t.sutun == sutun), None)
    return _terim_gorunumu(t) if t is not None else sutun


def bagimli_gorunumu(ad: str) -> str:
    lg = _log_mu(ad)
    return f"ln({lg})" if lg else ad


# --------------------------------------------------------------------------- sonuç
@dataclass
class EKKSonuc:
    """EKK tahmin sonuçları. ``ek.ekk(...)`` tarafından döndürülür."""

    formul: str
    bagimli: str
    katsayilar: pd.Series
    standart_hatalar: pd.Series
    t_degerleri: pd.Series
    p_degerleri: pd.Series
    kovaryans_matrisi: pd.DataFrame
    kovaryans_turu: str
    kovaryans_aciklama: str
    r2: float
    duzeltilmis_r2: float
    f_istatistigi: float
    f_p_degeri: float
    f_serbestlik: tuple[int, int]
    n: int
    k: int
    sse: float
    log_olabilirlik: float
    aic: float
    bic: float
    kalintilar: pd.Series
    tahminler: pd.Series
    X: pd.DataFrame = field(repr=False)
    y: pd.Series = field(repr=False)
    _design_info: patsy.DesignInfo = field(repr=False, default=None)
    _terimler: list = field(repr=False, default_factory=list)
    _birimler: dict = field(repr=False, default_factory=dict)
    _sabit: bool = True
    _veri: pd.DataFrame = field(repr=False, default=None)
    cikarim_sd: int | None = None  # t ve F testlerinin payda serbestlik derecesi (kümelenmişte G − 1, Stata gibi)
    _etiketler: dict = field(repr=False, default_factory=dict)
    _zaman_serisi: bool = False  # veri tsset ile tanımlandıysa otokorelasyon testleri otomatik yapılır
    agirlik: tuple | None = None  # (tür, değişken), ör. ("pw", "agirlik")
    _agirlik_vektor: np.ndarray | None = field(repr=False, default=None)

    model_turu = "EKK"
    ist_adi = "t"

    def _uyum(self) -> dict[str, str]:
        """Karşılaştırma tablosunun alt satırları."""
        return {
            "Gözlem sayısı": sayi(self.n, 0),
            "R²": sayi(self.r2, 3),
            "Düzeltilmiş R²": sayi(self.duzeltilmis_r2, 3),
            "F": sayi(self.f_istatistigi, 2) + yildiz(self.f_p_degeri),
            "AIC": sayi(self.aic, 1),
            "BIC": sayi(self.bic, 1),
        }

    # ---- türetilmiş
    @property
    def sd(self) -> int:
        """Kalıntı serbestlik derecesi (n - k)."""
        return self.n - self.k

    @property
    def regresyon_std_hatasi(self) -> float:
        return float(np.sqrt(self.sse / self.sd))

    def guven_araliklari(self, guven: float = 0.95) -> pd.DataFrame:
        """Katsayıların güven aralıkları (varsayılan %95)."""
        t = stats.t.ppf(0.5 + guven / 2, self.cikarim_sd or self.sd)
        alt = self.katsayilar - t * self.standart_hatalar
        ust = self.katsayilar + t * self.standart_hatalar
        return pd.DataFrame({"alt": alt, "üst": ust})

    def tahmin(self, yeni_veri: pd.DataFrame) -> pd.Series:
        """Yeni gözlemler için tahmin (formüldeki dönüşümler otomatik uygulanır)."""
        (Xn,) = patsy.build_design_matrices([self._design_info], yeni_veri, return_type="dataframe")
        return pd.Series(Xn.values @ self.katsayilar.values, index=Xn.index, name=f"tahmin({self.bagimli})")

    def wald_testi(self, kisit: str | list[str]) -> WaldSonuc:
        """Doğrusal kısıt testi. Örnek: ``model.wald_testi("egitim = 0.1")`` ya da ``["x1", "x2"]`` (ortak sıfır)."""
        if isinstance(kisit, (list, tuple)):
            kisit = ", ".join(f"{k} = 0" for k in kisit)
        dc = self._design_info.linear_constraint(kisit)
        R, q = dc.coefs, dc.constants.ravel()
        b = self.katsayilar.values
        V = self.kovaryans_matrisi.values
        fark = R @ b - q
        orta = np.linalg.inv(R @ V @ R.T)
        m = R.shape[0]
        F = float(fark @ orta @ fark) / m
        sd = self.cikarim_sd or self.sd
        p = float(stats.f.sf(F, m, sd))
        return WaldSonuc(kisit, F, p, (m, sd))

    # ---- çıktılar
    def ozet(self) -> str:
        """Türkçe özet tablo (metin)."""
        g = 78
        satirlar = [
            "═" * g,
            f" EKK Regresyon Sonuçları — bağımlı değişken: {bagimli_gorunumu(self.bagimli)}",
            "═" * g,
            f" Gözlem sayısı: {self.n:<12} R²: {sayi(self.r2, 4):<12} Düzeltilmiş R²: {sayi(self.duzeltilmis_r2, 4)}",
            f" F({self.f_serbestlik[0]}, {self.f_serbestlik[1]}): {sayi(self.f_istatistigi, 2):<10} p: {p_yaz(self.f_p_degeri):<12} AIC: {sayi(self.aic, 1):<10} BIC: {sayi(self.bic, 1)}",
            f" Standart hatalar: {self.kovaryans_aciklama}",
            "─" * g,
            f" {'Değişken':<28}{'Katsayı':>12}{'Std. hata':>12}{'t':>9}{'p':>10}",
            "─" * g,
        ]
        for ad in self.katsayilar.index:
            gorunen = gorunen_ad(self, ad)
            if len(gorunen) > 27:
                gorunen = gorunen[:26] + "…"
            p = self.p_degerleri[ad]
            satirlar.append(
                f" {gorunen:<28}{tablo_sayi(self.katsayilar[ad]):>12}{tablo_sayi(self.standart_hatalar[ad]):>12}"
                f"{sayi(self.t_degerleri[ad], 2):>9}{p_yaz(p):>10} {yildiz(p)}"
            )
        satirlar += ["─" * g, " Anlamlılık: *** %1  ** %5  * %10        Yorum için: model.yorumla()", "═" * g]
        return "\n".join(satirlar)

    def __str__(self) -> str:
        return self.ozet()

    def __repr__(self) -> str:
        return self.ozet()

    def _repr_html_(self) -> str:
        satirlar = "".join(
            f"<tr><td style='text-align:left'>{gorunen_ad(self, ad)}</td><td>{tablo_sayi(self.katsayilar[ad])}</td>"
            f"<td>{tablo_sayi(self.standart_hatalar[ad])}</td><td>{sayi(self.t_degerleri[ad], 2)}</td>"
            f"<td>{p_yaz(self.p_degerleri[ad])}</td><td>{yildiz(self.p_degerleri[ad])}</td></tr>"
            for ad in self.katsayilar.index
        )
        return (
            f"<div><b>EKK Regresyon Sonuçları</b> — bağımlı değişken: <code>{bagimli_gorunumu(self.bagimli)}</code><br>"
            f"Gözlem: {self.n} · R²: {sayi(self.r2, 4)} · Düzeltilmiş R²: {sayi(self.duzeltilmis_r2, 4)} · "
            f"F({self.f_serbestlik[0]}, {self.f_serbestlik[1]}) = {sayi(self.f_istatistigi, 2)} (p {p_yaz(self.f_p_degeri)})<br>"
            f"<small>Standart hatalar: {self.kovaryans_aciklama}</small>"
            "<table><thead><tr><th style='text-align:left'>Değişken</th><th>Katsayı</th><th>Std. hata</th><th>t</th><th>p</th><th></th></tr></thead>"
            f"<tbody>{satirlar}</tbody></table><small>*** %1 · ** %5 · * %10 — yorum için <code>model.yorumla()</code></small></div>"
        )

    def yorumla(self) -> Yorum:
        """Katsayıların ve modelin Türkçe yorumu."""
        from .yorum import yorumla

        return yorumla(self)

    def varsayimlar(self, zaman_serisi: bool | None = None, gecikme: int | None = None) -> VarsayimRaporu:
        """Klasik doğrusal regresyon varsayımlarını test eder ve Türkçe öneriler verir."""
        from .testler import varsayim_raporu

        if self.agirlik is not None:
            raise EkonometriHatasi(
                "Ağırlıklı tahminden sonra klasik varsayım testleri uygun değildir (Stata da estat hettest / ovtest için izin vermez). "
                "Ağırlıksız modeli test et ya da dayanıklı standart hata kullan."
            )
        return varsayim_raporu(self, zaman_serisi=zaman_serisi, gecikme=gecikme)

    def rapor(
        self,
        dosya: str | None = None,
        baslik: str = "Bulgular",
        etiketler: dict[str, str] | None = None,
        tablo_no: int = 1,
        yontem: bool = True,
        kaynakca: bool = True,
        zaman_serisi: bool | None = None,
        dil: str = "tr",
        bicim: str = "apa",
    ) -> Rapor:
        """Tez / makale / ödev için Türkçe bulgu raporu: Yöntem + Bulgular metni ve tez formatında tablolar.

        Parametreler
        ------------
        dosya : str, isteğe bağlı
            Verilirse rapor dosyaya yazılır; biçim uzantıdan anlaşılır: ``.docx`` (Word), ``.md``, ``.html``.
        baslik : str
            Rapor başlığı (varsayılan "Bulgular").
        etiketler : dict, isteğe bağlı
            Değişkenlerin metinde ve tablolarda görünecek adları, ör. ``{"egitim": "Eğitim süresi"}``.
        tablo_no : int
            İlk tablonun numarası (tezinde önceki tablolar varsa).
        yontem, kaynakca : bool
            Yöntem bölümü taslağı ve kaynakça eklensin mi?

        Rapor, modelin hesapladığı sayıları cümleye döker. Metni okuyup kendi çalışmana göre düzeltmek gerekir.
        """
        from .rapor import rapor_olustur

        r = rapor_olustur(
            self,
            baslik=baslik,
            etiketler=etiketler,
            tablo_no=tablo_no,
            yontem=yontem,
            kaynakca=kaynakca,
            zaman_serisi=zaman_serisi,
            dil=dil,
            bicim=bicim,
        )
        if dosya is not None:
            r.kaydet(dosya)
        return r


@dataclass
class WaldSonuc:
    kisit: str
    f_istatistigi: float
    p_degeri: float
    serbestlik: tuple[int, int]

    def __str__(self) -> str:
        karar = "reddedilir" if self.p_degeri < 0.05 else "reddedilemez"
        return (
            f"Wald testi — H0: {self.kisit}\n"
            f"F({self.serbestlik[0]}, {self.serbestlik[1]}) = {sayi(self.f_istatistigi, 3)}, {p_cumle(self.p_degeri)}\n"
            f"Sonuç: %5 düzeyinde H0 {karar}."
        )

    __repr__ = __str__


# --------------------------------------------------------------------------- ana fonksiyon
def ekk(
    formul: str,
    veri: pd.DataFrame,
    kovaryans: str = "klasik",
    gecikme: int | None = None,
    birimler: dict[str, str] | None = None,
    kume: str | None = None,
    agirlik: str | None = None,
    agirlik_turu: str = "aw",
) -> EKKSonuc:
    """En Küçük Kareler (EKK / OLS) regresyonu.

    Parametreler
    ------------
    formul : str
        R/statsmodels tarzı formül, ör. ``"np.log(ucret) ~ egitim + tecrube + I(tecrube**2) + C(bolge)"``.
        Metin (kategorik) sütunlar otomatik olarak kukla değişkene çevrilir.
    veri : pandas.DataFrame
    kovaryans : {"klasik", "HC0", "HC1", "HC2", "HC3", "HAC"}
        Standart hata türü. Değişen varyans şüphesinde ``"HC1"`` (Stata'nın ``robust``),
        zaman serisinde otokorelasyon şüphesinde ``"HAC"`` (Newey-West).
    gecikme : int, isteğe bağlı
        HAC için gecikme sayısı (varsayılan: 4·(n/100)^(2/9)).
    birimler : dict, isteğe bağlı
        Yorumlarda kullanılacak birimler, ör. ``{"ucret": "TL", "egitim": "yıl"}``.
    kume : str, isteğe bağlı
        Kümelenmiş standart hata için küme sütunu (``kovaryans="kume"``); Stata ``vce(cluster kume)`` ile aynıdır:
        çarpan G/(G−1)·(n−1)/(n−k), t ve F testlerinde G − 1 serbestlik derecesi.
    agirlik, agirlik_turu : ağırlık sütunu ve türü — Stata ``[aw=]`` (analitik, varsayılan), ``[fw=]`` (frekans),
        ``[pw=]`` (örneklem / olasılık ağırlığı; standart hatalar otomatik olarak dayanıklıdır).
    """
    if kume is not None and kovaryans == "klasik":
        kovaryans = "kume"
    if kovaryans == "kume" and kume is None:
        raise EkonometriHatasi("Kümelenmiş standart hata için küme sütununu ver: ek.ekk(..., kovaryans='kume', kume='firma').")
    if kovaryans not in KOVARYANS_TURLERI:
        raise EkonometriHatasi(f"Bilinmeyen kovaryans türü: {kovaryans!r}. Seçenekler: {', '.join(KOVARYANS_TURLERI)}")
    if not isinstance(veri, pd.DataFrame):
        raise EkonometriHatasi("veri bir pandas DataFrame olmalı (ör. pd.read_csv(...) ya da ek.veri.yukle(...)).")
    veri = benzersiz_indeks(veri)
    try:
        # çağıranın değişkenleri + kısa yollar: log(x) / ln(x) / exp(x) / np.*
        env = patsy.EvalEnvironment.capture(1).with_outer_namespace({"np": np, "log": np.log, "ln": np.log, "exp": np.exp, "sqrt": np.sqrt})
        y_df, X_df = patsy.dmatrices(formul, veri, return_type="dataframe", NA_action="drop", eval_env=env)
    except patsy.PatsyError as h:
        raise EkonometriHatasi(
            f"Formül okunamadı: {h}\nSütun adlarını ve yazımı kontrol et. Mevcut sütunlar: {list(veri.columns)}"
        ) from None
    if y_df.shape[1] != 1:
        raise EkonometriHatasi(
            "Bağımlı değişken tek bir sayısal sütun olmalı (kategorik bağımlı değişken için ileride logit/probit gelecek)."
        )

    di = X_df.design_info  # satır seçimi design_info'yu düşürür; önceden sakla
    w = None
    if agirlik is not None:
        if agirlik_turu not in ("aw", "fw", "pw"):
            raise EkonometriHatasi("Ağırlık türü 'aw', 'fw' ya da 'pw' olmalı.")
        if agirlik not in veri.columns:
            raise EkonometriHatasi(f"Ağırlık değişkeni {agirlik!r} veride yok.")
        w_tum = pd.to_numeric(veri.loc[y_df.index, agirlik], errors="coerce").astype(float)
        if (w_tum < 0).any():
            raise EkonometriHatasi("Ağırlıklar negatif olamaz.")
        gecerli = w_tum.notna() & (w_tum > 0)  # Stata: eksik ya da sıfır ağırlıklı gözlemler çıkarılır
        y_df, X_df, w_tum = y_df.loc[gecerli], X_df.loc[gecerli], w_tum.loc[gecerli]
        w = w_tum.values
        if agirlik_turu == "fw" and not np.allclose(w, np.round(w)):
            raise EkonometriHatasi("Frekans ağırlıkları (fw) tam sayı olmalı.")
        if agirlik_turu == "fw" and kovaryans in ("HC2", "HC3", "HAC", "newey", "kume"):
            raise EkonometriHatasi("Frekans ağırlıklarıyla yalnızca klasik, HC0 ve HC1 standart hatalar desteklenir.")
        if agirlik_turu in ("aw", "pw") and kovaryans in ("HAC", "newey"):
            raise EkonometriHatasi("Ağırlıklı tahminde HAC / Newey-West desteklenmez.")
        if agirlik_turu == "pw" and kovaryans == "klasik":
            kovaryans = "HC1"  # Stata: pweight → vce(robust)
    X = X_df.values.astype(float)
    y = y_df.iloc[:, 0].values.astype(float)
    n, k = X.shape
    if n <= k:
        raise EkonometriHatasi(f"Gözlem sayısı ({n}) parametre sayısından ({k}) fazla olmalı.")
    N = n  # Stata'daki gözlem sayısı (fw'de ağırlıklar toplamı)
    if w is not None:
        if agirlik_turu == "aw":
            w = w * n / w.sum()  # Stata: analitik ağırlıklar gözlem sayısına normalleştirilir
        if agirlik_turu == "fw":
            N = int(round(w.sum()))
        kok = np.sqrt(w)
        Xw, yw = X * kok[:, None], y * kok
    else:
        Xw, yw = X, y
    b, ew, rank = ekk_coz(Xw, yw)
    e = y - X @ b
    if rank < k:
        raise EkonometriHatasi(
            "Tam çoklu doğrusal bağlantı: bazı açıklayıcı değişkenler diğerlerinin doğrusal birleşimi.\n"
            "Sık neden: kukla değişken tuzağı (tüm kategoriler + sabit terim) ya da aynı bilginin iki kez girilmesi."
        )
    XtX_inv = np.linalg.inv(Xw.T @ Xw)
    cikarim_sd = None
    if kovaryans == "kume":
        if kume not in veri.columns:
            raise EkonometriHatasi(f"Küme sütunu {kume!r} veride yok.")
        kod = pd.factorize(veri.loc[y_df.index, kume])[0]
        if (kod < 0).any():
            raise EkonometriHatasi(f"Küme sütunu {kume!r} eksik değer içeriyor.")
        V, aciklama, G = kume_kovaryansi(Xw, ew, XtX_inv, kod)
        aciklama += f" ({kume})"
        cikarim_sd = G - 1
    else:
        if N != n and kovaryans in ("HC0", "HC1"):  # fw: orta matris Σ w·e²·xx' (gözlem çoğaltmakla aynı)
            V, aciklama = _kovaryans(Xw, e, XtX_inv, kovaryans, gecikme)
        else:
            V, aciklama = _kovaryans(Xw, ew, XtX_inv, kovaryans, gecikme)
        if N != n:  # frekans ağırlığı: serbestlik derecesi ağırlıklar toplamıyla
            if kovaryans == "klasik":
                V = V * (n - k) / (N - k)
            elif kovaryans == "HC1":
                V = V * (N / (N - k)) / (n / (n - k))
    if w is not None:
        aciklama += f"; ağırlık: [{agirlik_turu}={agirlik}]"
    se = np.sqrt(np.diag(V))
    t = b / se
    sd = N - k
    p = 2 * stats.t.sf(np.abs(t), cikarim_sd or sd)

    sabit = sabit_var_mi(X)
    sse = float(ew @ ew)
    if w is None:
        sst = float(np.sum((y - y.mean()) ** 2)) if sabit else float(y @ y)
    else:
        ybar = float(np.sum(w * y) / np.sum(w))
        sst = float(np.sum(w * (y - ybar) ** 2)) if sabit else float(np.sum(w * y**2))
    r2 = 1 - sse / sst
    dr2 = 1 - (1 - r2) * (N - 1 if sabit else N) / sd

    # genel anlamlılık (sabit hariç tüm katsayılar = 0)
    idx = [i for i, ad in enumerate(X_df.columns) if ad != "Intercept"] if sabit else list(range(k))
    q = len(idx)
    if q > 0:
        if kovaryans == "klasik" and sabit:
            F = ((sst - sse) / q) / (sse / sd)
        else:
            bi = b[idx]
            F = float(bi @ np.linalg.inv(V[np.ix_(idx, idx)]) @ bi) / q
        Fp = float(stats.f.sf(F, q, cikarim_sd or sd))
    else:
        F, Fp = float("nan"), float("nan")

    ll = -N / 2 * (np.log(2 * np.pi) + np.log(sse / N) + 1)
    aic = -2 * ll + 2 * k
    bic = -2 * ll + np.log(N) * k

    isimler = list(X_df.columns)
    kullanilan = [c for c in veri.columns if isinstance(c, str) and re.search(rf"(?<![\w.]){re.escape(c)}(?!\w)", formul)]
    seri = lambda v, ad=None: pd.Series(v, index=isimler, name=ad)  # noqa: E731
    return EKKSonuc(
        formul=formul,
        bagimli=y_df.columns[0],
        katsayilar=seri(b, "katsayı"),
        standart_hatalar=seri(se, "std. hata"),
        t_degerleri=seri(t, "t"),
        p_degerleri=seri(p, "p"),
        kovaryans_matrisi=pd.DataFrame(V, index=isimler, columns=isimler),
        kovaryans_turu=kovaryans,
        kovaryans_aciklama=aciklama,
        r2=float(r2),
        duzeltilmis_r2=float(dr2),
        f_istatistigi=float(F),
        f_p_degeri=Fp,
        f_serbestlik=(q, cikarim_sd or sd),
        n=N,
        k=k,
        sse=sse,
        log_olabilirlik=float(ll),
        aic=float(aic),
        bic=float(bic),
        kalintilar=pd.Series(e, index=y_df.index, name="kalıntı"),
        tahminler=pd.Series(X @ b, index=y_df.index, name="tahmin"),
        X=X_df,
        y=y_df.iloc[:, 0],
        _design_info=di,
        _terimler=_terimleri_coz(di),
        _birimler=dict(birimler or {}),
        _sabit=sabit,
        _veri=veri.loc[y_df.index, kullanilan],
        cikarim_sd=cikarim_sd,
        _etiketler=dict(veri.attrs.get("etiketler", {})),
        _zaman_serisi="tsset" in veri.attrs,
        agirlik=(agirlik_turu, agirlik) if w is not None else None,
        _agirlik_vektor=w,
    )


__all__ = ["ekk", "EKKSonuc", "WaldSonuc", "EkonometriHatasi", "KOVARYANS_TURLERI", "yuzde", "anlamlilik_ifadesi"]
