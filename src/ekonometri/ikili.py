"""İki değerli (0/1) bağımlı değişken modelleri: logit ve probit.

Tahmin en çok olabilirlik (Newton-Raphson) ile yapılır; yorumlar ortalama marjinal etkilere (AME) dayanır.
Marjinal etkiler ham değişken üzerinden hesaplanır: kare, logaritma ve etkileşim terimleri doğru biçimde hesaba katılır.

>>> veri = ek.veri.yukle("sinav")
>>> model = ek.logit("gecti ~ calisma_saati + devam + burs", veri)
>>> model.marjinal_etkiler()
>>> model.yorumla()
>>> model.rapor("bulgular.docx")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import patsy
from scipy import stats

from ._yardimci import DIL, anlamlilik_ifadesi, benzersiz_indeks, p_cumle, p_yaz, sayi, tablo_sayi, yildiz, yuzde
from .regresyon import EkonometriHatasi, _terimleri_coz, bagimli_gorunumu, gorunen_ad

_ALT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _F(xb: np.ndarray, tur: str) -> np.ndarray:
    return 1 / (1 + np.exp(-xb)) if tur == "logit" else stats.norm.cdf(xb)


def _f(xb: np.ndarray, tur: str) -> np.ndarray:
    if tur == "logit":
        p = 1 / (1 + np.exp(-xb))
        return p * (1 - p)
    return stats.norm.pdf(xb)


def _ll(y, xb, tur, ag=None) -> float:
    ag = 1.0 if ag is None else ag
    if tur == "logit":
        return float(np.sum(ag * (y * xb - np.logaddexp(0, xb))))
    return float(np.sum(ag * (y * stats.norm.logcdf(xb) + (1 - y) * stats.norm.logcdf(-xb))))


def _skor_hessian(X, y, b, tur, ag=None):
    xb = X @ b
    if tur == "logit":
        p = _F(xb, tur)
        s = y - p
        w = p * (1 - p)
    else:
        lam = np.where(
            y == 1, np.exp(stats.norm.logpdf(xb) - stats.norm.logcdf(xb)), -np.exp(stats.norm.logpdf(xb) - stats.norm.logcdf(-xb))
        )
        s = lam
        w = lam * (lam + xb)
    if ag is not None:
        s, w = s * ag, w * ag
    return X * s[:, None], -(X * w[:, None]).T @ X


@dataclass
class IkiliSonuc:
    """Logit / probit tahmin sonuçları. ``ek.logit(...)`` ve ``ek.probit(...)`` tarafından döndürülür."""

    tur: str  # logit | probit
    formul: str
    bagimli: str
    katsayilar: pd.Series
    standart_hatalar: pd.Series
    z_degerleri: pd.Series
    p_degerleri: pd.Series
    kovaryans_matrisi: pd.DataFrame
    kovaryans_turu: str
    kovaryans_aciklama: str
    n: int
    k: int
    log_olabilirlik: float
    log_olabilirlik_sifir: float
    pseudo_r2: float
    lr_istatistigi: float
    lr_p_degeri: float
    aic: float
    bic: float
    olasiliklar: pd.Series
    yineleme: int
    wald_istatistigi: float = float("nan")
    wald_p_degeri: float = float("nan")
    X: pd.DataFrame = field(repr=False, default=None)
    y: pd.Series = field(repr=False, default=None)
    _design_info: patsy.DesignInfo = field(repr=False, default=None)
    _terimler: list = field(repr=False, default_factory=list)
    _birimler: dict = field(repr=False, default_factory=dict)
    _veri: pd.DataFrame = field(repr=False, default=None)
    _ame: pd.DataFrame = field(repr=False, default=None)
    _etiketler: dict = field(repr=False, default_factory=dict)
    _agirliklar: np.ndarray | None = field(repr=False, default=None)

    ist_adi = "z"

    @property
    def model_turu(self) -> str:
        return "Logit" if self.tur == "logit" else "Probit"

    def _uyum(self) -> dict[str, str]:
        return {
            "Gözlem sayısı": sayi(self.n, 0),
            "McFadden sözde R²": sayi(self.pseudo_r2, 3),
            "LR χ²": sayi(self.lr_istatistigi, 2) + yildiz(self.lr_p_degeri),
            "Log-olabilirlik": sayi(self.log_olabilirlik, 1),
            "AIC": sayi(self.aic, 1),
            "BIC": sayi(self.bic, 1),
        }

    # ---- türetilmiş sonuçlar
    def guven_araliklari(self, guven: float = 0.95) -> pd.DataFrame:
        z = stats.norm.ppf(0.5 + guven / 2)
        return pd.DataFrame({"alt": self.katsayilar - z * self.standart_hatalar, "üst": self.katsayilar + z * self.standart_hatalar})

    def odds_oranlari(self, guven: float = 0.95) -> pd.DataFrame:
        """Odds (bahis) oranları: e^β ve güven aralığı. Yalnızca logit için anlamlıdır."""
        if self.tur != "logit":
            raise EkonometriHatasi("Odds oranı yalnızca logit modelinde tanımlıdır; probit için marjinal_etkiler() kullan.")
        ga = self.guven_araliklari(guven)
        return pd.DataFrame({"odds oranı": np.exp(self.katsayilar), "alt": np.exp(ga["alt"]), "üst": np.exp(ga["üst"])}).drop(
            index=[i for i in self.katsayilar.index if i == "Intercept"]
        )

    def tahmin(self, yeni_veri: pd.DataFrame) -> pd.Series:
        """Yeni gözlemler için P(y = 1) olasılık tahmini."""
        (Xn,) = patsy.build_design_matrices([self._design_info], yeni_veri, return_type="dataframe")
        return pd.Series(_F(Xn.values @ self.katsayilar.values, self.tur), index=Xn.index, name=f"P({self.bagimli} = 1)")

    def siniflandirma(self, esik: float = 0.5) -> pd.DataFrame:
        """Sınıflandırma (karışıklık) tablosu. ``.attrs`` içinde doğruluk, duyarlılık ve özgüllük bulunur."""
        t = (self.olasiliklar.values >= esik).astype(int)
        g = self.y.values.astype(int)
        tablo = pd.DataFrame(
            [[int(np.sum((g == i) & (t == j))) for j in (0, 1)] for i in (0, 1)],
            index=pd.Index(["Gerçek: 0", "Gerçek: 1"]),
            columns=["Tahmin: 0", "Tahmin: 1"],
        )
        dp, yn = tablo.iloc[1, 1], tablo.iloc[1, 0]
        dn, yp = tablo.iloc[0, 0], tablo.iloc[0, 1]
        tablo.attrs = {
            "dogruluk": (dp + dn) / len(g),
            "duyarlilik": dp / (dp + yn) if dp + yn else float("nan"),
            "ozgulluk": dn / (dn + yp) if dn + yp else float("nan"),
            "pozitif_ongoru": dp / (dp + yp) if dp + yp else float("nan"),  # Stata: Positive predictive value
            "negatif_ongoru": dn / (dn + yn) if dn + yn else float("nan"),
            "esik": esik,
        }
        return tablo

    def roc(self) -> pd.DataFrame:
        """ROC eğrisi noktaları ve eğri altındaki alan (Stata ``lroc``). Alan ``.attrs['auc']`` içindedir.

        Alan, Mann-Whitney istatistiğine eşittir: rastgele seçilen bir 1 gözleminin tahmin edilen olasılığının
        rastgele bir 0 gözlemininkinden büyük olma olasılığı (eşitlikler yarım sayılır).
        """
        p = self.olasiliklar.values.astype(float)
        g = self.y.values.astype(int)
        esikler = np.r_[np.inf, np.sort(np.unique(p))[::-1]]
        n1, n0 = int(g.sum()), int((1 - g).sum())
        duy = np.array([np.sum((p >= c) & (g == 1)) / n1 for c in esikler])
        yanlis = np.array([np.sum((p >= c) & (g == 0)) / n0 for c in esikler])
        sira = stats.rankdata(p)
        auc = (sira[g == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        t = pd.DataFrame({"Eşik": esikler, "Duyarlılık": duy, "1 − özgüllük": yanlis})
        t.attrs["auc"] = float(auc)
        return t

    def hosmer_lemeshow(self, gruplar: int = 10) -> dict:
        """Hosmer-Lemeshow uyum iyiliği testi (Stata ``estat gof, group(10)``). H0: model veriye uyumludur.

        Gözlemler tahmin edilen olasılığın yüzdeliklerine göre gruplanır (Stata ``xtile`` gibi, eşit olasılıklar aynı grupta),
        χ² = Σ (gözlenen₁ − beklenen₁)² / [beklenen₁ · (1 − beklenen₁ / nᵍ)], serbestlik derecesi grup sayısı − 2.
        """
        from .karsilastirma import stata_yuzdelik

        p = self.olasiliklar.values.astype(float)
        g = self.y.values.astype(float)
        kesimler = sorted({stata_yuzdelik(p, 100 * j / gruplar) for j in range(1, gruplar)})
        kod = np.searchsorted(np.asarray(kesimler), p, side="left")
        satirlar, ki = [], 0.0
        for k in np.unique(kod):
            m = kod == k
            n_g, o1, e1 = int(m.sum()), float(g[m].sum()), float(p[m].sum())
            ki += (o1 - e1) ** 2 / (e1 * (1 - e1 / n_g))
            satirlar.append(
                {
                    "Grup": int(k) + 1,
                    "Üst olasılık": float(p[m].max()),
                    "Gözlenen 1": o1,
                    "Beklenen 1": e1,
                    "Gözlenen 0": n_g - o1,
                    "Beklenen 0": n_g - e1,
                    "N": n_g,
                }
            )
        sd = len(satirlar) - 2
        return {"ki_kare": float(ki), "sd": sd, "p": float(stats.chi2.sf(ki, sd)), "gruplar": pd.DataFrame(satirlar)}

    def marjinal_etkiler(self) -> pd.DataFrame:
        """Ortalama marjinal etkiler (AME) ve delta yöntemiyle standart hataları.

        Sürekli değişkende: değişken 1 birim artınca P(y = 1) ortalama kaç puan değişir (türev).
        Kategorik / 0-1 değişkende: herkes o grupta olsaydı ile herkes referans grupta olsaydı arasındaki ortalama olasılık farkı.
        """
        if self._ame is None:
            self._ame = _marjinal_etkiler(self)
        return self._ame.copy()

    # ---- çıktılar
    def ozet(self) -> str:
        g = 78
        s = [
            "═" * g,
            f" {self.model_turu} Sonuçları — bağımlı değişken: {bagimli_gorunumu(self.bagimli)}",
            "═" * g,
            f" Gözlem sayısı: {self.n:<10} McFadden sözde R²: {sayi(self.pseudo_r2, 4):<10} Log-olabilirlik: {sayi(self.log_olabilirlik, 2)}",
            (
                f" LR χ²({self.k - 1}): {sayi(self.lr_istatistigi, 2):<10} p: {p_yaz(self.lr_p_degeri):<12}"
                if self.kovaryans_turu == "klasik"
                else f" Wald χ²({self.k - 1}): {sayi(self.wald_istatistigi, 2):<8} p: {p_yaz(self.wald_p_degeri):<12}"
            )
            + f" AIC: {sayi(self.aic, 1):<10} BIC: {sayi(self.bic, 1)}",
            f" Standart hatalar: {self.kovaryans_aciklama}",
            "─" * g,
            f" {'Değişken':<28}{'Katsayı':>12}{'Std. hata':>12}{'z':>9}{'p':>10}",
            "─" * g,
        ]
        for ad in self.katsayilar.index:
            gor = gorunen_ad(self, ad)
            gor = gor if len(gor) <= 27 else gor[:26] + "…"
            p = self.p_degerleri[ad]
            s.append(
                f" {gor:<28}{tablo_sayi(self.katsayilar[ad]):>12}{tablo_sayi(self.standart_hatalar[ad]):>12}"
                f"{sayi(self.z_degerleri[ad], 2):>9}{p_yaz(p):>10} {yildiz(p)}"
            )
        s += ["─" * g, " Katsayılar olasılık değişimi DEĞİLDİR. Yorum: model.yorumla() · model.marjinal_etkiler()", "═" * g]
        return "\n".join(s)

    def __str__(self) -> str:
        return self.ozet()

    __repr__ = __str__

    def _repr_html_(self) -> str:
        satir = "".join(
            f"<tr><td style='text-align:left'>{gorunen_ad(self, ad)}</td><td>{tablo_sayi(self.katsayilar[ad])}</td>"
            f"<td>{tablo_sayi(self.standart_hatalar[ad])}</td><td>{sayi(self.z_degerleri[ad], 2)}</td>"
            f"<td>{p_yaz(self.p_degerleri[ad])}</td><td>{yildiz(self.p_degerleri[ad])}</td></tr>"
            for ad in self.katsayilar.index
        )
        return (
            f"<div><b>{self.model_turu} Sonuçları</b> — bağımlı değişken: <code>{bagimli_gorunumu(self.bagimli)}</code><br>"
            f"Gözlem: {self.n} · McFadden sözde R²: {sayi(self.pseudo_r2, 4)} · LR χ²({self.k - 1}) = {sayi(self.lr_istatistigi, 2)} (p {p_yaz(self.lr_p_degeri)})<br>"
            f"<small>Standart hatalar: {self.kovaryans_aciklama}</small>"
            "<table><thead><tr><th style='text-align:left'>Değişken</th><th>Katsayı</th><th>Std. hata</th><th>z</th><th>p</th><th></th></tr></thead>"
            f"<tbody>{satir}</tbody></table><small>Katsayılar olasılık değişimi değildir — <code>model.yorumla()</code>, <code>model.marjinal_etkiler()</code></small></div>"
        )

    def yorumla(self):
        """Modelin ve ortalama marjinal etkilerin Türkçe yorumu."""
        from .yorum import Yorum

        Y = f"'{self.bagimli}' = 1"
        yr = Yorum(baslik=f"📊 Yorum — {self.model_turu}: {Y} olma olasılığı")
        sin = self.siniflandirma()
        yr.model_satirlari += [
            (
                f"Model, yalnızca sabit terimden oluşan modele göre anlamlı biçimde daha iyi (LR χ²({self.k - 1}) = {sayi(self.lr_istatistigi, 2)}, {p_cumle(self.lr_p_degeri)})."
                if self.lr_p_degeri < 0.05
                else f"Açıklayıcı değişkenler birlikte anlamlı değil (LR χ²({self.k - 1}) = {sayi(self.lr_istatistigi, 2)}, {p_cumle(self.lr_p_degeri)})."
            ),
            f"McFadden sözde R² = {sayi(self.pseudo_r2, 3)} (EKK'deki R² gibi 'açıklanan varyans' olarak yorumlanmaz; modeller arası karşılaştırma içindir).",
            f"Gözlemlerin {yuzde(100 * sin.attrs['dogruluk'])} kadarı doğru sınıflandırılıyor (eşik 0,5); örneklemde {Y} oranı {yuzde(100 * float(self.y.mean()))}.",
            f"Gözlem sayısı: {sayi(self.n, 0)} · standart hatalar: {self.kovaryans_aciklama}.",
        ]
        for _, s in self.marjinal_etkiler().iterrows():
            puan = f"{sayi(abs(100 * s['AME']), 2)} yüzde puan"
            anl = f"{anlamlilik_ifadesi(s['p'])} ({p_cumle(s['p'])})"
            if s["tür"] == "sürekli":
                b = self._birimler.get(s["değişken"])
                yr.katsayi_satirlari.append(
                    f"'{s['değişken']}' 1{' ' + b if b else ' birim'} artınca {Y} olma olasılığı ortalama {puan} {'artıyor' if s['AME'] > 0 else 'azalıyor'} — {anl}."
                )
            else:
                yr.katsayi_satirlari.append(
                    f"'{s['değişken']}' değişkeninde '{s['grup']}' grubunun {Y} olma olasılığı, referans gruba ('{s['referans']}') göre ortalama {puan} "
                    f"{'daha yüksek' if s['AME'] > 0 else 'daha düşük'} — {anl}."
                )
        yr.notlar += [
            "Yorumlar ortalama marjinal etkilere (AME) dayanır: katsayıların kendisi olasılıktaki değişimi vermez, yalnızca yönü gösterir.",
            "Marjinal etki her gözlem için farklıdır; burada tüm gözlemlerin ortalaması verildi (tablo: model.marjinal_etkiler()).",
        ]
        if self.tur == "logit":
            yr.notlar.append("Odds (bahis) oranları için: model.odds_oranlari()")
        yr.notlar.append("Bu yorumlar ilişkiyi anlatır; nedensellik için araştırma tasarımının buna izin vermesi gerekir.")
        return yr

    def rapor(
        self,
        dosya: str | None = None,
        baslik: str = "Bulgular",
        etiketler: dict[str, str] | None = None,
        tablo_no: int = 1,
        yontem: bool = True,
        kaynakca: bool = True,
        dil: str = "tr",
        bicim: str = "apa",
    ):
        """Tez / makale için bulgu raporu (Yöntem + Bulgular + tablolar). Biçim uzantıdan: .docx, .md, .html"""
        r = _rapor(self, baslik, etiketler, tablo_no, yontem, kaynakca, dil, bicim)
        if dosya is not None:
            r.kaydet(dosya)
        return r


# --------------------------------------------------------------------------- marjinal etkiler
def _marjinal_etkiler(m: IkiliSonuc) -> pd.DataFrame:
    di, b, V = m._design_info, m.katsayilar.values, m.kovaryans_matrisi.values
    veri = m._veri
    y_adlari = {c for c in veri.columns if re.search(rf"(?<![\w.]){re.escape(c)}(?!\w)", m.formul.split("~")[0])}
    kategoriler: dict[str, list] = {}
    for faktor, bilgi in di.factor_infos.items():
        if bilgi.type == "categorical":
            yalin = re.sub(r"^C\((.+?)(?:,.*)?\)$", r"\1", faktor.name()).strip()
            kategoriler[yalin] = list(bilgi.categories)

    def tasarim(d: pd.DataFrame) -> np.ndarray:
        (X,) = patsy.build_design_matrices([di], d)
        return np.asarray(X)

    def etki(agirlikli: list[tuple[float, pd.DataFrame]]) -> tuple[float, float]:
        ame, grad = 0.0, np.zeros(len(b))
        for w, d in agirlikli:
            X = tasarim(d)
            xb = X @ b
            ame += w * float(np.average(_F(xb, m.tur), weights=m._agirliklar))
            grad += w * np.average(_f(xb, m.tur)[:, None] * X, axis=0, weights=m._agirliklar)
        return ame, float(np.sqrt(grad @ V @ grad))

    def sabitle(sutun: str, deger) -> pd.DataFrame:
        d = veri.copy()
        if isinstance(veri[sutun].dtype, pd.CategoricalDtype):
            d[sutun] = pd.Categorical([deger] * len(d), categories=veri[sutun].cat.categories)
        else:
            d[sutun] = deger
        return d

    satirlar = []
    for c in veri.columns:
        if c in y_adlari:
            continue
        if c in kategoriler:
            ref, *digerleri = kategoriler[c]
            for seviye in digerleri:
                ame, sh = etki([(1.0, sabitle(c, seviye)), (-1.0, sabitle(c, ref))])
                satirlar.append((c, str(seviye), str(ref), "kategorik", ame, sh))
        elif not pd.api.types.is_numeric_dtype(veri[c]):
            continue
        else:  # Stata margins gibi: sayısal değişkende (0/1 olsa bile) türev; ayrık fark için C(x) / i.x kullan
            h = 1e-4 * max(float(veri[c].std()), 1e-8)
            art, eks = veri.copy(), veri.copy()
            art[c] = veri[c].astype(float) + h
            eks[c] = veri[c].astype(float) - h
            ame, sh = etki([(1 / (2 * h), art), (-1 / (2 * h), eks)])
            satirlar.append((c, "", "", "sürekli", ame, sh))
    t = pd.DataFrame(satirlar, columns=["değişken", "grup", "referans", "tür", "AME", "std. hata"])
    t["z"] = t["AME"] / t["std. hata"]
    t["p"] = 2 * stats.norm.sf(np.abs(t["z"]))
    t.index = [f"{d}: {g}" if tur == "kategorik" else d for d, g, tur in zip(t["değişken"], t["grup"], t["tür"])]
    return t


# --------------------------------------------------------------------------- tahmin
def _tahmin_et(
    formul: str,
    veri: pd.DataFrame,
    tur: str,
    kovaryans: str,
    birimler: dict | None,
    derinlik: int = 2,
    kume: str | None = None,
    agirlik: str | None = None,
    agirlik_turu: str = "pw",
) -> IkiliSonuc:
    if kume is not None and kovaryans == "klasik":
        kovaryans = "kume"
    if kovaryans not in ("klasik", "saglam", "kume"):
        raise EkonometriHatasi(
            f"Bilinmeyen kovaryans türü: {kovaryans!r}. Seçenekler: 'klasik', 'saglam' (Stata vce(robust)), 'kume' (vce(cluster))"
        )
    if kovaryans == "kume" and (kume is None or kume not in veri.columns):
        raise EkonometriHatasi("Kümelenmiş standart hata için veride bulunan bir küme sütunu ver: kume='okul'.")
    if not isinstance(veri, pd.DataFrame):
        raise EkonometriHatasi("veri bir pandas DataFrame olmalı (ör. pd.read_csv(...) ya da ek.veri.yukle(...)).")
    veri = benzersiz_indeks(veri)
    try:
        env = patsy.EvalEnvironment.capture(derinlik).with_outer_namespace(
            {"np": np, "log": np.log, "ln": np.log, "exp": np.exp, "sqrt": np.sqrt}
        )
        y_df, X_df = patsy.dmatrices(formul, veri, return_type="dataframe", NA_action="drop", eval_env=env)
    except patsy.PatsyError as h:
        raise EkonometriHatasi(
            f"Formül okunamadı: {h}\nSütun adlarını ve yazımı kontrol et. Mevcut sütunlar: {list(veri.columns)}"
        ) from None
    ornek = "veri['y01'] = (veri['durum'] == 'Evet').astype(int)"
    if y_df.shape[1] != 1:
        raise EkonometriHatasi(f"Bağımlı değişken 0/1 kodlu tek bir sayısal sütun olmalı. Metin sütununu çevirmek için: {ornek}")
    y = y_df.iloc[:, 0].values.astype(float)
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise EkonometriHatasi(
            f"Bağımlı değişken yalnızca 0 ve 1 değerlerini almalı (bulunan değerler: {np.unique(y)[:6].tolist()}…). Sürekli bağımlı değişken için ek.ekk() kullan."
        )
    if len(np.unique(y)) < 2:
        raise EkonometriHatasi("Bağımlı değişkenin tüm değerleri aynı; logit/probit için hem 0 hem 1 gözlenmeli.")
    di = X_df.design_info  # satır seçimi design_info'yu düşürür; önceden sakla
    ag = None
    if agirlik is not None:
        if agirlik_turu not in ("fw", "pw"):
            raise EkonometriHatasi(
                "Logit/probit için yalnızca frekans (fw) ve örneklem (pw) ağırlıkları kullanılabilir (Stata da aw'ye izin vermez)."
            )
        if agirlik not in veri.columns:
            raise EkonometriHatasi(f"Ağırlık değişkeni {agirlik!r} veride yok.")
        w_tum = pd.to_numeric(veri.loc[y_df.index, agirlik], errors="coerce").astype(float)
        if (w_tum < 0).any():
            raise EkonometriHatasi("Ağırlıklar negatif olamaz.")
        gecerli = (w_tum.notna() & (w_tum > 0)).values
        y_df, X_df, y = y_df.loc[gecerli], X_df.loc[gecerli], y[gecerli]
        ag = w_tum.values[gecerli]
        if agirlik_turu == "fw" and not np.allclose(ag, np.round(ag)):
            raise EkonometriHatasi("Frekans ağırlıkları (fw) tam sayı olmalı.")
        if agirlik_turu == "fw" and kovaryans == "kume":
            raise EkonometriHatasi("Frekans ağırlıklarıyla kümelenmiş standart hata desteklenmez.")
        if agirlik_turu == "pw" and kovaryans == "klasik":
            kovaryans = "saglam"  # Stata: pweight → vce(robust)
    X = X_df.values.astype(float)
    n, k = X.shape
    if np.linalg.matrix_rank(X) < k:
        raise EkonometriHatasi(
            "Tam çoklu doğrusal bağlantı: bazı açıklayıcı değişkenler diğerlerinin doğrusal birleşimi (ör. kukla değişken tuzağı)."
        )

    b = np.zeros(k)
    ll = _ll(y, X @ b, tur, ag)
    yakinsadi = False
    it = 0
    for _ in range(100):
        it += 1
        S, H = _skor_hessian(X, y, b, tur, ag)
        g = S.sum(axis=0)
        try:
            adim = np.linalg.solve(-H, g)
        except np.linalg.LinAlgError:
            break
        yeni, olcek = b + adim, 1.0
        ll_yeni = _ll(y, X @ yeni, tur, ag)
        while not np.isfinite(ll_yeni) or ll_yeni < ll - 1e-12:
            olcek /= 2
            if olcek < 1e-8:
                break
            yeni = b + olcek * adim
            ll_yeni = _ll(y, X @ yeni, tur, ag)
        degisim = ll_yeni - ll
        b, ll = yeni, ll_yeni
        if np.max(np.abs(adim)) < 1e-9 or abs(degisim) < 1e-13:
            yakinsadi = True
            break
    p = _F(X @ b, tur)
    if np.all((p < 1e-6) | (p > 1 - 1e-6)) or not yakinsadi or np.max(np.abs(X @ b)) > 30:
        raise EkonometriHatasi(
            "Model yakınsamadı: büyük olasılıkla (yarı) tam ayrışma var — bir değişken ya da değişken bileşimi sonucu kusursuz biçimde belirliyor.\n"
            "Hangi değişkenin 0 ve 1 gruplarını tamamen ayırdığını kontrol et, o değişkeni çıkar ya da kategorileri birleştir."
        )
    S, H = _skor_hessian(X, y, b, tur, ag)
    Hinv = np.linalg.inv(-H)
    N = int(round(ag.sum())) if (ag is not None and agirlik_turu == "fw") else n
    if kovaryans == "klasik":
        V, aciklama = Hinv, "klasik (en çok olabilirlik, bilgi matrisi)"
    elif kovaryans == "saglam":  # Stata vce(robust): çarpan N/(N−1)
        Sm = S / np.sqrt(ag)[:, None] if N != n else S  # fw: Σ w·s²·xx' (gözlemleri çoğaltmakla aynı)
        V, aciklama = N / (N - 1) * Hinv @ (Sm.T @ Sm) @ Hinv, "dayanıklı (sandviç / Huber-White)"
    else:  # Stata vce(cluster): çarpan G/(G−1)
        kod = pd.factorize(veri.loc[y_df.index, kume])[0]
        G = int(kod.max()) + 1
        toplam = np.column_stack([np.bincount(kod, weights=S[:, j], minlength=G) for j in range(k)])
        V = G / (G - 1) * Hinv @ (toplam.T @ toplam) @ Hinv
        aciklama = f"kümelenmiş, {G} küme ({kume})"
    se = np.sqrt(np.diag(V))
    idx = [i for i, c in enumerate(X_df.columns) if c != "Intercept"]
    if idx:
        wald = float(b[idx] @ np.linalg.inv(V[np.ix_(idx, idx)]) @ b[idx])
        wald_p = float(stats.chi2.sf(wald, len(idx)))
    else:
        wald, wald_p = float("nan"), float("nan")
    z = b / se
    pd_ = 2 * stats.norm.sf(np.abs(z))
    agt = np.ones(n) if ag is None else ag
    ybar = float(np.sum(agt * y) / np.sum(agt))
    ll0 = float(np.sum(agt) * (ybar * np.log(ybar) + (1 - ybar) * np.log(1 - ybar)))
    if ag is not None:
        aciklama += f"; ağırlık: [{agirlik_turu}={agirlik}]"
    lr = 2 * (ll - ll0)
    isimler = list(X_df.columns)
    seri = lambda v, ad: pd.Series(v, index=isimler, name=ad)  # noqa: E731
    kullanilan = [c for c in veri.columns if isinstance(c, str) and re.search(rf"(?<![\w.]){re.escape(c)}(?!\w)", formul)]
    return IkiliSonuc(
        tur=tur,
        formul=formul,
        bagimli=y_df.columns[0],
        katsayilar=seri(b, "katsayı"),
        standart_hatalar=seri(se, "std. hata"),
        z_degerleri=seri(z, "z"),
        p_degerleri=seri(pd_, "p"),
        kovaryans_matrisi=pd.DataFrame(V, index=isimler, columns=isimler),
        kovaryans_turu=kovaryans,
        kovaryans_aciklama=aciklama,
        n=N,
        k=k,
        log_olabilirlik=ll,
        log_olabilirlik_sifir=ll0,
        pseudo_r2=float(1 - ll / ll0),
        lr_istatistigi=float(lr),
        lr_p_degeri=float(stats.chi2.sf(lr, k - 1)) if k > 1 else float("nan"),
        aic=float(-2 * ll + 2 * k),
        bic=float(-2 * ll + np.log(N) * k),
        olasiliklar=pd.Series(p, index=y_df.index, name=f"P({y_df.columns[0]} = 1)"),
        yineleme=it,
        wald_istatistigi=wald,
        wald_p_degeri=wald_p,
        X=X_df,
        y=y_df.iloc[:, 0],
        _design_info=di,
        _terimler=_terimleri_coz(di),
        _birimler=dict(birimler or {}),
        _veri=veri.loc[y_df.index, kullanilan],
        _agirliklar=ag,
        _etiketler=dict(veri.attrs.get("etiketler", {})),
    )


def logit(
    formul: str,
    veri: pd.DataFrame,
    kovaryans: str = "klasik",
    birimler: dict[str, str] | None = None,
    kume: str | None = None,
    agirlik: str | None = None,
    agirlik_turu: str = "pw",
) -> IkiliSonuc:
    """Logit (lojistik regresyon): P(y = 1) = 1 / (1 + e^(−xβ)). Bağımlı değişken 0/1 olmalıdır.

    kovaryans : "klasik" (Stata varsayılanı, OIM), "saglam" (Stata ``vce(robust)``) ya da "kume" (``vce(cluster kume)``)
    """
    return _tahmin_et(formul, veri, "logit", kovaryans, birimler, kume=kume, agirlik=agirlik, agirlik_turu=agirlik_turu)


def probit(
    formul: str,
    veri: pd.DataFrame,
    kovaryans: str = "klasik",
    birimler: dict[str, str] | None = None,
    kume: str | None = None,
    agirlik: str | None = None,
    agirlik_turu: str = "pw",
) -> IkiliSonuc:
    """Probit: P(y = 1) = Φ(xβ) (standart normal dağılım fonksiyonu). Bağımlı değişken 0/1 olmalıdır."""
    return _tahmin_et(formul, veri, "probit", kovaryans, birimler, kume=kume, agirlik=agirlik, agirlik_turu=agirlik_turu)


# --------------------------------------------------------------------------- rapor
def _rapor_en(m: IkiliSonuc, baslik, etiketler, tablo_no, yontem, kaynakca, bicim):
    from . import __version__
    from .karsilastirma import _kov_en
    from .rapor import KAYNAKLAR, Rapor, Tablo, _Adlar, _duzey, _p, _p_hucre, _tanimlayici

    ad = _Adlar({**m._etiketler, **(etiketler or {})})
    Y = f"{ad.metin(m.bagimli)} = 1"
    r = Rapor(baslik, bicim=bicim, dil="en")
    sembol = "Λ" if m.tur == "logit" else "Φ"
    model_ad = "logit (logistic regression)" if m.tur == "logit" else "probit"
    if yontem:
        sag, i = [], 0
        for t in m._terimler:
            if t.tur == "sabit":
                sag.append("β₀")
            else:
                i += 1
                sag.append(f"β{str(i).translate(_ALT)}·({ad.satir(t)})")
        dagilim = "the logistic cdf" if m.tur == "logit" else "the standard normal cdf"
        r.bloklar += [
            ("baslik", "Method"),
            (
                "paragraf",
                f"Because {ad.metin(m.bagimli)} is binary, the probability that {Y} was modelled with {model_ad} and estimated by maximum likelihood (Wooldridge, 2020). The estimated model is:",
            ),
            ("denklem", f"P({ad.yalin(m.bagimli)} = 1) = {sembol}(" + " + ".join(sag) + ")"),
            (
                "paragraf",
                f"Here {sembol} denotes {dagilim}. The coefficients do not equal the change in probability, so the findings are interpreted through average marginal effects (AME). "
                "Marginal effects are computed at every observation and then averaged; their standard errors use the delta method. For a categorical variable the marginal effect is the average difference in probability between that group and the reference group. "
                f"Standard errors: {_kov_en(m.kovaryans_aciklama)}. Overall significance is assessed with a likelihood-ratio (LR) test, fit with McFadden's (1974) pseudo R², and classification accuracy. "
                f"The calculations were done with ekonometri-tr {__version__}.",
            ),
        ]
    b, no = _tanimlayici(m, ad, tablo_no)
    r.bloklar += b
    ga = m.guven_araliklari()
    sutunlar = ["Variable", "β", "SE", "z", "p"] + (["Odds ratio"] if m.tur == "logit" else []) + ["95% CI"]
    satirlar = []
    for t in m._terimler:
        s = t.sutun
        satir = [
            ad.satir(t),
            tablo_sayi(m.katsayilar[s]) + yildiz(m.p_degerleri[s]),
            tablo_sayi(m.standart_hatalar[s]),
            sayi(m.z_degerleri[s], 2),
            _p_hucre(m.p_degerleri[s]),
        ]
        if m.tur == "logit":
            satir.append("" if t.tur == "sabit" else sayi(float(np.exp(m.katsayilar[s])), 3))
        satirlar.append(satir + [f"[{tablo_sayi(ga.loc[s, 'alt'])}; {tablo_sayi(ga.loc[s, 'üst'])}]"])
    lr = f"LR χ²({m.k - 1}) = {sayi(m.lr_istatistigi, 2)}"
    sin = m.siniflandirma()
    kat_no, ame_no, sin_no = no, no + 1, no + 2
    anlam = (
        f"The likelihood-ratio test shows that the model is jointly significant ({lr}; {_p(m.lr_p_degeri)}). "
        if m.lr_p_degeri < 0.05
        else f"The likelihood-ratio test does not find the model jointly significant ({lr}; {_p(m.lr_p_degeri)}). "
    )
    r.bloklar += [
        ("baslik", f"{m.model_turu} results"),
        (
            "paragraf",
            f"{m.model_turu} estimates are reported in Table {kat_no}. {anlam}McFadden pseudo R² is {sayi(m.pseudo_r2, 3)}.",
        ),
        (
            "tablo",
            Tablo(
                kat_no,
                f"{m.model_turu} estimates",
                sutunlar,
                satirlar,
                f"N = {sayi(m.n, 0)}; {lr} ({_p(m.lr_p_degeri)}); McFadden pseudo R² = {sayi(m.pseudo_r2, 3)}; log-likelihood = {sayi(m.log_olabilirlik, 2)}. "
                f"Dependent variable: {ad.yalin(m.bagimli)} (0/1). Standard errors: {_kov_en(m.kovaryans_aciklama)}. SE: standard error; CI: confidence interval. *** p < 0.01; ** p < 0.05; * p < 0.10.",
            ),
        ),
    ]
    ame = m.marjinal_etkiler()
    zk = stats.norm.ppf(0.975)
    ame_satir, cumle = [], []
    for _, s in ame.iterrows():
        gor = f"{ad.yalin(s['değişken'])}: {s['grup']}" if s["tür"] == "kategorik" else ad.yalin(s["değişken"])
        ame_satir.append(
            [
                gor,
                tablo_sayi(s["AME"]) + yildiz(s["p"]),
                tablo_sayi(s["std. hata"]),
                sayi(s["z"], 2),
                _p_hucre(s["p"]),
                f"[{tablo_sayi(s['AME'] - zk * s['std. hata'])}; {tablo_sayi(s['AME'] + zk * s['std. hata'])}]",
            ]
        )
        ist = f"(AME = {tablo_sayi(s['AME'])}; SE = {tablo_sayi(s['std. hata'])}; z = {sayi(s['z'], 2)}; {_p(s['p'])})"
        puan = f"{sayi(abs(100 * s['AME']), 2)} percentage points"
        X = ad.metin(s["değişken"])
        birim = m._birimler.get(s["değişken"], "unit")
        if s["p"] >= 0.10:
            grup = f" for the group '{s['grup']}'" if s["tür"] == "kategorik" else ""
            cumle.append(f"The marginal effect of {X}{grup} on the probability that {Y} is not statistically significant {ist}.")
        elif s["tür"] == "sürekli":
            yon = "increases" if s["AME"] > 0 else "decreases"
            cumle.append(
                f"Holding the other variables fixed, a one-{birim} increase in {X} {yon} the probability that {Y} by {puan} on average; this effect is {_duzey(s['p'])} {ist}."
            )
        else:
            yon = "higher" if s["AME"] > 0 else "lower"
            cumle.append(
                f"For {X}, the probability that {Y} is {puan} {yon} on average in the '{s['grup']}' group than in the reference group ('{s['referans']}'); this difference is {_duzey(s['p'])} {ist}."
            )
    gercek = {"Gerçek: 0": "Actual: 0", "Gerçek: 1": "Actual: 1"}
    r.bloklar += [
        (
            "paragraf",
            f"Average marginal effects of the regressors on the probability that {Y} are reported in Table {ame_no}. " + " ".join(cumle),
        ),
        (
            "tablo",
            Tablo(
                ame_no,
                "Average marginal effects (AME)",
                ["Variable", "AME", "SE", "z", "p", "95% CI"],
                ame_satir,
                f"AME is the average change in the probability that {ad.yalin(m.bagimli)} = 1 for a one-unit increase (or, for a categorical variable, the difference from the reference group); 0.01 = 1 percentage point. Standard errors use the delta method. *** p < 0.01; ** p < 0.05; * p < 0.10.",
            ),
        ),
        ("baslik", "Classification"),
        (
            "paragraf",
            f"Classification performance is reported in Table {sin_no}. Observations with a predicted probability of at least 0.50 are classified as 1, and the model classifies "
            f"{yuzde(100 * sin.attrs['dogruluk'])} of the observations correctly (sensitivity = {yuzde(100 * sin.attrs['duyarlilik'])}; specificity = {yuzde(100 * sin.attrs['ozgulluk'])}). "
            f"The sample share with {Y} is {yuzde(100 * float(m.y.mean()))}; accuracy should be read together with that base rate.",
        ),
        (
            "tablo",
            Tablo(
                sin_no,
                "Classification table",
                ["", "Predicted: 0", "Predicted: 1", "Total"],
                [
                    [gercek.get(i, i), sayi(int(row.iloc[0]), 0), sayi(int(row.iloc[1]), 0), sayi(int(row.sum()), 0)]
                    for i, row in sin.iterrows()
                ],
                f"Threshold 0.50. Correct classification rate: {yuzde(100 * sin.attrs['dogruluk'])}.",
            ),
        ),
        (
            "paragraf",
            "The findings describe associations. A causal reading is warranted only to the extent that the research design supports it.",
        ),
    ]
    if yontem and kaynakca:
        r.bloklar += [("baslik", "References"), ("liste", sorted([KAYNAKLAR["wooldridge"], KAYNAKLAR["mcfadden"]]))]
    return r


def _rapor(m: IkiliSonuc, baslik, etiketler, tablo_no, yontem, kaynakca, dil="tr", bicim="apa"):
    from .rapor import rapor_ayari

    if dil == "en" and baslik == "Bulgular":
        baslik = "Findings"
    token = rapor_ayari(dil, bicim)
    try:
        if dil == "en":
            return _rapor_en(m, baslik, etiketler, tablo_no, yontem, kaynakca, bicim)
        return _rapor_tr(m, baslik, etiketler, tablo_no, yontem, kaynakca, bicim)
    finally:
        DIL.reset(token)


def _rapor_tr(m: IkiliSonuc, baslik, etiketler, tablo_no, yontem, kaynakca, bicim="apa"):
    from . import __version__
    from .rapor import KAYNAKLAR, Rapor, Tablo, _Adlar, _duzey, _p, _p_hucre, _tanimlayici

    ad = _Adlar({**m._etiketler, **(etiketler or {})})
    Y = f"{ad.metin(m.bagimli)} = 1"
    r = Rapor(baslik, bicim=bicim, dil="tr")
    sembol = "Λ" if m.tur == "logit" else "Φ"
    if yontem:
        sag, i = [], 0
        for t in m._terimler:
            if t.tur == "sabit":
                sag.append("β₀")
            else:
                i += 1
                sag.append(f"β{str(i).translate(_ALT)}·({ad.satir(t)})")
        r.bloklar += [
            ("baslik", "Yöntem"),
            (
                "paragraf",
                f"Bağımlı değişken iki değerli (0/1) olduğundan {Y} olma olasılığı "
                + ("logit (lojistik regresyon)" if m.tur == "logit" else "probit")
                + " modeliyle incelenmiş, model en çok olabilirlik yöntemiyle tahmin edilmiştir (Wooldridge, 2020). Tahmin edilen model aşağıdaki gibidir:",
            ),
            ("denklem", f"P({ad.yalin(m.bagimli)} = 1) = {sembol}(" + " + ".join(sag) + ")"),
            (
                "paragraf",
                f"Burada {sembol}, "
                + ("lojistik dağılım fonksiyonunu" if m.tur == "logit" else "standart normal birikimli dağılım fonksiyonunu")
                + " göstermektedir. Bu tür modellerde katsayılar olasılıktaki değişimi doğrudan vermediğinden bulgular ortalama marjinal etkiler (AME) üzerinden yorumlanmıştır. "
                "Marjinal etkiler her gözlem için hesaplanıp ortalaması alınmış, standart hataları delta yöntemiyle elde edilmiştir; kategorik değişkenlerde marjinal etki, ilgili grup ile referans grup arasındaki ortalama olasılık farkıdır. "
                f"Standart hatalar: {m.kovaryans_aciklama}. Modelin genel anlamlılığı olabilirlik oranı (LR) testiyle, uyum iyiliği McFadden (1974) sözde R² ölçüsü ve doğru sınıflandırma oranıyla değerlendirilmiştir. "
                f"Hesaplar ekonometri-tr {__version__} ile yapılmıştır.",
            ),
        ]
    b, no = _tanimlayici(m, ad, tablo_no)
    r.bloklar += b

    # ---- katsayı tablosu
    ga = m.guven_araliklari()
    sutunlar = ["Değişken", "β", "SH", "z", "p"] + (["Odds oranı"] if m.tur == "logit" else []) + ["%95 GA"]
    satirlar = []
    for t in m._terimler:
        s = t.sutun
        satir = [
            ad.satir(t),
            tablo_sayi(m.katsayilar[s]) + yildiz(m.p_degerleri[s]),
            tablo_sayi(m.standart_hatalar[s]),
            sayi(m.z_degerleri[s], 2),
            _p_hucre(m.p_degerleri[s]),
        ]
        if m.tur == "logit":
            satir.append("" if t.tur == "sabit" else sayi(float(np.exp(m.katsayilar[s])), 3))
        satirlar.append(satir + [f"[{tablo_sayi(ga.loc[s, 'alt'])}; {tablo_sayi(ga.loc[s, 'üst'])}]"])
    lr = f"LR χ²({m.k - 1}) = {sayi(m.lr_istatistigi, 2)}"
    sin = m.siniflandirma()
    kat_no, ame_no, sin_no = no, no + 1, no + 2
    r.bloklar += [
        ("baslik", f"{m.model_turu} Modeli Bulguları"),
        (
            "paragraf",
            f"{m.model_turu} modelinin tahmin sonuçları Tablo {kat_no} ile sunulmuştur. "
            + (
                f"Olabilirlik oranı testi modelin bir bütün olarak istatistiksel olarak anlamlı olduğunu göstermektedir ({lr}; {_p(m.lr_p_degeri)}). "
                if m.lr_p_degeri < 0.05
                else f"Olabilirlik oranı testine göre model bir bütün olarak istatistiksel olarak anlamlı bulunmamıştır ({lr}; {_p(m.lr_p_degeri)}). "
            )
            + f"McFadden sözde R² değeri {sayi(m.pseudo_r2, 3)} olarak hesaplanmıştır.",
        ),
        (
            "tablo",
            Tablo(
                kat_no,
                f"{m.model_turu} modeli tahmin sonuçları",
                sutunlar,
                satirlar,
                f"N = {sayi(m.n, 0)}; {lr} ({_p(m.lr_p_degeri)}); McFadden sözde R² = {sayi(m.pseudo_r2, 3)}; log-olabilirlik = {sayi(m.log_olabilirlik, 2)}. "
                f"Bağımlı değişken: {ad.yalin(m.bagimli)} (0/1). Standart hatalar: {m.kovaryans_aciklama}. SH: standart hata; GA: güven aralığı. *** p < 0,01; ** p < 0,05; * p < 0,10.",
            ),
        ),
    ]

    # ---- marjinal etkiler
    ame = m.marjinal_etkiler()
    zk = stats.norm.ppf(0.975)
    ame_satir, cumle = [], []
    for _, s in ame.iterrows():
        gor = f"{ad.yalin(s['değişken'])}: {s['grup']}" if s["tür"] == "kategorik" else ad.yalin(s["değişken"])
        ame_satir.append(
            [
                gor,
                tablo_sayi(s["AME"]) + yildiz(s["p"]),
                tablo_sayi(s["std. hata"]),
                sayi(s["z"], 2),
                _p_hucre(s["p"]),
                f"[{tablo_sayi(s['AME'] - zk * s['std. hata'])}; {tablo_sayi(s['AME'] + zk * s['std. hata'])}]",
            ]
        )
        ist = f"(AME = {tablo_sayi(s['AME'])}; SH = {tablo_sayi(s['std. hata'])}; z = {sayi(s['z'], 2)}; {_p(s['p'])})"
        puan = f"{sayi(abs(100 * s['AME']), 2)} yüzde puan"
        X = ad.metin(s["değişken"])
        if s["p"] >= 0.10:
            cumle.append(
                f"{X} değişkeninin{(' ' + chr(39) + s['grup'] + chr(39) + ' grubu için') if s['tür'] == 'kategorik' else ''} {Y} olma olasılığı üzerindeki marjinal etkisi istatistiksel olarak anlamlı bulunmamıştır {ist}."
            )
        elif s["tür"] == "sürekli":
            cumle.append(
                f"Diğer değişkenler sabitken, {X} değişkeni 1 {m._birimler.get(s['değişken'], 'birim')} arttığında {Y} olma olasılığı ortalama {puan} "
                f"{'artmaktadır' if s['AME'] > 0 else 'azalmaktadır'}; bu etki {_duzey(s['p'])} {ist}."
            )
        else:
            cumle.append(
                f"{X} değişkeninde '{s['grup']}' grubunun {Y} olma olasılığı, referans gruba ('{s['referans']}') kıyasla ortalama {puan} "
                f"daha {'yüksektir' if s['AME'] > 0 else 'düşüktür'}; bu fark {_duzey(s['p'])} {ist}."
            )
    r.bloklar += [
        (
            "paragraf",
            f"Açıklayıcı değişkenlerin {Y} olma olasılığı üzerindeki ortalama marjinal etkileri Tablo {ame_no} ile sunulmuştur. "
            + " ".join(cumle),
        ),
        (
            "tablo",
            Tablo(
                ame_no,
                "Ortalama marjinal etkiler (AME)",
                ["Değişken", "AME", "SH", "z", "p", "%95 GA"],
                ame_satir,
                f"AME, ilgili değişkendeki bir birimlik artışın (kategorik değişkenlerde referans gruba göre farkın) {ad.yalin(m.bagimli)} = 1 olma olasılığında yarattığı ortalama değişimdir; 0,01 = 1 yüzde puan. Standart hatalar delta yöntemiyle hesaplanmıştır. *** p < 0,01; ** p < 0,05; * p < 0,10.",
            ),
        ),
        ("baslik", "Sınıflandırma Başarısı"),
        (
            "paragraf",
            f"Modelin sınıflandırma başarısı Tablo {sin_no} ile sunulmuştur. Tahmin edilen olasılığın 0,50 ve üzerinde olduğu gözlemler 1 olarak sınıflandırıldığında "
            f"model gözlemlerin {yuzde(100 * sin.attrs['dogruluk'])} kadarını doğru sınıflandırmaktadır (duyarlılık = {yuzde(100 * sin.attrs['duyarlilik'])}; özgüllük = {yuzde(100 * sin.attrs['ozgulluk'])}). "
            f"Örneklemde {Y} olan gözlemlerin oranı {yuzde(100 * float(m.y.mean()))} düzeyindedir; doğru sınıflandırma oranı bu taban oranla birlikte değerlendirilmelidir.",
        ),
        (
            "tablo",
            Tablo(
                sin_no,
                "Sınıflandırma tablosu",
                ["", "Tahmin: 0", "Tahmin: 1", "Toplam"],
                [[i, sayi(int(row.iloc[0]), 0), sayi(int(row.iloc[1]), 0), sayi(int(row.sum()), 0)] for i, row in sin.iterrows()],
                f"Eşik değer 0,50. Doğru sınıflandırma oranı: {yuzde(100 * sin.attrs['dogruluk'])}.",
            ),
        ),
        (
            "paragraf",
            "Bulgular değişkenler arasındaki ilişkiyi göstermektedir; nedensel bir yorum, araştırma tasarımının buna izin verdiği ölçüde yapılabilir.",
        ),
    ]
    if yontem and kaynakca:
        r.bloklar += [("baslik", "Kaynakça"), ("liste", sorted([KAYNAKLAR["wooldridge"], KAYNAKLAR["mcfadden"]]))]
    return r


__all__ = ["logit", "probit", "IkiliSonuc"]
