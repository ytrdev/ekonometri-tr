"""Türkçe grafikler (Stata: rvfplot, coefplot, marginsplot, tsline, histogram, scatter, lroc).

Her fonksiyon bir ``Sekil`` döndürür:
>>> s = ek.kalinti_grafigi(model)
>>> s.kaydet("kalinti.png")          # 300 dpi PNG (ya da .pdf / .svg)
>>> r = model.rapor(); r.sekil_ekle(s, "Kalıntılar ve tahmin edilen değerler"); r.kaydet("bulgular.docx")
Jupyter'de doğrudan görüntülenir. Eksen sayıları Türkçe ondalık virgülle yazılır.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .regresyon import EkonometriHatasi


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        raise EkonometriHatasi("Grafikler için matplotlib gerekli:  pip install matplotlib") from None
    return plt


_UST = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _virgul(x, _pos=None) -> str:
    """Türkçe sayı: ondalık virgül, binlik nokta; çok küçük / çok büyük değerler 1,4×10⁻⁵ biçiminde ('e' gösterimi yok)."""
    if abs(x) < 1e-12:  # 0 ve eksen hesaplarındaki kayan nokta artıkları (2,7×10⁻¹⁷ gibi)
        return "0"
    if 1e-3 <= abs(x) < 1000:
        return f"{x:g}".replace(".", ",")
    if 1000 <= abs(x) < 1e15:
        return f"{x:,.0f}".replace(",", ".")
    m, us = f"{x:.1e}".split("e")
    m = m[:-2] if m.endswith(".0") else m
    return f"{m.replace('.', ',')}×10{str(int(us)).translate(_UST)}"


def _stil(ax, baslik: str | None, x: str | None, y: str | None) -> None:
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    if baslik:
        ax.set_title(baslik, fontsize=11)
    if x:
        ax.set_xlabel(x)
    if y:
        ax.set_ylabel(y)
    for eksen in (ax.xaxis, ax.yaxis):
        if not getattr(eksen, "_kategorik", False):
            eksen.set_major_locator(MaxNLocator(nbins=7))  # büyük sayılarda etiketler üst üste binmesin
            eksen.set_major_formatter(FuncFormatter(_virgul))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


def _yeni(genislik: float = 6.4, yukseklik: float = 4.2):
    plt = _plt()
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.titleweight": "bold", "figure.dpi": 100})
    fig, ax = plt.subplots(figsize=(genislik, yukseklik))
    return fig, ax


@dataclass
class Sekil:
    """Grafik. ``kaydet()`` ile dosyaya yazılır; ``png()`` bayt döndürür (Word raporuna eklemek için)."""

    fig: object
    baslik: str = ""

    def png(self, dpi: int = 300) -> bytes:
        tampon = io.BytesIO()
        self.fig.savefig(tampon, format="png", dpi=dpi, bbox_inches="tight")
        return tampon.getvalue()

    def kaydet(self, dosya: str | Path, dpi: int = 300) -> Path:
        yol = Path(dosya)
        self.fig.savefig(yol, dpi=dpi, bbox_inches="tight")
        return yol

    def _repr_png_(self) -> bytes:
        return self.png(dpi=110)

    def __repr__(self) -> str:
        return f"<Şekil: {self.baslik or 'grafik'} — .kaydet('sekil.png') ile kaydet>"

    def kapat(self) -> None:
        _plt().close(self.fig)


def _bitir(fig, baslik) -> Sekil:
    fig.tight_layout()
    _plt().close(fig)  # Jupyter'de iki kez çizilmesin; Sekil kendi gösterir
    return Sekil(fig, baslik or "")


def _gorunen(model, sutun: str) -> str:
    from .regresyon import gorunen_ad

    return gorunen_ad(model, sutun)


# --------------------------------------------------------------------------- regresyon grafikleri
def kalinti_grafigi(model, baslik: str | None = "Kalıntılar ve tahmin edilen değerler") -> Sekil:
    """Kalıntı–tahmin grafiği (Stata ``rvfplot``): değişen varyans ve doğrusal olmayan ilişkiyi gözle kontrol etmek için."""
    e = np.asarray(model.kalintilar, dtype=float)
    if hasattr(model, "tahminler"):
        yhat = np.asarray(model.tahminler, dtype=float)
    else:
        yhat = np.asarray(model.y, dtype=float) - e
    fig, ax = _yeni()
    ax.scatter(yhat, e, s=10, alpha=0.5, color="#1f4e79", edgecolors="none")
    ax.axhline(0, color="#b22222", linewidth=1)
    if len(e) > 20:  # yerel ortalama eğrisi (eğilim)
        sira = np.argsort(yhat)
        k = max(5, len(e) // 20)
        hareketli = pd.Series(e[sira]).rolling(k, center=True, min_periods=1).mean()
        ax.plot(yhat[sira], hareketli, color="#e08a00", linewidth=1.5, label="yerel ortalama")
        ax.legend(frameon=False, fontsize=8)
    from .regresyon import bagimli_gorunumu

    _stil(ax, baslik, f"Tahmin edilen değerler ({bagimli_gorunumu(model.bagimli)})", "Kalıntılar")
    return _bitir(fig, baslik)


def normallik_grafigi(model, baslik: str | None = "Kalıntıların normal olasılık grafiği") -> Sekil:
    """Kalıntılar için normal Q-Q grafiği (Stata ``qnorm``)."""
    from scipy import stats

    e = np.sort(np.asarray(model.kalintilar, dtype=float))
    n = len(e)
    teorik = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))  # Stata qnorm: P_i = i / (N + 1)
    fig, ax = _yeni(5.2, 4.6)
    ax.scatter(teorik, e, s=10, alpha=0.6, color="#1f4e79", edgecolors="none")
    ax.plot(teorik, e.mean() + e.std(ddof=1) * teorik, color="#b22222", linewidth=1)
    _stil(ax, baslik, "Teorik normal değerler", "Kalıntılar")
    return _bitir(fig, baslik)


def katsayi_grafigi(
    *modeller,
    adlar: list[str] | None = None,
    guven: float = 0.95,
    sabit: bool = False,
    etiketler: dict | None = None,
    baslik: str | None = None,
) -> Sekil:
    """Katsayılar ve güven aralıkları (Stata ``coefplot``); birden çok model yan yana gösterilebilir."""
    from .rapor import _Adlar

    if not modeller:
        raise EkonometriHatasi("En az bir model ver: ek.katsayi_grafigi(m1, m2)")
    adlar = adlar or ([f"Model {i}" for i in range(1, len(modeller) + 1)] if len(modeller) > 1 else [""])
    ad = _Adlar(etiketler)
    sira: dict[str, str] = {}
    for m in modeller:
        for t in m._terimler:
            if t.tur == "sabit" and not sabit:
                continue
            sira.setdefault(t.sutun, "Sabit" if t.tur == "sabit" else ad.satir(t))
    terimler = list(sira)
    fig, ax = _yeni(6.4, max(2.5, 0.42 * len(terimler) + 1.2))
    renkler = ["#1f4e79", "#b22222", "#2e7d32", "#e08a00", "#6a1b9a"]
    kayma = np.linspace(-0.15, 0.15, len(modeller)) if len(modeller) > 1 else [0.0]
    for j, m in enumerate(modeller):
        ga = m.guven_araliklari(guven)
        konum, b, alt, ust = [], [], [], []
        for i, t in enumerate(terimler):
            if t in m.katsayilar.index:
                konum.append(len(terimler) - 1 - i + kayma[j])
                b.append(m.katsayilar[t])
                alt.append(ga.loc[t].iloc[0])
                ust.append(ga.loc[t].iloc[1])
        b, alt, ust = map(np.asarray, (b, alt, ust))
        ax.errorbar(
            b, konum, xerr=[b - alt, ust - b], fmt="o", color=renkler[j % len(renkler)], capsize=3, markersize=4, label=adlar[j] or None
        )
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(terimler)))
    ax.set_yticklabels([sira[t] for t in reversed(terimler)])
    ax.yaxis._kategorik = True
    if len(modeller) > 1:
        ax.legend(frameon=False, fontsize=8)
    _stil(ax, baslik, f"Katsayı ve %{int(guven * 100)} güven aralığı", None)
    return _bitir(fig, baslik)


def marjinal_grafik(marjlar, baslik: str | None = None, x_etiketi: str | None = None, y_etiketi: str | None = None) -> Sekil:
    """``ek.marjlar(...)`` sonucunu güven aralıklarıyla çizer (Stata ``marginsplot``)."""
    t = marjlar.tablo if hasattr(marjlar, "tablo") else marjlar
    fig, ax = _yeni()
    grup = [c for c in t.columns if c.startswith("grup:")]
    xsut = [c for c in t.columns if c.startswith("x:")]
    varsayilan_y = "Ortalama marjinal etki" if getattr(marjlar, "tur", "") == "etki" else "Tahmin edilen değer"
    if not xsut:
        if not grup:
            raise EkonometriHatasi("marginsplot için at() ya da bir faktör değişkeniyle (margins cinsiyet) hesaplanmış marjlar gerekli.")
        # Stata: faktör düzeyleri x ekseninde, nokta + güven aralığı
        konum = np.arange(len(t))
        hata = np.vstack([t["Marj"] - t["%95 GA alt"], t["%95 GA üst"] - t["Marj"]])
        ax.errorbar(konum, t["Marj"], yerr=hata, fmt="o-", color="#1f4e79", capsize=4, linewidth=1.2)
        ax.set_xticks(konum)
        etiket = [str(g) for g in t[grup[0]]]
        uzun = sum(len(e) for e in etiket) > 45
        ax.set_xticklabels(etiket, rotation=30 if uzun else 0, ha="right" if uzun else "center")
        ax.xaxis._kategorik = True
        _stil(ax, baslik, x_etiketi or grup[0][5:], y_etiketi or varsayilan_y)
        return _bitir(fig, baslik)
    xs = xsut[0]
    kumeler = t.groupby(grup[0], sort=False) if grup else [("", t)]
    renkler = ["#1f4e79", "#b22222", "#2e7d32", "#e08a00", "#6a1b9a", "#00838f"]
    for i, (g, d) in enumerate(kumeler):
        x = d[xs].astype(float).values
        ax.plot(x, d["Marj"], marker="o", color=renkler[i % len(renkler)], label=str(g) if grup else None)
        ax.fill_between(x, d["%95 GA alt"], d["%95 GA üst"], color=renkler[i % len(renkler)], alpha=0.15)
    if grup:
        ax.legend(title=grup[0][5:], frameon=False, fontsize=8)
    _stil(ax, baslik, x_etiketi or xs[2:], y_etiketi or varsayilan_y)
    return _bitir(fig, baslik)


# --------------------------------------------------------------------------- veri grafikleri
def zaman_grafigi(
    veri: pd.DataFrame, sutunlar: list[str] | str, zaman: str | None = None, baslik: str | None = None, etiketler: dict | None = None
) -> Sekil:
    """Zaman serisi çizgi grafiği (Stata ``tsline``). Zaman, ``zaman`` sütunundan, tsset bilgisinden ya da indeksten alınır."""
    sutunlar = [sutunlar] if isinstance(sutunlar, str) else list(sutunlar)
    et = {**veri.attrs.get("etiketler", {}), **(etiketler or {})}
    zaman = zaman or veri.attrs.get("tsset")
    if zaman and zaman in veri.columns:
        x = veri[zaman]
    else:
        x = veri.index
    x = (
        x.to_timestamp()
        if isinstance(x, pd.PeriodIndex)
        else (x.dt.to_timestamp() if hasattr(x, "dt") and isinstance(x.dtype, pd.PeriodDtype) else x)
    )
    fig, ax = _yeni(7.0, 3.8)
    for i, c in enumerate(sutunlar):
        ax.plot(x, veri[c].values, linewidth=1.4, label=et.get(c, c), color=["#1f4e79", "#b22222", "#2e7d32", "#e08a00"][i % 4])
    if len(sutunlar) > 1:
        ax.legend(frameon=False, fontsize=8)
    ax.xaxis._kategorik = True
    _stil(ax, baslik, None, et.get(sutunlar[0], sutunlar[0]) if len(sutunlar) == 1 else None)
    return _bitir(fig, baslik)


def histogram(veri: pd.DataFrame, sutun: str, normal: bool = True, kutu: int | None = None, baslik: str | None = None) -> Sekil:
    """Histogram ve (isteğe bağlı) normal dağılım eğrisi (Stata ``histogram x, normal``)."""
    from scipy import stats

    x = veri[sutun].dropna().astype(float).values
    fig, ax = _yeni()
    kutu = kutu or int(min(np.sqrt(len(x)), 10 * np.log10(len(x))))  # Stata'nın varsayılan kutu sayısı kuralı
    ax.hist(x, bins=max(kutu, 5), density=True, color="#9fb7d0", edgecolor="white")
    if normal:
        g = np.linspace(x.min(), x.max(), 200)
        ax.plot(g, stats.norm.pdf(g, x.mean(), x.std(ddof=1)), color="#b22222", linewidth=1.4, label="normal dağılım")
        ax.legend(frameon=False, fontsize=8)
    _stil(ax, baslik, veri.attrs.get("etiketler", {}).get(sutun, sutun), "Yoğunluk")
    return _bitir(fig, baslik)


def serpme(veri: pd.DataFrame, y: str, x: str, dogru: bool = True, baslik: str | None = None) -> Sekil:
    """Serpme grafiği ve EKK doğrusu (Stata ``twoway (scatter y x) (lfit y x)``)."""
    d = veri[[x, y]].dropna().astype(float)
    et = veri.attrs.get("etiketler", {})
    fig, ax = _yeni()
    ax.scatter(d[x], d[y], s=10, alpha=0.5, color="#1f4e79", edgecolors="none")
    if dogru:
        b = np.polyfit(d[x], d[y], 1)
        g = np.linspace(d[x].min(), d[x].max(), 100)
        ax.plot(g, np.polyval(b, g), color="#b22222", linewidth=1.4, label="EKK doğrusu")
        ax.legend(frameon=False, fontsize=8)
    _stil(ax, baslik, et.get(x, x), et.get(y, y))
    return _bitir(fig, baslik)


def roc_grafigi(model, baslik: str | None = "ROC eğrisi") -> Sekil:
    """Logit / probit için ROC eğrisi ve eğri altındaki alan (Stata ``lroc``)."""
    roc = model.roc()
    fig, ax = _yeni(4.8, 4.6)
    ax.plot(roc["1 − özgüllük"], roc["Duyarlılık"], color="#1f4e79", linewidth=1.6)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8)
    ax.text(0.55, 0.08, f"Eğri altındaki alan = {_virgul(round(roc.attrs['auc'], 4))}", fontsize=9)
    _stil(ax, baslik, "1 − özgüllük", "Duyarlılık")
    return _bitir(fig, baslik)


__all__ = [
    "Sekil",
    "kalinti_grafigi",
    "normallik_grafigi",
    "katsayi_grafigi",
    "marjinal_grafik",
    "zaman_grafigi",
    "histogram",
    "serpme",
    "roc_grafigi",
]
