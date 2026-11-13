"""Ortak yardımcılar: anlamlılık ifadeleri, sayı biçimlendirme, küçük doğrusal cebir araçları."""

from __future__ import annotations

import contextvars

import numpy as np

DIL = contextvars.ContextVar("ekonometri_dil", default="tr")


def l(tr: str, en: str) -> str:  # noqa: E743
    """Etkin rapor diline göre cümle seçer. Varsayılan Türkçedir."""
    return en if DIL.get() == "en" else tr


ANLAMLILIK_DUZEYLERI = (0.01, 0.05, 0.10)


def yildiz(p: float) -> str:
    """p-değerine göre anlamlılık yıldızı: *** %1, ** %5, * %10."""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def anlamlilik_ifadesi(p: float) -> str:
    """p-değerini Türkçe cümle parçasına çevirir."""
    if p < 0.01:
        return "%1 düzeyinde istatistiksel olarak anlamlı"
    if p < 0.05:
        return "%5 düzeyinde istatistiksel olarak anlamlı"
    if p < 0.10:
        return "yalnızca %10 düzeyinde (zayıf) anlamlı"
    return "istatistiksel olarak anlamlı değil"


def sayi(x: float, basamak: int = 4) -> str:
    """Türkçe ondalık ayırıcıyla sayı: 1234.5 -> '1.234,5'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if abs(x) >= 1e6 or (abs(x) < 1e-4 and x != 0):
        s = f"{x:.{basamak}e}"
        return s if DIL.get() == "en" else s.replace(".", ",")
    s = f"{x:,.{basamak}f}"
    if DIL.get() == "en":
        return s
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def tablo_sayi(x: float) -> str:
    """Tablo hücresi: küçük değerlerde bilimsel gösterim yerine daha çok basamak (0,000060)."""
    if x != 0 and 1e-7 <= abs(x) < 0.001:
        s = f"{x:.6f}"
        return s if DIL.get() == "en" else s.replace(".", ",")
    return sayi(x, 4)


def p_yaz(p: float) -> str:
    if p < 0.0001:
        return "<0.0001" if DIL.get() == "en" else "<0,0001"
    return sayi(p, 4)


def p_cumle(p: float) -> str:
    """Cümle içi p gösterimi: 'p < 0,0001' ya da 'p = 0,0123'."""
    if p < 0.0001:
        return "p < 0.0001" if DIL.get() == "en" else "p < 0,0001"
    return f"p = {sayi(p, 4)}"


def yuzde(x: float, basamak: int = 1) -> str:
    return "%" + sayi(x, basamak)


def ekk_coz(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Sayısal olarak kararlı EKK çözümü (QR / en küçük kareler). Döner: (b, kalıntılar, rank)."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    rank = np.linalg.matrix_rank(X)
    return b, e, rank


def r_kare(y: np.ndarray, e: np.ndarray, sabit_var: bool = True) -> float:
    sst = np.sum((y - y.mean()) ** 2) if sabit_var else np.sum(y**2)
    return 1.0 - np.sum(e**2) / sst if sst > 0 else float("nan")


def sabit_var_mi(X: np.ndarray) -> bool:
    """Tasarım matrisinde sabit terim (tüm değerleri eşit, sıfır olmayan sütun) var mı?"""
    return bool(np.any(np.all(np.isclose(X, X[0:1, :]), axis=0) & ~np.isclose(X[0, :], 0)))


def benzersiz_indeks(veri):
    """Tekrarlanan satır indeksi varsa (ör. birleştirme sonrası) indeksi sıfırlar; .loc ile satır çoğalmasını önler."""
    if veri.index.is_unique:
        return veri
    attrs = dict(veri.attrs)
    yeni = veri.reset_index(drop=True)
    yeni.attrs = attrs
    return yeni
