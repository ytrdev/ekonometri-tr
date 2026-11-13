"""ARDL sınır testi: F istatistiği statsmodels UECM ile, kritik değerler PSS k tanımıyla."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.ardl import UECM

import ekonometri as ek
from ekonometri._pss import crit_vals

DURUM = [1, 2, 3, 4, 5]


def _seri(n=90, seed=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    e = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.4 * y[t - 1] + 0.3 * x[t] + 0.2 * x[t - 1] + e[t]
    return pd.DataFrame({"y": y, "x": x})


@pytest.mark.parametrize("durum", DURUM)
def test_f_statsmodels_ile_ayni(durum):
    veri = _seri()
    a = ek.ardl_sinir("y ~ x", veri, gecikme=2, durum=durum)
    b = UECM(veri["y"], 2, veri[["x"]], order=2, trend="c").fit().bounds_test(durum)
    assert np.isclose(a.F, float(b.statistic), rtol=1e-8, atol=1e-8)


def test_kritik_deger_pss_k_tanimini_kullanir():
    veri = _seri()
    a = ek.ardl_sinir("y ~ x", veri, gecikme=2, durum=3)
    b = UECM(veri["y"], 2, veri[["x"]], order=2, trend="c").fit().bounds_test(3)
    assert a.k == 1
    assert np.isclose(a.alt[5], crit_vals[(1, 3, False)][1])
    assert np.isclose(a.ust[5], crit_vals[(1, 3, True)][1])
    # statsmodels satırı k = 1 + nx üzerinden okur
    assert not np.isclose(a.alt[5], float(b.critical_values.loc[95, "lower"]))


def test_es_zamanli_olmayan_fark():
    veri = _seri()
    a = ek.ardl_sinir("y ~ x", veri, gecikme=2, durum=3, es_zamanli=False)
    b = UECM(veri["y"], 2, veri[["x"]], order=2, trend="c", causal=True).fit().bounds_test(3)
    assert np.isclose(a.F, float(b.statistic), rtol=1e-8, atol=1e-8)


def test_esbutunlesik_karar():
    rng = np.random.default_rng(11)
    n = 250
    x = np.cumsum(rng.normal(size=n))
    e = rng.normal(scale=0.4, size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = y[t - 1] - 0.45 * (y[t - 1] - 0.8 * x[t - 1]) + 0.3 * (x[t] - x[t - 1]) + e[t]
    veri = pd.DataFrame({"y": y, "x": x})
    a = ek.ardl_sinir("y ~ x", veri, gecikme=1, durum=3)
    assert a.karar == "esbutunlesik"
    assert a.hiz < 0
    assert a.uzun_donem()["x"] > 0


def test_rapor_ve_hatalar(tmp_path):
    veri = _seri(n=40)
    a = ek.ardl_sinir("y ~ x", veri, gecikme=1)
    metin = a.rapor(dil="en").markdown()
    assert "ARDL bounds test" in metin and "Pesaran" in metin
    assert "Table 1" in metin
    a.rapor(tmp_path / "ardl.docx", bicim="izgara")
    with pytest.raises(ek.EkonometriHatasi):
        ek.ardl_sinir("y ~ x", veri, durum=6)
    kisa = veri.iloc[:12]
    with pytest.raises(ek.EkonometriHatasi):
        ek.ardl_sinir("y ~ x", kisa, gecikme=2)
