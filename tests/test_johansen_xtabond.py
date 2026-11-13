"""Johansen istatistikleri statsmodels ile; Arellano-Bond bilinen bir dinamik panelde katsayıyı bulur."""

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import ekonometri as ek


def test_johansen_statsmodels_ile_ayni():
    rng = np.random.default_rng(3)
    n = 120
    e = rng.normal(size=(n, 3))
    y = np.zeros((n, 3))
    for t in range(1, n):
        y[t] = y[t - 1] + e[t]
    y[:, 2] = 0.5 * y[:, 0] + 0.3 * y[:, 1] + rng.normal(size=n)
    veri = pd.DataFrame(y, columns=["a", "b", "c"])
    for det, kod in (("yok", -1), ("sabit", 0), ("trend", 1)):
        a = ek.johansen(veri, ["a", "b", "c"], gecikme=1, deterministik=det)
        b = coint_johansen(y, kod, 1)
        assert np.allclose(a.iz, b.trace_stat, rtol=1e-8, atol=1e-8)
        assert np.allclose(a.ozdeger_istatistik, b.max_eig_stat, rtol=1e-8, atol=1e-8)
        assert np.allclose(a.iz_kritik, b.trace_stat_crit_vals, rtol=1e-6, atol=1e-6)


def test_johansen_esbutunlesik_derece():
    rng = np.random.default_rng(9)
    n = 200
    x = np.cumsum(rng.normal(size=n))
    z = np.cumsum(rng.normal(size=n))
    y = 0.7 * x + 0.2 * z + rng.normal(scale=0.4, size=n)
    veri = pd.DataFrame({"y": y, "x": x, "z": z})
    a = ek.johansen(veri, ["y", "x", "z"], gecikme=1)
    assert a.derece >= 1
    beklenen = {"yok": "no deterministic terms", "sabit": "a constant", "trend": "a constant and a linear trend"}
    for kod, etiket in beklenen.items():
        ing = ek.johansen(veri, ["y", "x", "z"], gecikme=1, deterministik=kod).rapor(dil="en").markdown()
        assert "Johansen" in ing and etiket in ing and f"'{kod}'" not in ing


def test_xtabond_dinamik_katsayiyi_bulur():
    rng = np.random.default_rng(4)
    n, t, isinma = 250, 8, 30
    ys, xs, birim, zaman = [], [], [], []
    for i in range(n):
        etki = rng.normal()
        hata = rng.normal(size=t + isinma)
        x = rng.normal(size=t + isinma)
        y = 0.0
        for s in range(t + isinma):
            y = 0.45 * y + 0.3 * x[s] + etki + hata[s]
            if s >= isinma:
                ys.append(y)
                xs.append(x[s])
                birim.append(i)
                zaman.append(s - isinma)
    veri = pd.DataFrame({"birim": birim, "zaman": zaman, "y": ys, "x": xs})
    m = ek.xtabond("y ~ x", veri, birim="birim", zaman="zaman")
    assert abs(m.katsayilar["L.y"] - 0.45) < 0.08
    assert abs(m.katsayilar["x"] - 0.3) < 0.08
    assert m.sargan_sd > 0 and 0 <= m.sargan_p <= 1
    assert "Arellano and Bond" in m.rapor(dil="en").markdown()


def _ab_verisi():
    """R plm::pgmm ile karşılaştırılan simüle dinamik panel (aynı tohum, aynı veri)."""
    rng = np.random.default_rng(4)
    satirlar = []
    for i in range(120):
        a = rng.normal()
        y = a + rng.normal()
        for t in range(8):
            x = 0.5 * a + rng.normal()
            y = 0.5 * y + 0.8 * x + a + rng.normal()
            satirlar.append((i, 2000 + t, y, x))
    return pd.DataFrame(satirlar, columns=["i", "t", "y", "x"])


def test_xtabond_r_plm_ile_ayni():
    """Referans: R plm::pgmm(y ~ lag(y,1) + x | lag(y, 2:99), model = "onestep"), vcovHC ve mtest."""
    d = _ab_verisi()
    m = ek.xtabond("y ~ x", d, birim="i", zaman="t", kovaryans="saglam")
    assert np.allclose(m.katsayilar.values, [0.508652870749, 0.807287291692], rtol=1e-9)
    assert np.allclose(m.standart_hatalar.values, [0.066780467411, 0.046225530656], rtol=1e-8)
    assert np.isclose(m.ar_testleri[1][0], -6.296711866005, rtol=1e-8)
    assert np.isclose(m.ar_testleri[2][0], -1.291021370338, rtol=1e-8)


def test_xtabond_sargan_ve_klasik_sh_boyutu():
    """Doğru modelde Sargan ortalaması serbestlik derecesine yakın, klasik SH tahminlerin dağılımına uygun olmalı."""
    rng = np.random.default_rng(10)
    sargan, b, se = [], [], []
    for _ in range(60):
        satirlar = []
        for i in range(100):
            a = rng.normal()
            y = a + rng.normal()
            for t in range(6):
                x = 0.5 * a + rng.normal()
                y = 0.5 * y + 0.8 * x + a + rng.normal()
                satirlar.append((i, t, y, x))
        m = ek.xtabond("y ~ x", pd.DataFrame(satirlar, columns=["i", "t", "y", "x"]), birim="i", zaman="t")
        sargan.append(m.sargan)
        b.append(m.katsayilar.iloc[0])
        se.append(m.standart_hatalar.iloc[0])
    assert 7 < np.mean(sargan) < 11.5  # sd = 9 (eski hatalı ölçekte ~4,9 çıkıyordu)
    assert 0.8 < np.mean(se) / np.std(b) < 1.25
