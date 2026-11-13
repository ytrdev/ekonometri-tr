"""Yatay kesit bağımlılığı (Pesaran CD, LM) ve CIPS: R plm::pcdtest / cipstest ile karşılaştırma."""

import numpy as np
import pandas as pd
import pytest

import ekonometri as ek


@pytest.fixture(scope="module")
def d():
    rng = np.random.default_rng(9)
    N, T = 20, 50
    f = rng.normal(size=T).cumsum() * 0.3
    rows = []
    for i in range(N):
        lam = rng.uniform(0.5, 1.5)
        y = np.zeros(T)
        x = rng.normal(size=T)
        for t in range(1, T):
            y[t] = (0.9 if i % 2 else 1.0) * y[t - 1] + lam * (f[t] - f[t - 1]) + rng.normal()
        rows += [(i + 1, t + 1, y[t], x[t]) for t in range(T)]
    v = pd.DataFrame(rows, columns=["id", "t", "y", "x"])
    v["z"] = v.y + 0.5 * v.x
    return ek.xtset(v, "id", "t")


def test_cd_degisken_ve_fe_kalinti(d):
    r = ek.yatay_kesit_bagimliligi(d, "y")
    assert np.allclose(r.tablo["İstatistik"].values, [1531.8422664338, 68.8350558637, 5.1966857347], rtol=1e-10)
    assert np.isclose(r.ortalama_rho, 0.0533168707, atol=1e-10) and np.isclose(r.ortalama_mutlak_rho, 0.3415223172, atol=1e-10)
    assert r.bagimlilik_var and r.tablo.loc["Breusch-Pagan LM", "sd"] == 190
    fe = ek.panel("z ~ x", d, yontem="sabit")
    r = ek.yatay_kesit_bagimliligi(d, fe)
    assert np.allclose(r.tablo["İstatistik"].values, [1529.2978378406, 68.7045294309, 68.5004477983, 5.1550346078], rtol=1e-10)
    assert "CIPS" in r.karar()


# plm::cipstest(p$y, lags, type, model = "cmg", truncated)
@pytest.mark.parametrize(
    "det,p,ref",
    [
        ("sabit", 1, -2.0419267565),
        ("sabit", 2, -1.9758059858),
        ("trend", 1, -2.6529360149),
        ("trend", 2, -2.6563858343),
        ("yok", 1, -1.3365776728),
        ("yok", 2, -1.1780477420),
    ],
)
def test_cips_plm(d, det, p, ref):
    for kesik in (False, True):
        c = ek.cips(d, "y", gecikme=p, deterministik=det, kesik=kesik)
        assert np.isclose(c.istatistik, ref, atol=1e-9) and c.T == 50 - p - 1 and c.N == 20


def test_cips_kesik_ve_dengesiz_cd():
    rng = np.random.default_rng(2)
    N, T = 12, 13
    rows = []
    for i in range(N):
        y = np.cumsum(rng.normal(size=T)) * rng.uniform(0.2, 3)
        rows += [(i + 1, t + 1, y[t]) for t in range(T)]
    v = pd.DataFrame(rows, columns=["id", "t", "y"])
    assert np.isclose(ek.cips(v, "y", "id", "t", 1, "trend").istatistik, -3.0468314744, atol=1e-9)
    assert np.isclose(ek.cips(v, "y", "id", "t", 1, "trend", kesik=True).istatistik, -2.7784488240, atol=1e-9)
    u = v.drop(index=[3, 4, 20, 50, 51, 52]).reset_index(drop=True)
    r = ek.yatay_kesit_bagimliligi(u, "y", "id", "t")
    assert np.allclose(r.tablo["İstatistik"].values, [151.4806644695, 7.4401368502, 1.2666982304], rtol=1e-10)
    with pytest.raises(ek.EkonometriHatasi, match="dengeli"):
        ek.cips(u, "y", "id", "t")


def test_iki_donemlik_ortusme_atlanir():
    rng = np.random.default_rng(0)
    y1 = rng.normal(size=6)
    y3 = y1 + rng.normal(size=6)
    rows = [(1, t + 1, y1[t]) for t in range(6)] + [(3, t + 1, y3[t]) for t in range(6)]
    rows += [(2, 1, 0.0), (2, 2, 5.0)]  # 1 ve 3 ile yalnız iki dönem: korelasyon mekanik olarak ±1
    v = pd.DataFrame(rows, columns=["id", "t", "y"])
    r = ek.yatay_kesit_bagimliligi(v, "y", "id", "t")
    assert r.eksik_cift == 2 and np.isclose(r.ortalama_rho, np.corrcoef(y1, y3)[0, 1])
    assert "3'ten az" in str(r)
    kisa = pd.DataFrame([(i, t, float(t + i)) for i in (1, 2, 3) for t in (1, 2)], columns=["id", "t", "y"])
    with pytest.raises(ek.EkonometriHatasi, match="3 ortak dönem"):
        ek.yatay_kesit_bagimliligi(kisa, "y", "id", "t")


def test_kritik_deger_tablosu():
    from ekonometri.kesit import _cips_kritik

    # Pesaran (2007) Tablo II(b), sabitli: N = 20, T = 20 → %1 −2,40, %5 −2,21, %10 −2,10; N = 10, T = 10 kesiksiz %5 −2,52
    assert _cips_kritik(20, 20, "sabit", False) == {"%1": -2.40, "%5": -2.21, "%10": -2.10}
    assert _cips_kritik(10, 10, "sabit", False)["%5"] == -2.52
    with pytest.warns(UserWarning, match="aralığı dışında"):
        _cips_kritik(300, 20, "sabit", False)


def test_boyut():
    """Ortak faktörlü (yatay kesit bağımlı) rassal yürüyüşlerde CIPS boyutu nominal düzeye yakın; CD bağımsız seride ~%5."""
    rng = np.random.default_rng(4)
    red_c = red_cd = 0
    R = 150
    for _ in range(R):
        N, T = 20, 40
        f = rng.normal(size=T)
        e = rng.normal(size=(T, N)) + np.outer(f, rng.uniform(0.5, 1.5, N))
        Y = np.cumsum(e, axis=0)
        v = pd.DataFrame({"id": np.repeat(range(N), T), "t": np.tile(range(T), N), "y": Y.T.ravel()})
        red_c += ek.cips(v, "y", "id", "t", gecikme=1).birim_kok_reddedildi
        w = pd.DataFrame({"id": np.repeat(range(N), T), "t": np.tile(range(T), N), "y": rng.normal(size=N * T)})
        red_cd += ek.yatay_kesit_bagimliligi(w, "y", "id", "t").bagimlilik_var
    assert red_c / R <= 0.12 and 0.01 <= red_cd / R <= 0.10
