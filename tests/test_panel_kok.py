"""xtunitroot llc / ips / fisher: R plm::purtest ile karşılaştırma."""

import numpy as np
import pandas as pd
import pytest

import ekonometri as ek


@pytest.fixture(scope="module")
def d():
    rng = np.random.default_rng(7)
    N, T = 12, 40
    rows = []
    for i in range(N):
        rho = 0.85 if i % 2 else 1.0
        y = np.zeros(T)
        e = rng.normal(size=T)
        for t in range(1, T):
            y[t] = 0.3 * i / N + rho * y[t - 1] + e[t] + 0.3 * e[t - 1]
        rows += [(i + 1, t + 1, y[t] + 0.5 * i) for t in range(T)]
    return ek.xtset(pd.DataFrame(rows, columns=["id", "yil", "y"]), "id", "yil")


# plm::purtest(pd$y, test="levinlin", exo=..., lags=...)  (plm tabloyu T ile okur → _llc_tablo="T")
@pytest.mark.parametrize(
    "det,L,ref",
    [
        ("sabit", 0, 0.2252148330),
        ("sabit", 2, -0.4192295724),
        ("trend", 0, 0.3670810734),
        ("trend", 2, -0.2739335585),
        ("yok", 0, 1.1698998779),
        ("yok", 2, -0.0483105807),
    ],
)
def test_llc_plm(d, det, L, ref):
    r = ek.panel_birim_kok(d, "y", "llc", gecikme=L, deterministik=det, _llc_tablo="T")
    assert np.isclose(r.istatistikler.loc["Düzeltilmiş t*", "Değer"], ref, atol=1e-9)


# plm::purtest(pd$y, test="ips", exo=..., lags=..., dfcor=TRUE) → W-t-bar ve t-bar
@pytest.mark.parametrize(
    "det,L,w,tbar",
    [
        ("sabit", 0, 0.6174252388, -1.3667063036),
        ("sabit", 2, -0.5643421610, -1.6205553324),
        ("trend", 0, 2.2210294369, -1.6526146693),
        ("trend", 2, 0.6430078633, -1.9551928868),
    ],
)
def test_ips_plm(d, det, L, w, tbar):
    r = ek.panel_birim_kok(d, "y", "ips", gecikme=L, deterministik=det)
    assert np.allclose(r.istatistikler["Değer"].values, [tbar, w], atol=1e-9)


# plm madwu / invnormal / logit / Pm, p.approx="MacKinnon1994" (Stata dfuller'ın p-değeri)
@pytest.mark.parametrize(
    "det,ref",
    [
        ("sabit", [36.8836690430, -1.2552293284, -1.3783996673, 1.8595974475]),
        ("trend", [29.5136038338, -0.2980794827, -0.2150260841, 0.7958201644]),
    ],
)
def test_fisher_plm(d, det, ref):
    r = ek.panel_birim_kok(d, "y", "fisher", gecikme=1, deterministik=det)
    assert np.allclose(r.istatistikler["Değer"].values, ref, atol=1e-8)


def test_aic_secim_ve_cikti(d):
    r = ek.panel_birim_kok(d, "y", "ips", gecikme="aic", en_fazla_gecikme=4)
    assert r.gecikmeler.tolist() == [1, 0, 1, 0, 0, 1, 1, 1, 2, 1, 0, 0]  # plm lagsel AIC
    s = str(ek.panel_birim_kok(d, "y", "llc", gecikme=1))
    assert "Düzeltilmiş t*" in s and "Bartlett" in s
    # birinci farkta birim kök reddedilmeli
    t = ek.panel_duraganlik(d, ["y"], gecikme=1)
    assert t.loc[t.Seviye == "Birinci fark", "IPS p"].iloc[0] < 0.01


def test_boyut_monte_carlo():
    """H0 altında (saf rassal yürüyüş) %5 reddetme oranı makul olmalı."""
    rng = np.random.default_rng(3)
    red = {"llc": 0, "ips": 0, "fisher": 0}
    R = 150
    for _ in range(R):
        y = rng.normal(size=(10, 50)).cumsum(axis=1)
        df = pd.DataFrame({"id": np.repeat(range(10), 50), "t": np.tile(range(50), 10), "y": y.ravel()})
        for t in red:
            red[t] += ek.panel_birim_kok(df, "y", t, "id", "t").p < 0.05
    # IPS ve Fisher nominal düzeye yakın; LLC küçük T'de (Bartlett bandı uzun dönem varyansı küçük tahmin eder) bir miktar
    # fazla reddeder — plm ile birebir aynı hesapta da böyledir.
    assert red["ips"] / R <= 0.10 and red["fisher"] / R <= 0.12 and red["llc"] / R <= 0.25, red


def test_hatalar(d):
    with pytest.raises(ek.EkonometriHatasi, match="dengeli"):
        ek.panel_birim_kok(d.iloc[1:], "y", "llc")
    with pytest.raises(ek.EkonometriHatasi, match="IPS"):
        ek.panel_birim_kok(d, "y", "ips", deterministik="yok")
    assert ek.panel_birim_kok(d.iloc[1:], "y", "ips").N == 12  # dengesiz panelde IPS çalışır
