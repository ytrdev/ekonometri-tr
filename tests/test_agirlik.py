"""Ağırlıklar (Stata [aw=], [fw=], [pw=]): statsmodels WLS ve gözlem çoğaltma ile karşılaştırma."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import ekonometri as ek


@pytest.fixture(scope="module")
def veri():
    rng = np.random.default_rng(12)
    n = 400
    d = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n), "g": rng.integers(0, 25, n)})
    d["w"] = rng.uniform(0.2, 3.0, n)
    d["f"] = rng.integers(1, 5, n)
    d["y"] = 1 + 2 * d["x"] - d["z"] + rng.normal(size=n) * (1 + d["x"].abs())
    d["b"] = (0.3 * d["x"] - 0.5 * d["z"] + rng.logistic(size=n) > 0).astype(int)
    return d


def _wls(d, w):
    return sm.WLS(d["y"], sm.add_constant(d[["x", "z"]]), weights=w)


def test_aw_statsmodels_wls(veri):
    m = ek.ekk("y ~ x + z", veri, agirlik="w", agirlik_turu="aw")
    r = _wls(veri, veri["w"]).fit()
    assert np.allclose(m.katsayilar.values, r.params.values) and np.allclose(m.standart_hatalar.values, r.bse.values)
    assert np.isclose(m.r2, r.rsquared) and m.n == len(veri)
    h = ek.ekk("y ~ x + z", veri, agirlik="w", agirlik_turu="aw", kovaryans="HC1")
    assert np.allclose(h.standart_hatalar.values, _wls(veri, veri["w"]).fit(cov_type="HC1").bse.values)


def test_pw_otomatik_dayanikli_ve_kume(veri):
    m = ek.ekk("y ~ x + z", veri, agirlik="w", agirlik_turu="pw")
    assert m.kovaryans_turu == "HC1"
    assert np.allclose(m.standart_hatalar.values, _wls(veri, veri["w"]).fit(cov_type="HC1").bse.values)
    k = ek.ekk("y ~ x + z", veri, agirlik="w", agirlik_turu="pw", kume="g")
    r = _wls(veri, veri["w"]).fit(cov_type="cluster", cov_kwds={"groups": veri["g"]}, use_t=True)
    assert np.allclose(k.standart_hatalar.values, r.bse.values) and k.cikarim_sd == 24
    with pytest.raises(ek.EkonometriHatasi, match="varsayım"):
        m.varsayimlar()
    assert "## Varsayım Testleri" not in str(m.rapor()) and "örneklem (olasılık) ağırlıkları" in str(m.rapor())


def test_fw_gozlem_coğaltma_ile_ayni(veri):
    m = ek.ekk("y ~ x + z", veri, agirlik="f", agirlik_turu="fw")
    cok = veri.loc[veri.index.repeat(veri["f"])]
    r = ek.ekk("y ~ x + z", cok)
    assert m.n == len(cok) and np.allclose(m.katsayilar.values, r.katsayilar.values)
    assert np.allclose(m.standart_hatalar.values, r.standart_hatalar.values) and np.isclose(m.r2, r.r2)
    h = ek.ekk("y ~ x + z", veri, agirlik="f", agirlik_turu="fw", kovaryans="HC1")
    assert np.allclose(h.standart_hatalar.values, ek.ekk("y ~ x + z", cok, kovaryans="HC1").standart_hatalar.values)
    with pytest.raises(ek.EkonometriHatasi, match="tam sayı"):
        ek.ekk("y ~ x", veri, agirlik="w", agirlik_turu="fw")


def test_logit_fw_ve_pw(veri):
    m = ek.logit("b ~ x + z", veri, agirlik="f", agirlik_turu="fw")
    cok = veri.loc[veri.index.repeat(veri["f"])]
    r = ek.logit("b ~ x + z", cok)
    assert m.n == len(cok)
    assert np.allclose(m.katsayilar.values, r.katsayilar.values, rtol=1e-7) and np.allclose(
        m.standart_hatalar.values, r.standart_hatalar.values, rtol=1e-6
    )
    assert np.isclose(m.log_olabilirlik, r.log_olabilirlik) and np.isclose(m.pseudo_r2, r.pseudo_r2)
    assert np.allclose(m.marjinal_etkiler()["AME"].values, r.marjinal_etkiler()["AME"].values, rtol=1e-6)
    rs = ek.logit("b ~ x + z", veri, agirlik="f", agirlik_turu="fw", kovaryans="saglam")
    assert np.allclose(rs.standart_hatalar.values, ek.logit("b ~ x + z", cok, kovaryans="saglam").standart_hatalar.values, rtol=1e-6)
    p = ek.logit("b ~ x + z", veri, agirlik="w", agirlik_turu="pw")
    assert p.kovaryans_turu == "saglam"
    g = sm.GLM(veri["b"], sm.add_constant(veri[["x", "z"]]), family=sm.families.Binomial(), freq_weights=veri["w"]).fit()
    assert np.allclose(p.katsayilar.values, g.params.values, rtol=1e-7)  # pw katsayıları ağırlıklı olabilirlikle aynı
    with pytest.raises(ek.EkonometriHatasi, match="aw"):
        ek.logit("b ~ x", veri, agirlik="w", agirlik_turu="aw")


def test_stata_agirlik_sozdizimi(veri):
    s = ek.Stata(veri, yazdir=False)
    m = s("regress y x z [aw=w]")
    assert np.allclose(m.katsayilar.values, _wls(veri, veri["w"]).fit().params.values)
    assert s("regress y x z [pweight = w]").kovaryans_turu == "HC1"
    assert s("regress y x z [fw=f]").n == int(veri["f"].sum())
    assert s("logit b x z [pw=w]").kovaryans_turu == "saglam"
    t = s("summarize x [aw=w]")
    w = veri["w"] * len(veri) / veri["w"].sum()
    ort = np.sum(w * veri["x"]) / np.sum(w)
    assert np.isclose(t.loc["x", "Ortalama"], ort)
    assert np.isclose(t.loc["x", "Std. sapma"], np.sqrt(np.sum(w * (veri["x"] - ort) ** 2) / (len(veri) - 1)))
    tf = s("summarize x [fw=f]")
    cok = veri.loc[veri.index.repeat(veri["f"]), "x"]
    assert tf.loc["x", "N"] == len(cok) and np.isclose(tf.loc["x", "Std. sapma"], cok.std())


def test_tekrarli_indeks_sonucu_bozmaz(veri):
    """Birleştirme sonrası tekrarlanan satır indeksi gözlemleri çoğaltmamalı."""
    iki = pd.concat([veri, veri])  # indeksler tekrarlı
    m = ek.ekk("y ~ x + z", iki)
    assert m.n == 2 * len(veri) and len(m._veri) == 2 * len(veri)
    lg = ek.logit("b ~ x + z", iki)
    assert np.isclose(
        lg.marjinal_etkiler()["AME"].iloc[0], ek.logit("b ~ x + z", iki.reset_index(drop=True)).marjinal_etkiler()["AME"].iloc[0]
    )
