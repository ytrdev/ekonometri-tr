"""Zivot-Andrews: R urca::ur.za ile tüm aday kırılma tarihlerinde karşılaştırma."""

import numpy as np
import pandas as pd
import pytest

import ekonometri as ek


@pytest.fixture(scope="module")
def y():
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.normal(size=100))
    y[60:] += 4
    return y


# urca::ur.za(y, model, lag): teststat ve bpoint (kirpma = 0: urca tüm tarihleri tarar)
@pytest.mark.parametrize(
    "model,p,ist,tb",
    [
        ("sabit", 0, -5.2215532737, 21),
        ("sabit", 2, -4.7075323188, 21),
        ("trend", 0, -4.8209402996, 33),
        ("trend", 2, -4.3619143243, 33),
        ("ikisi", 0, -4.9785231634, 21),
        ("ikisi", 2, -4.5585985025, 21),
    ],
)
def test_urca_ile_ayni(y, model, p, ist, tb):
    r = ek.zivot_andrews(y, model, gecikme=p, kirpma=0)
    assert np.isclose(r.istatistik, ist, atol=1e-9) and r.kirilma_sirasi == tb and len(r.t_serisi) == 99


def test_kirilmayi_bulur_ve_rapor(tmp_path):
    rng = np.random.default_rng(8)
    n = 150
    t = np.arange(n)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.5 * e[i - 1] + rng.normal()
    s = pd.Series(0.05 * t + 5 * (t >= 70) + e, index=pd.period_range("1980Q1", periods=n, freq="Q"), name="gsyh")
    r = ek.zivot_andrews(s, "sabit", gecikme=0)
    assert r.kirilma_sirasi == 70 and str(r.kirilma) == "1997Q2" and r.birim_kok_reddedildi
    assert r.t_serisi.index.min() >= s.index[int(0.15 * n) - 1]  # kırpılmış tarama
    for g in ("aic", "bic", "t"):
        assert ek.zivot_andrews(s, "sabit", gecikme=g, en_fazla_gecikme=4).birim_kok_reddedildi
    rap = ek.zivot_andrews_raporu([r, ek.zivot_andrews(s.cumsum().rename("birikimli"), "ikisi", gecikme=1)], etiketler={"gsyh": "GSYH"})
    assert len(rap.tablolar[0].satirlar) == 2 and "Zivot ve Andrews (1992)" in str(rap) and "GSYH" in str(rap)
    r.rapor(tmp_path / "za.docx", dil="en")
    assert (tmp_path / "za.docx").exists()
    assert r.grafik().baslik


def test_boyut():
    rng = np.random.default_rng(8)
    red = sum(ek.zivot_andrews(np.cumsum(rng.normal(size=200)), "ikisi", gecikme=0).birim_kok_reddedildi for _ in range(200))
    assert 0.01 <= red / 200 <= 0.10


def test_varsayilanlar(y):
    r = ek.zivot_andrews(y)
    assert r.model == "sabit" and r.secim == "aic" and 0 <= r.gecikme <= int(12 * (100 / 100) ** 0.25)


def test_hatalar(y):
    with pytest.raises(ek.EkonometriHatasi, match="model"):
        ek.zivot_andrews(y, "C")
    with pytest.raises(ek.EkonometriHatasi, match="20 gözlem"):
        ek.zivot_andrews(y[:15])
    with pytest.raises(ek.EkonometriHatasi, match="gecikme"):
        ek.zivot_andrews(y, gecikme="hq")
