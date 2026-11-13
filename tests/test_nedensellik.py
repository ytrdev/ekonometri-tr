"""Toda-Yamamoto: R (lm + car::linearHypothesis, Wald χ²) ile karşılaştırma."""

import numpy as np
import pandas as pd
import pytest

import ekonometri as ek


@pytest.fixture(scope="module")
def d():
    rng = np.random.default_rng(11)
    T = 150
    x = np.cumsum(rng.normal(size=T))
    z = np.cumsum(rng.normal(size=T))
    y = np.zeros(T)
    for t in range(2, T):
        y[t] = 0.6 * y[t - 1] + 0.3 * x[t - 1] - 0.2 * x[t - 2] + rng.normal()
    return pd.DataFrame({"y": y, "x": x, "z": z})


@pytest.mark.parametrize(
    "sut,p,dm,tr,ref",
    [
        (["y", "x"], 2, 1, False, [18.1209221324, 0.4768181728]),
        (
            ["y", "x", "z"],
            2,
            1,
            False,
            [
                17.6424479982,
                1.7289573000,
                19.8307922799,
                0.3957045597,
                0.3509373036,
                0.8189699965,
                0.3935519895,
                0.6712444278,
                1.1005705993,
            ],
        ),
        (["y", "x"], 3, 2, True, [23.0724449079, 4.4561423848]),
    ],
)
def test_r_ile_ayni(d, sut, p, dm, tr, ref):
    r = ek.toda_yamamoto(d, sut, gecikme=p, dmax=dm, trend=tr)
    assert np.allclose(r.tablo["χ²"].values, ref, rtol=1e-9)
    assert (r.tablo["sd"] == [p * (len(sut) - 1) if n == "Tümü" else p for n in r.tablo["Neden"]]).all()


def test_karar_secim_rapor(d, tmp_path):
    r = ek.toda_yamamoto(d, ["y", "x"], dmax=1)
    assert r.secim == "aic" and r.gecikme >= 1 and r.T == len(d) - r.gecikme - 1
    assert r.nedensel("x", "y") and not r.nedensel("y", "x")
    m = str(r.rapor(etiketler={"x": "Kur"}))
    assert "Toda ve Yamamoto (1995)" in m and "Kur" in m
    assert "Toda-Yamamoto" in str(r.rapor(dil="en"))
    r.rapor(tmp_path / "ty.docx")
    assert (tmp_path / "ty.docx").exists()


def test_boyut_monte_carlo():
    """H0 altında (bağımsız iki rassal yürüyüş) reddetme oranı nominal düzeye yakın olmalı."""
    rng = np.random.default_rng(5)
    red = 0
    R = 200
    for _ in range(R):
        e = rng.normal(size=(120, 2)).cumsum(axis=0)
        red += ek.toda_yamamoto(pd.DataFrame(e, columns=["a", "b"]), ["a", "b"], gecikme=1, dmax=1).tablo["p"].iloc[0] < 0.05
    assert 0.01 <= red / R <= 0.10


def test_hatalar(d):
    with pytest.raises(ek.EkonometriHatasi, match="en az 1"):
        ek.toda_yamamoto(d, ["y", "x"], gecikme=0)
    with pytest.raises(ek.EkonometriHatasi, match="bilgi_olcutu"):
        ek.toda_yamamoto(d, ["y", "x"], bilgi_olcutu="xyz")
    with pytest.raises(ek.EkonometriHatasi, match="yetersiz"):
        ek.toda_yamamoto(d.head(12), ["y", "x", "z"], gecikme=3, dmax=1)
