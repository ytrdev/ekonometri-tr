"""ologit, oprobit, mlogit, poisson, nbreg, tobit: R (MASS::polr, nnet::multinom, glm, MASS::glm.nb, AER::tobit) ile karşılaştırma."""

import numpy as np
import pandas as pd
import pytest

import ekonometri as ek


@pytest.fixture(scope="module")
def d():
    """R referanslarının üretildiği simüle veri (aynı tohum)."""
    rng = np.random.default_rng(21)
    n = 800
    d = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n), "g": rng.choice(["A", "B", "C"], n)})
    lat = 0.8 * d.x - 0.5 * d.z + 0.4 * (d.g == "B") + rng.logistic(size=n)
    d["o"] = np.digitize(lat, [-1, 0.5, 2]) + 1
    u = np.column_stack(
        [
            np.zeros(n),
            0.5 + 0.8 * d.x + rng.gumbel(size=n) - rng.gumbel(size=n),
            -0.2 - 0.6 * d.z + 0.3 * d.x + rng.gumbel(size=n) - rng.gumbel(size=n),
        ]
    )
    d["m"] = np.array(["ev", "is", "okul"])[u.argmax(axis=1)]
    mu = np.exp(0.3 + 0.5 * d.x - 0.3 * d.z)
    d["c"] = rng.poisson(mu * rng.gamma(2, 0.5, n))
    d["t"] = np.maximum(1 + 2 * d.x - d.z + rng.normal(scale=1.5, size=n), 0)
    return d


SIRA = ["x", "z", "g[T.B]", "g[T.C]"]


@pytest.mark.parametrize(
    "islev,b,se,kesim,ll",
    [
        (
            ek.ologit,
            [0.7844219151, -0.5502019559, 0.2545846436, -0.0543388434],
            [0.0731278717, 0.0730022099, 0.1571317454, 0.1607906457],
            [-1.2236286769, 0.3871436205, 1.8169860480],
            -1008.0241197467,
        ),
        (
            ek.oprobit,
            [0.4646349308, -0.3183003731, 0.1354280140, -0.0535194138],
            [0.0423830087, 0.0422667608, 0.0934195059, 0.0950458788],
            [-0.7441062825, 0.2195340791, 1.0622766381],
            -1008.9300060877,
        ),
    ],
)
def test_sirali_r_polr(d, islev, b, se, kesim, ll):
    m = islev("o ~ x + z + g", d)
    assert np.allclose(m.katsayilar.reindex(SIRA).values, b, rtol=1e-6)
    assert np.allclose(m.standart_hatalar.reindex(SIRA).values, se, rtol=1e-4)  # iki taraf da sayısal Hessian
    assert np.allclose(m.ek_parametreler["Değer"].values, kesim, rtol=1e-6) and np.isclose(m.log_olabilirlik, ll, rtol=1e-11)
    ame = m.marjinal_etkiler()
    assert set(ame["sonuç"]) == {"1", "2", "3", "4"}
    assert np.allclose(ame.groupby("değişken")["AME"].sum().loc[["x", "z"]].values, 0, atol=1e-8)  # olasılık değişimleri toplamı 0


def test_mlogit_r_multinom(d):
    m = ek.mlogit("m ~ x + z", d, temel="ev")
    assert np.allclose(
        m.katsayilar.values, [0.9206198698, 0.8440756056, -0.3358141415, 0.4071449106, 0.5345070249, -0.7231755378], rtol=1e-6
    )
    assert np.allclose(
        m.standart_hatalar.values, [0.1022268954, 0.1085551679, 0.1080238121, 0.1112686123, 0.1149352593, 0.1181425672], rtol=1e-6
    )
    assert np.isclose(m.log_olabilirlik, -778.3683824484, rtol=1e-11)
    assert ek.mlogit("m ~ x + z", d).temel == "is"  # Stata: en sık kategori temel
    assert m.oranlar().shape[0] == 4


def test_poisson_nbreg_tobit(d):
    p = ek.poisson("c ~ x + z", d)
    assert np.allclose(p.katsayilar.values, [0.3622371930, 0.4862919831, -0.3171959867], rtol=1e-9)
    assert np.allclose(p.standart_hatalar.values, [0.0314985487, 0.0282970462, 0.0291195818], rtol=1e-8)
    assert np.isclose(p.log_olabilirlik, -1425.8783425990, rtol=1e-12)
    nb = ek.nbreg("c ~ x + z", d)
    assert np.allclose(nb.katsayilar.values, [0.3629594201, 0.4837774212, -0.3162572290], rtol=1e-7)
    assert np.isclose(nb.ek_parametreler.loc["α", "Değer"], 0.4740656720, rtol=1e-6) and np.isclose(
        nb.log_olabilirlik, -1334.0045853404, rtol=1e-11
    )
    assert nb.ek_bilgi["alfa_lr"][1] < 0.001  # aşırı yayılım var
    # SH: Stata nbreg gibi β ve α birlikte (tam bilgi matrisi); R glm.nb α'yı sabit tutar → küçük fark beklenir
    assert np.allclose(nb.standart_hatalar.values, [0.0400474758, 0.0403246692, 0.0407164362], rtol=5e-3)
    t = ek.tobit("t ~ x + z", d, alt=0)
    assert np.allclose(t.katsayilar.values, [0.9628781389, 2.0343058551, -0.9391522432], rtol=1e-7)
    assert np.allclose(t.standart_hatalar.values, [0.0622704209, 0.0693459698, 0.0625497199], rtol=1e-6)
    assert np.isclose(t.ek_parametreler.loc["/σ", "Değer"], 1.4542730726, rtol=1e-7) and np.isclose(
        t.log_olabilirlik, -1095.9122441299, rtol=1e-11
    )
    assert t.ek_bilgi["sol"] == int((d.t == 0).sum())


def test_dayanikli_yorum_rapor(d, tmp_path):
    for m in (
        ek.ologit("o ~ x + z + g", d, kovaryans="saglam"),
        ek.poisson("c ~ x + z", d, kovaryans="saglam"),
        ek.tobit("t ~ x + z", d, kume="g"),
    ):
        assert m.kovaryans_turu in ("saglam", "kume")
        assert len(m.yorumla().katsayi_satirlari) > 0
        assert "## Kaynakça" in str(m.rapor())
    ek.nbreg("c ~ x + z", d).rapor(tmp_path / "nb.docx")
    assert (tmp_path / "nb.docx").stat().st_size > 5000
    assert np.isclose(ek.poisson("c ~ x + z", d).oranlar().loc["x", "oran"], np.exp(0.4862919831), rtol=1e-8)


def test_hatalar(d):
    with pytest.raises(ek.EkonometriHatasi, match="3 kategorisi"):
        ek.ologit("t2 ~ x", d.assign(t2=(d.x > 0).astype(int)))
    with pytest.raises(ek.EkonometriHatasi, match="sayım"):
        ek.poisson("t ~ x", d)
    with pytest.raises(ek.EkonometriHatasi, match="Sansürlü"):
        ek.tobit("x ~ z", d, alt=-100)
