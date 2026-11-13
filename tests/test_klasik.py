"""t testi, χ², ANOVA (scipy ile), Newey-West (statsmodels ile), Prais-Winsten ve xttest0; Stata veri düzenleme komutları."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

import ekonometri as ek


@pytest.fixture(scope="module")
def ucret():
    return ek.veri.yukle("ucret")


def test_ttest_scipy_ile_ayni(ucret):
    erkek = ucret.loc[ucret.cinsiyet == "Erkek", "ucret"]
    kadin = ucret.loc[ucret.cinsiyet == "Kadın", "ucret"]
    for esit in (True, False):
        t = ek.ttest(ucret, "ucret", by="cinsiyet", esit_varyans=esit)
        r = stats.ttest_ind(erkek, kadin, equal_var=esit)
        assert np.isclose(t.t, r.statistic) and np.isclose(t.p_iki, r.pvalue) and np.isclose(t.fark, erkek.mean() - kadin.mean())
    t1 = ek.ttest(ucret, "egitim", deger=12)
    r1 = stats.ttest_1samp(ucret["egitim"], 12)
    assert np.isclose(t1.t, r1.statistic) and np.isclose(t1.p_iki, r1.pvalue) and t1.sd == len(ucret) - 1
    assert np.isclose(t1.p_sol + t1.p_sag, 1)
    te = ek.ttest(ucret, "egitim", ikinci="tecrube")
    assert np.isclose(te.t, stats.ttest_rel(ucret["egitim"], ucret["tecrube"]).statistic)
    with pytest.raises(ek.EkonometriHatasi, match="2 düzeyi"):
        ek.ttest(ucret, "ucret", by="bolge")


def test_ki_kare_ve_anova(ucret):
    k = ek.ki_kare(ucret, "cinsiyet", "sektor")
    chi2, p, sd, _ = stats.chi2_contingency(pd.crosstab(ucret.cinsiyet, ucret.sektor), correction=False)
    assert np.isclose(k.ki_kare, chi2) and np.isclose(k.p_degeri, p) and k.sd == sd
    k3 = ek.ki_kare(ucret, "bolge", "sektor")
    assert np.isclose(k3.cramer_v, np.sqrt(k3.ki_kare / (len(ucret) * 1)))
    a = ek.anova(ucret, "ucret", "bolge")
    r = stats.f_oneway(*[g["ucret"].values for _, g in ucret.groupby("bolge", observed=True)])
    assert np.isclose(a.F, r.statistic) and np.isclose(a.p_degeri, r.pvalue) and a.sd == (6, 1493)


def test_newey_statsmodels_ile_ayni():
    v = ek.veri.yukle("ihracat")
    m = ek.ekk("np.log(ihracat) ~ np.log(reel_kur) + dunya_buyumesi", v, kovaryans="newey", gecikme=3)
    X = sm.add_constant(np.column_stack([np.log(v.reel_kur), v.dunya_buyumesi]))
    r = sm.OLS(np.log(v.ihracat).values, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": True})
    assert np.allclose(m.standart_hatalar.values, r.bse)
    with pytest.raises(ek.EkonometriHatasi, match="gecikme"):
        ek.ekk("ihracat ~ reel_kur", v, kovaryans="newey")


def test_prais_winsten():
    rng = np.random.default_rng(3)
    n = 400
    x = np.cumsum(rng.normal(size=n)) * 0.1 + rng.normal(size=n)
    u = np.zeros(n)
    for i in range(1, n):
        u[i] = 0.7 * u[i - 1] + rng.normal()
    d = pd.DataFrame({"t": range(n), "x": x, "y": 1 + 2 * x + u})
    m = ek.prais("y ~ x", d)
    assert abs(m.rho - 0.7) < 0.08 and abs(m.katsayilar["x"] - 2) < 0.1
    # sabit nokta: son ρ, son katsayıların kalıntılarından yeniden tahmin edilen ρ ile aynı; katsayılar dönüştürülmüş EKK
    e = d["y"].values - m.X.values @ m.katsayilar.values
    assert np.isclose(m.rho, e[1:] @ e[:-1] / (e[:-1] @ e[:-1]), atol=1e-5)
    c = ek.prais("y ~ x", d, corc=True)
    assert c.n == n - 1 and c.model_turu == "Cochrane-Orcutt"
    assert m.dw_donusmus > m.dw_ilk
    metin = str(m.rapor())
    assert "Prais-Winsten genelleştirilmiş en küçük kareler" in metin and "## Varsayım Testleri" not in metin


def test_xttest0():
    v = ek.veri.yukle("firma")
    re_ = ek.panel("np.log(satis) ~ np.log(calisan) + arge", v, birim="firma", zaman="yil", yontem="rassal")
    t = ek.xttest0(re_)
    from ekonometri.panel import bp_lm_testi

    h = ek.panel("np.log(satis) ~ np.log(calisan) + arge", v, birim="firma", zaman="yil", yontem="havuz")
    assert np.isclose(t.istatistik, bp_lm_testi(h).istatistik) and np.isclose(t.p_degeri, bp_lm_testi(h).p_degeri / 2)


def test_stata_test_komutlari(ucret):
    s = ek.Stata(ucret, yazdir=False)
    assert np.isclose(s("ttest ucret, by(cinsiyet) unequal").t, ek.ttest(ucret, "ucret", by="cinsiyet", esit_varyans=False).t)
    assert s("ttest egitim == 12").sd == 1499
    assert s("ttest egitim == tecrube").tur == "eslesik"
    assert s("tab cinsiyet sektor, chi2").sd == 1
    assert s("oneway ucret bolge").sd == (6, 1493)


def test_stata_newey_prais_xttest0():
    s = ek.Stata(yazdir=False)
    s("sysuse ihracat")
    with pytest.raises(ek.EkonometriHatasi, match="lag"):
        s("newey ihracat reel_kur")
    m = s("newey ihracat reel_kur, lag(2)")
    assert m.kovaryans_turu == "newey"
    assert s("prais ihracat reel_kur").model_turu == "Prais-Winsten"
    assert s("prais ihracat reel_kur, corc").model_turu == "Cochrane-Orcutt"
    s("sysuse firma")
    s("xtset firma yil")
    s("xtreg satis calisan, re")
    assert s("xttest0").istatistik > 0


# --------------------------------------------------------------------------- veri düzenleme
@pytest.fixture
def s(tmp_path):
    iller = pd.DataFrame({"il": ["A", "B", "C"], "nufus": [100, 200, 300], "bolge": ["X", "X", "Y"]})
    ek.kaydet(iller, tmp_path / "iller.dta")
    ek.kaydet(pd.DataFrame({"il": ["D"], "yil": [2020], "gsyh": [9.0]}), tmp_path / "ek.dta")
    st = ek.Stata(
        pd.DataFrame({"il": ["A", "A", "B", "B", "E"], "yil": [2020, 2021, 2020, 2021, 2020], "gsyh": [1.0, 2, 3, 4, 5]}), yazdir=False
    )
    st.dizin = tmp_path
    return st


def test_merge_append(s):
    sayim = s(f'merge m:1 il using "{s.dizin / "iller.dta"}"')
    v = s.veri
    assert dict(sayim) == {3: 4, 1: 1, 2: 1}
    assert v.loc[v.il == "A", "nufus"].tolist() == [100, 100] and v.loc[v.il == "C", "_merge"].item() == 2
    s("drop _merge")
    with pytest.raises(ek.EkonometriHatasi, match="tekil"):
        s(f'merge 1:1 il using "{s.dizin / "iller.dta"}"')
    s(f'append using "{s.dizin / "ek.dta"}"')
    assert len(s.veri) == 7 and s.veri["il"].iloc[-1] == "D"


def test_merge_keep_nogenerate(s):
    s(f'merge m:1 il using "{s.dizin / "iller.dta"}", keep(match) nogenerate keepusing(nufus)')
    assert len(s.veri) == 4 and "_merge" not in s.veri.columns and "bolge" not in s.veri.columns


def test_reshape_ve_collapse(s):
    s("reshape wide gsyh, i(il) j(yil)")
    assert list(s.veri.columns) == ["il", "gsyh2020", "gsyh2021"] and len(s.veri) == 3
    assert np.isnan(s.veri.loc[s.veri.il == "E", "gsyh2021"].item())
    s("reshape long gsyh, i(il) j(yil)")
    assert len(s.veri) == 6 and list(s.veri.columns[:2]) == ["il", "yil"]
    s("collapse (mean) ort=gsyh (sum) toplam=gsyh (count) n=gsyh, by(il)")
    a = s.veri.set_index("il")
    assert a.loc["A", "ort"] == 1.5 and a.loc["A", "toplam"] == 3 and a.loc["E", "n"] == 1


def test_encode_destring_label_recode_duplicates():
    st = ek.Stata(
        pd.DataFrame({"il": ["Konya", "Adana", "Konya", "Bursa"], "gelir": ["1.234,5", "2.000", "abc", "10,25"], "k": [1, 2, 3, 9]}),
        yazdir=False,
    )
    st("encode il, gen(il_kod)")
    assert list(st.veri["il_kod"].cat.categories) == ["Adana", "Bursa", "Konya"]
    assert st.veri.attrs["deger_etiketleri"]["il_kod"] == {1: "Adana", 2: "Bursa", 3: "Konya"}
    assert st("count if il_kod == 3") == 2
    st("destring gelir, replace dpcomma")  # 'abc' sayı değil → force olmadan değişmez
    assert not pd.api.types.is_numeric_dtype(st.veri["gelir"])
    st("destring gelir, gen(g2) dpcomma force")
    assert st.veri["g2"].tolist()[:2] == [1234.5, 2000.0] and np.isnan(st.veri["g2"].iloc[2]) and st.veri["g2"].iloc[3] == 10.25
    st('label define evet 1 "Evet" 2 "Hayır"')
    st("label values k evet")
    assert list(st.veri["k"].astype(str)) == ["Evet", "Hayır", "3", "9"] and st("count if k == 1") == 1
    st2 = ek.Stata(pd.DataFrame({"x": [1, 2, 3, 4, 5, np.nan], "y": [1, 1, 2, 2, 2, 2]}), yazdir=False)
    st2("recode x (1 2 = 0) (3/max = 1) (missing = 9), gen(xr)")
    assert st2.veri["xr"].tolist() == [0, 0, 1, 1, 1, 9]
    st2("duplicates drop y, force")
    assert len(st2.veri) == 2
    st2("order y")
    assert list(st2.veri.columns)[0] == "y"
