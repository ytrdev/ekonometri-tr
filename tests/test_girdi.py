"""Kendi verini yükleme: Stata .dta (etiketler), Excel, Türkçe CSV; xtset / tsset; L. ve D. işlemleri."""

import numpy as np
import pandas as pd
import pytest

import ekonometri as ek


@pytest.fixture
def ornek():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "ucret": rng.integers(20000, 90000, 50).astype(float),
            "egitim": rng.integers(5, 20, 50),
            "cinsiyet": rng.choice([1, 2], 50),
        }
    )


def test_dta_etiketler_ve_deger_etiketleri(tmp_path, ornek):
    yol = tmp_path / "anket.dta"
    pd.io.stata.StataWriterUTF8(
        yol,
        ornek,
        write_index=False,
        variable_labels={"ucret": "Aylık net ücret (TL)", "egitim": "Eğitim süresi (yıl)"},
        value_labels={"cinsiyet": {1: "Erkek", 2: "Kadın"}},
        version=118,
    ).write_file()
    v = ek.oku(yol)
    assert v.attrs["etiketler"]["ucret"] == "Aylık net ücret (TL)"
    assert isinstance(v["cinsiyet"].dtype, pd.CategoricalDtype)
    assert list(v["cinsiyet"].cat.categories) == ["Erkek", "Kadın"]  # temel düzey = en küçük kod (Stata)
    m = ek.ekk("np.log(ucret) ~ egitim + cinsiyet", v)
    assert "cinsiyet[T.Kadın]" in m.katsayilar.index
    metin = str(m.rapor())
    assert "Eğitim süresi (yıl) değişken" in metin and "Aylık net ücret (TL) değişken" in metin  # etiketler metinde
    assert "| Eğitim süresi (yıl) |" in metin


def test_kaydet_ve_geri_oku(tmp_path, ornek):
    ek.etiketle(ornek, {"ucret": "Ücret"})
    ek.kaydet(ornek, tmp_path / "x.dta")
    assert ek.oku(tmp_path / "x.dta").attrs["etiketler"]["ucret"] == "Ücret"
    ek.kaydet(ornek, tmp_path / "x.xlsx")
    v = ek.oku(tmp_path / "x.xlsx")
    assert list(v.columns) == ["ucret", "egitim", "cinsiyet"] and len(v) == 50


def test_excel_basliklari_duzeltilir(tmp_path):
    pd.DataFrame({"Aylık Ücret (TL)": [1.0, 2, 3], "Eğitim": [8, 12, 16], "2023 Satış": [5, 6, 7]}).to_excel(
        tmp_path / "h.xlsx", index=False
    )
    v = ek.oku(tmp_path / "h.xlsx")
    assert list(v.columns) == ["aylik_ucret_tl", "Eğitim", "v2023_satis"]
    assert v.attrs["etiketler"]["aylik_ucret_tl"] == "Aylık Ücret (TL)"


def test_turkce_csv(tmp_path):
    metin = "il;yil;gsyh;issizlik\nAnkara;2022;1.234,5;9,1\nİzmir;2022;987,25;11,4\n"
    (tmp_path / "t.csv").write_bytes(metin.encode("cp1254"))
    v = ek.oku(tmp_path / "t.csv")
    assert v["gsyh"].tolist() == [1234.5, 987.25] and v["issizlik"].tolist() == [9.1, 11.4]
    assert v["il"].tolist() == ["Ankara", "İzmir"]
    (tmp_path / "u.csv").write_text("a,b\n1.5,2\n3.5,4\n", encoding="utf-8")
    assert ek.oku(tmp_path / "u.csv")["a"].tolist() == [1.5, 3.5]


def test_hatalar(tmp_path):
    with pytest.raises(ek.EkonometriHatasi, match="bulunamadı"):
        ek.oku(tmp_path / "yok.dta")
    (tmp_path / "a.json").write_text("{}")
    with pytest.raises(ek.EkonometriHatasi, match="Desteklenmeyen"):
        ek.oku(tmp_path / "a.json")


def test_xtset_ve_panel_varsayilanlari(capsys):
    v = ek.veri.yukle("firma").sample(frac=1, random_state=3)
    x = ek.xtset(v, "firma", "yil")
    assert "dengeli, 150 birim" in capsys.readouterr().out
    assert x.attrs["xtset"] == ("firma", "yil") and x["yil"].iloc[:10].tolist() == list(range(2015, 2025))
    m1 = ek.panel("np.log(satis) ~ np.log(calisan) + arge", x)
    m2 = ek.panel("np.log(satis) ~ np.log(calisan) + arge", v, birim="firma", zaman="yil")
    assert np.allclose(m1.katsayilar.values, m2.katsayilar.values)
    f = "np.log(satis) ~ np.log(calisan) + np.log(sermaye) + arge"
    assert ek.panel_analizi(f, x).onerilen == ek.panel_analizi(f, v, birim="firma", zaman="yil").onerilen == "sabit"
    with pytest.raises(ek.EkonometriHatasi, match="xtset"):
        ek.panel("np.log(satis) ~ arge", v)
    with pytest.raises(ek.EkonometriHatasi, match="repeated"):
        ek.xtset(pd.concat([v, v.head(1)]), "firma", "yil", sessiz=True)


def test_gecikme_ve_fark_stata_mantigi():
    d = pd.DataFrame({"il": ["A"] * 4 + ["B"] * 3, "yil": [2001, 2002, 2004, 2005, 2001, 2002, 2003], "x": [1.0, 2, 4, 5, 10, 20, 30]})
    d = ek.xtset(d, "il", "yil", sessiz=True)
    L = ek.gecikme(d, "x").tolist()
    # 2004'ün bir önceki yılı (2003) A için yok → eksik (Stata L.x)
    assert np.isnan(L[0]) and L[1] == 1 and np.isnan(L[2]) and L[3] == 4 and np.isnan(L[4]) and L[5] == 10
    D = ek.fark(d, "x").tolist()
    assert D[1] == 1 and np.isnan(D[2]) and D[6] == 10
    L2 = ek.gecikme(d, "x", 2).tolist()
    assert np.isnan(L2[3]) and L2[6] == 10  # A için 2003 yok → eksik; B 2003 → 2001
    with pytest.raises(ek.EkonometriHatasi, match="tsset"):
        ek.gecikme(pd.DataFrame({"x": [1, 2]}), "x")


def test_tsset_otokorelasyon_testlerini_acar():
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"yil": range(1980, 2024), "x": rng.normal(size=44)})
    d["y"] = 1 + d["x"] + rng.normal(size=44)
    d = ek.tsset(d, "yil", sessiz=True)
    adlar = [t.ad for t in ek.ekk("y ~ x", d).varsayimlar().testler]
    assert "Durbin-Watson" in adlar
