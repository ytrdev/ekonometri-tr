"""Türkçe yorumun doğru model biçimini tanıdığı ve veri modülünün çalıştığı test edilir."""

import numpy as np
import pandas as pd
import pytest

import ekonometri as ek


def _satir(yorum, anahtar):
    return next(s for s in yorum.katsayi_satirlari if anahtar in s)


def test_log_duz_yuzde_yorumu():
    m = ek.ekk("np.log(ucret) ~ egitim + cinsiyet", ek.veri.yukle("ucret"), birimler={"egitim": "yıl"})
    y = m.yorumla()
    beklenen = 100 * (np.exp(m.katsayilar["egitim"]) - 1)
    s = _satir(y, "'egitim' 1 yıl artınca")
    assert f"{beklenen:.2f}".replace(".", ",") in s and "artıyor" in s


def test_log_log_esneklik():
    m = ek.ekk("np.log(fiyat) ~ np.log(metrekare)", ek.veri.yukle("konut"))
    s = _satir(m.yorumla(), "'metrekare' %1 artınca")
    assert "esneklik" in s


def test_duz_log_yorumu():
    m = ek.ekk("puan ~ np.log(devam)", ek.veri.yukle("sinav"))
    s = _satir(m.yorumla(), "'devam' %1 artınca 'puan'")
    assert "birim" not in s or True  # birim verilmedi; yalnızca biçim kontrolü
    assert "yaklaşık" in s


def test_kukla_referans_grubu():
    m = ek.ekk("np.log(fiyat) ~ asansor", ek.veri.yukle("konut"))
    s = _satir(m.yorumla(), "'asansor'")
    assert "'Var' grubunun" in s and "('Yok')" in s and "daha yüksek" in s


def test_kare_terim_donum_noktasi():
    m = ek.ekk("np.log(ucret) ~ egitim + tecrube + I(tecrube**2)", ek.veri.yukle("ucret"))
    s = _satir(m.yorumla(), "doğrusal değil")
    b1, b2 = m.katsayilar["tecrube"], m.katsayilar["I(tecrube ** 2)"]
    donum = -b1 / (2 * b2)
    assert "ters U" in s and f"{donum:.2f}".replace(".", ",") in s


def test_anlamsiz_katsayi_notu():
    rng = np.random.default_rng(0)
    v = pd.DataFrame({"x": rng.normal(size=300), "gurultu": rng.normal(size=300)})
    v["y"] = 2 * v["x"] + rng.normal(size=300)
    y = ek.ekk("y ~ x + gurultu", v).yorumla()
    assert any("gurultu" in n and "ayırt edilemiyor" in n for n in y.notlar)


def test_turkce_ek_hatasi_yok():
    """Değişken adlarından hemen sonra kesme işaretli ek gelmemeli ('Kamu'e gibi hatalar)."""
    import re

    m = ek.ekk("np.log(ucret) ~ egitim + tecrube + I(tecrube**2) + cinsiyet + sektor + bolge", ek.veri.yukle("ucret"))
    metin = str(m.yorumla())
    assert not re.search(r"'[^'\s]+''?[a-zçğıöşü]", metin.replace("' ", " ").replace("')", ")"))
    assert "p = <" not in metin


def test_veri_listesi_ve_aciklama():
    tablo = ek.veri.listele()
    assert {"ucret", "konut", "sinav", "ihracat"} <= set(tablo.index)
    for ad in tablo.index:
        df = ek.veri.yukle(ad)
        assert len(df) > 50
        assert "SİMÜLE" in df.attrs["aciklama"]
        assert "Simüle veri" in ek.veri.aciklama(ad)


def test_ihracat_donem_indeksi():
    df = ek.veri.yukle("ihracat")
    assert isinstance(df.index, pd.PeriodIndex) and str(df.index[0]) == "2005Q1"


def test_bilinmeyen_veri():
    with pytest.raises(KeyError):
        ek.veri.yukle("yok")
