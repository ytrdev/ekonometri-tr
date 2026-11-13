"""VAR, panel birim kök, yatay kesit bağımlılığı ve CIPS raporları / yorumları."""

import numpy as np
import pytest

import ekonometri as ek


@pytest.fixture(scope="module")
def f():
    v = ek.xtset(ek.veri.yukle("firma"), "firma", "yil")
    v["ls"], v["lk"] = np.log(v.satis), np.log(v.sermaye)
    return v


def test_var_rapor_yorum(tmp_path):
    d = ek.veri.yukle("ihracat")
    d["li"], d["lk"] = np.log(d.ihracat), np.log(d.reel_kur)
    v = ek.var(d[["li", "lk"]].diff().dropna(), ["li", "lk"], 2)
    y = str(v.yorumla())
    assert "kararlıdır" in y and "li denkleminde" in y
    r = v.rapor(etiketler={"li": "ln ihracat"})
    assert len(r.tablolar) == 2 and len(r.sekiller) == 1 and "Sims, 1980" in str(r) and "L.ln ihracat" in str(r)
    assert len(r.tablolar[0].satirlar) == 2 * len(v.katsayilar)  # katsayı + SH satırı
    assert "Lütkepohl" in str(v.rapor(dil="en", etki_tepki=None)) and not v.rapor(etki_tepki=None).sekiller
    v.rapor(tmp_path / "var.docx")
    assert (tmp_path / "var.docx").stat().st_size > 20_000  # şekil gömülü


def test_panel_birim_kok_rapor(f, tmp_path):
    with pytest.warns(UserWarning):
        a = ek.panel_birim_kok(f, "ls", "ips", gecikme=1)
    b = ek.panel_birim_kok(f, "lk", "fisher", gecikme=1)
    r = ek.panel_birim_kok_raporu([a, b], etiketler={"ls": "ln satış"})
    assert len(r.tablolar[0].satirlar) == 2 and "ln satış" in str(r) and "Choi, I. (2001)" in str(r)
    assert "W-t-bar" in str(a.yorumla())
    a.rapor(tmp_path / "pbk.docx", dil="en")
    assert (tmp_path / "pbk.docx").exists()


def test_kesit_ve_cips_rapor(f):
    cd_d = ek.yatay_kesit_bagimliligi(f, "ls")
    cd_m = ek.yatay_kesit_bagimliligi(f, ek.panel("ls ~ lk", f, yontem="sabit"))
    tr, en = str(ek.yatay_kesit_raporu([cd_d, cd_m])), str(ek.yatay_kesit_raporu([cd_d, cd_m], dil="en"))
    assert "model kalıntıları" in tr and "model residuals" in en and "model kalıntıları" not in en
    assert "Baltagi, B. H., Feng" in tr  # FE kalıntısında yanlılığı düzeltilmiş LM
    assert "Baltagi, B. H., Feng" not in str(cd_d.rapor())
    with pytest.warns(UserWarning, match="aralığı dışında"):  # T = 9 < 10
        c = ek.cips(f, "ls", gecikme=0)
    r = ek.cips_raporu([c])
    assert "Pesaran (2007" in str(r) and r.tablolar[0].satirlar[0][3] == "-3,118"
