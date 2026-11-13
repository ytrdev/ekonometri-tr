"""Türkçe ekonometri kütüphanesi. Tahmin, Türkçe yorum, varsayım testi ve Word raporu.

>>> import ekonometri as ek
>>> veri = ek.veri.yukle("ucret")
>>> model = ek.ekk("np.log(ucret) ~ egitim + tecrube + I(tecrube**2) + cinsiyet", veri)
>>> model                 # Türkçe özet tablo
>>> model.yorumla()       # katsayıların Türkçe yorumu
>>> model.varsayimlar()   # varsayım testleri + öneriler
>>> model.rapor("bulgular.docx")   # tez/ödev için Yöntem + Bulgular metni ve tablolar (Word)
"""

from . import veri
from .arac import IVSonuc, iv2sls
from .ardl import ARDLSinirSonucu, ardl_sinir
from .dinamik import DinamikSonuc, xtabond
from .girdi import etiketle, fark, gecikme, kaydet, oku, tsset, xtset
from .grafik import (
    Sekil,
    histogram,
    kalinti_grafigi,
    katsayi_grafigi,
    marjinal_grafik,
    normallik_grafigi,
    roc_grafigi,
    serpme,
    zaman_grafigi,
)
from .ikili import IkiliSonuc, logit, probit
from .johansen import JohansenSonucu, johansen
from .karsilastirma import Karsilastirma, KorelasyonSonuc, karsilastir, korelasyon, tanimlayici
from .klasik import anova, ki_kare, prais, ttest
from .marjinal import MarjSonucu, marjlar
from .panel import PanelAnalizi, PanelSonuc, hausman, panel, panel_analizi, xttest0
from .panel_kok import PanelBirimKokSonucu, panel_birim_kok, panel_duraganlik
from .rapor import Rapor
from .regresyon import EKKSonuc, EkonometriHatasi, WaldSonuc, ekk
from .secim import MLSonuc, mlogit, nbreg, ologit, oprobit, poisson, tobit
from .stata import Stata, stata
from .testler import (
    TestSonucu,
    VarsayimRaporu,
    breusch_godfrey,
    breusch_pagan,
    durbin_watson,
    jarque_bera,
    reset,
    vif,
    white,
)
from .vektor import VARSonucu, var, var_gecikme_secimi
from .yorum import Yorum
from .zaman import (
    BirimKokSonucu,
    DuraganlikSonucu,
    EngleGrangerSonucu,
    GrangerSonucu,
    GrangerYon,
    HataDuzeltmeSonucu,
    VARGrangerSonucu,
    VARGrangerYon,
    adf,
    duraganlik,
    engle_granger,
    granger,
    hata_duzeltme,
    kpss,
    var_granger,
)

__version__ = "0.8.0"

__all__ = [
    "ekk",
    "EKKSonuc",
    "WaldSonuc",
    "EkonometriHatasi",
    "Yorum",
    "Rapor",
    "logit",
    "probit",
    "IkiliSonuc",
    "karsilastir",
    "Karsilastirma",
    "korelasyon",
    "KorelasyonSonuc",
    "tanimlayici",
    "TestSonucu",
    "VarsayimRaporu",
    "breusch_pagan",
    "white",
    "durbin_watson",
    "breusch_godfrey",
    "jarque_bera",
    "vif",
    "reset",
    "adf",
    "kpss",
    "duraganlik",
    "engle_granger",
    "hata_duzeltme",
    "granger",
    "var_granger",
    "johansen",
    "xtabond",
    "ardl_sinir",
    "BirimKokSonucu",
    "DuraganlikSonucu",
    "EngleGrangerSonucu",
    "HataDuzeltmeSonucu",
    "GrangerSonucu",
    "GrangerYon",
    "VARGrangerSonucu",
    "VARGrangerYon",
    "ARDLSinirSonucu",
    "JohansenSonucu",
    "DinamikSonuc",
    "panel",
    "panel_analizi",
    "hausman",
    "xttest0",
    "PanelSonuc",
    "PanelAnalizi",
    "iv2sls",
    "IVSonuc",
    "oku",
    "kaydet",
    "etiketle",
    "xtset",
    "tsset",
    "gecikme",
    "fark",
    "Stata",
    "stata",
    "ttest",
    "ki_kare",
    "anova",
    "prais",
    "marjlar",
    "ologit",
    "oprobit",
    "mlogit",
    "poisson",
    "nbreg",
    "tobit",
    "MLSonuc",
    "var",
    "panel_birim_kok",
    "panel_duraganlik",
    "PanelBirimKokSonucu",
    "var_gecikme_secimi",
    "VARSonucu",
    "MarjSonucu",
    "Sekil",
    "kalinti_grafigi",
    "normallik_grafigi",
    "katsayi_grafigi",
    "marjinal_grafik",
    "zaman_grafigi",
    "histogram",
    "serpme",
    "roc_grafigi",
    "veri",
    "__version__",
]
