"""Regresyon sonuçlarının Türkçe yorumu.

Model biçimine göre doğru yorum:
  düz – düz      : x 1 birim artınca y ortalama b birim değişir
  log – düz      : x 1 birim artınca y yaklaşık %100·(e^b − 1) değişir   (yarı esneklik)
  düz – log      : x %1 artınca y yaklaşık b/100 birim değişir
  log – log      : x %1 artınca y yaklaşık %b değişir                     (esneklik)
  kukla          : grup, referans gruba göre ortalama b birim (log ise %) farklı
  kare terim     : etki doğrusal değil; dönüm noktası x* = −b1 / (2·b2)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._yardimci import anlamlilik_ifadesi, l, p_cumle, sayi, yuzde


@dataclass
class Yorum:
    """Türkçe yorum metni; ``print()`` ile okunur, notebook'ta düzgün görünür."""

    baslik: str
    model_satirlari: list[str] = field(default_factory=list)
    katsayi_satirlari: list[str] = field(default_factory=list)
    notlar: list[str] = field(default_factory=list)

    def metin(self) -> str:
        parcalar = [self.baslik, ""]
        if self.model_satirlari:
            parcalar += ["▸ Model geneli"] + [f"  • {s}" for s in self.model_satirlari] + [""]
        if self.katsayi_satirlari:
            parcalar += ["▸ Katsayılar (diğer değişkenler sabitken)"] + [f"  • {s}" for s in self.katsayi_satirlari] + [""]
        if self.notlar:
            parcalar += ["▸ Notlar"] + [f"  • {s}" for s in self.notlar]
        return "\n".join(parcalar).rstrip()

    def __str__(self) -> str:
        return self.metin()

    __repr__ = __str__

    def _repr_html_(self) -> str:
        def liste(baslik, s):
            return f"<p><b>{baslik}</b></p><ul>{''.join(f'<li>{x}</li>' for x in s)}</ul>" if s else ""

        return (
            f"<div><p><b>{self.baslik}</b></p>"
            + liste("Model geneli", self.model_satirlari)
            + liste("Katsayılar (diğer değişkenler sabitken)", self.katsayi_satirlari)
            + liste("Notlar", self.notlar)
            + "</div>"
        )


def _birim(sonuc, ad: str) -> str:
    b = sonuc._birimler.get(ad)
    return f" {b}" if b else ""


def _ad(ad: str) -> str:
    # Türkçe ekler (ünlü uyumu) değişken adına göre değiştiği için adlardan sonra ek getirilmez;
    # cümleler "… değişkeni / değişkenindeki / grubu" gibi sabit eklerle kurulur.
    return f"'{ad}'"


def etkilesim_cumlesi(t, bi: float, Y: str, ad=_ad, akademik: bool = False) -> str | None:
    """İki bileşenli etkileşim teriminin sözel yorumu (istatistikler hariç). Tanınmayan yapı için None."""
    if len(t.parcalar) != 2:
        return None
    sayisal = [p for p in t.parcalar if p.tur in ("surekli", "log")]
    kukla = [p for p in t.parcalar if p.tur == "kukla"]
    if len(sayisal) == 1 and len(kukla) == 1:
        s, k = sayisal[0], kukla[0]
        yon = ("yüksektir" if bi > 0 else "düşüktür") if akademik else ("yüksek" if bi > 0 else "düşük")
        yon_en = "higher" if bi > 0 else "lower"
        return l(
            f"{ad(s.degisken)} değişkeninin {Y} değişkeni üzerindeki etkisi, {ad(k.degisken)} değişkeninin '{k.seviye}' grubunda "
            f"referans gruba ('{k.referans}') kıyasla daha {yon} (eğim farkı = {sayi(bi, 4)})",
            f"The effect of {ad(s.degisken)} on {Y} is {yon_en} for the '{k.seviye}' group of {ad(k.degisken)} "
            f"than for the reference group ('{k.referans}') (slope difference = {sayi(bi, 4)})",
        )
    if len(sayisal) == 2:
        a, b = sayisal
        if akademik:
            yon = "artmaktadır" if bi > 0 else "azalmaktadır"
        else:
            yon = "artıyor" if bi > 0 else "azalıyor"
        yon_en = "increases" if bi > 0 else "decreases"
        return l(
            f"{ad(a.degisken)} değişkeninin {Y} değişkeni üzerindeki etkisi {ad(b.degisken)} değişkeninin düzeyine bağlı: "
            f"{ad(b.degisken)} değişkeni yükseldikçe bu etki {yon} (etkileşim katsayısı = {sayi(bi, 4)})",
            f"The effect of {ad(a.degisken)} on {Y} depends on the level of {ad(b.degisken)}: "
            f"as {ad(b.degisken)} rises, this effect {yon_en} (interaction coefficient = {sayi(bi, 4)})",
        )
    if len(kukla) == 2:
        a, b = kukla
        isaret = "pozitif" if bi > 0 else "negatif"
        isaret_en = "positive" if bi > 0 else "negative"
        ek = "yaratmaktadır" if akademik else "yaratıyor"
        return l(
            f"{ad(a.degisken)} değişkeninin '{a.seviye}' grubu ile {ad(b.degisken)} değişkeninin '{b.seviye}' grubunun birlikte bulunması, "
            f"iki grubun ayrı ayrı etkilerinin toplamına ek olarak {sayi(abs(bi), 4)} büyüklüğünde {isaret} bir fark {ek}",
            f"The joint presence of the '{a.seviye}' group of {ad(a.degisken)} and the '{b.seviye}' group of {ad(b.degisken)} "
            f"adds a {isaret_en} difference of {sayi(abs(bi), 4)} beyond the sum of the two separate effects",
        )
    return None


def yorumla(sonuc) -> Yorum:
    etiketler = getattr(sonuc, "_etiketler", {}) or {}

    def _ad(ad: str) -> str:  # değişken etiketi varsa (ör. .dta'dan) onu kullan
        return etiketler[ad] if ad in etiketler else f"'{ad}'"

    from .regresyon import _log_mu  # döngüsel içe aktarmayı önlemek için

    y_log = _log_mu(sonuc.bagimli)
    y_ad = y_log or sonuc.bagimli
    y_birim = "" if y_log else _birim(sonuc, y_ad)
    yr = Yorum(baslik=f"📊 Yorum — {_ad(y_ad)} {'(logaritması)' if y_log else ''}".rstrip())

    # ---- model geneli
    yr.model_satirlari.append(
        f"Model, {_ad(y_ad)} {'logaritmasındaki' if y_log else 'değişkenindeki'} değişimin yaklaşık {yuzde(sonuc.r2 * 100)} kadarını açıklıyor "
        f"(R² = {sayi(sonuc.r2, 3)}, düzeltilmiş R² = {sayi(sonuc.duzeltilmis_r2, 3)})."
    )
    if not np.isnan(sonuc.f_p_degeri):
        genel_test = getattr(sonuc, "genel_test", None) or f"F = {sayi(sonuc.f_istatistigi, 2)}"
        if sonuc.f_p_degeri < 0.05:
            yr.model_satirlari.append(
                f"Açıklayıcı değişkenler birlikte anlamlı ({genel_test}, {p_cumle(sonuc.f_p_degeri)}): model, yalnızca sabit terimden oluşan modelden daha iyi açıklıyor."
            )
        else:
            yr.model_satirlari.append(
                f"Açıklayıcı değişkenler birlikte anlamlı değil ({genel_test}, {p_cumle(sonuc.f_p_degeri)}): model, {_ad(y_ad)} için güçlü bir açıklama sunmuyor."
            )
    yr.model_satirlari.append(
        f"Gözlem sayısı: {sayi(sonuc.n, 0)} · parametre sayısı: {sonuc.k} · standart hatalar: {sonuc.kovaryans_aciklama}."
    )

    b = sonuc.katsayilar
    p = sonuc.p_degerleri
    kareler = {t.degisken: t for t in sonuc._terimler if t.tur == "kare"}

    for t in sonuc._terimler:
        ad, bi, pi = t.sutun, float(b[t.sutun]), float(p[t.sutun])
        anl = anlamlilik_ifadesi(pi)
        if t.tur == "sabit":
            continue
        if t.tur == "kare":
            continue  # doğrusal terimle birlikte anlatılır
        if t.tur == "etkilesim":
            c = etkilesim_cumlesi(t, bi, _ad(y_ad), ad=_ad)
            if c and pi < 0.10:
                yr.katsayi_satirlari.append(f"{c} — {anl} ({p_cumle(pi)}).")
            elif c:
                yr.katsayi_satirlari.append(
                    f"Etkileşim terimi {_ad(ad)} anlamlı değil ({p_cumle(pi)}): etkinin diğer değişkene göre farklılaştığına dair kanıt yok."
                )
            else:
                yr.katsayi_satirlari.append(f"{_ad(ad)} katsayısı {sayi(bi, 4)} — {anl} ({p_cumle(pi)}).")
            continue
        x = t.degisken
        xb = _birim(sonuc, x)

        if t.tur == "kukla":
            if y_log:
                etki = 100 * (np.exp(bi) - 1)
                cumle = (
                    f"{_ad(x)} değişkeninde '{t.seviye}' grubunun {_ad(y_ad)} değeri, referans grupla ('{t.referans}') karşılaştırıldığında ortalama "
                    f"{yuzde(abs(etki))} {'daha yüksek' if etki > 0 else 'daha düşük'}"
                )
            else:
                cumle = (
                    f"{_ad(x)} değişkeninde '{t.seviye}' grubunun {_ad(y_ad)} değeri, referans grupla ('{t.referans}') karşılaştırıldığında ortalama "
                    f"{sayi(abs(bi), 3)}{y_birim} {'daha yüksek' if bi > 0 else 'daha düşük'}"
                )
            yr.katsayi_satirlari.append(f"{cumle} — {anl} ({p_cumle(pi)}).")
            continue

        if t.tur == "log":
            if y_log:
                cumle = f"{_ad(x)} %1 artınca {_ad(y_ad)} yaklaşık {yuzde(abs(bi), 3)} {'artıyor' if bi > 0 else 'azalıyor'} (esneklik = {sayi(bi, 3)})"
            else:
                cumle = f"{_ad(x)} %1 artınca {_ad(y_ad)} yaklaşık {sayi(abs(bi) / 100, 4)}{y_birim} {'artıyor' if bi > 0 else 'azalıyor'}"
            yr.katsayi_satirlari.append(f"{cumle} — {anl} ({p_cumle(pi)}).")
            continue

        if t.tur == "surekli":
            if x in kareler:
                b2 = float(b[kareler[x].sutun])
                p2 = float(p[kareler[x].sutun])
                donum = -bi / (2 * b2) if b2 != 0 else float("nan")
                sekil = "önce artan sonra azalan (ters U)" if b2 < 0 else "önce azalan sonra artan (U)"
                yr.katsayi_satirlari.append(
                    f"{_ad(x)} değişkeninin etkisi doğrusal değil: {sekil} bir ilişki var. Etki, {_ad(x)} ≈ {sayi(donum, 2)}{xb} civarında yön değiştiriyor "
                    f"(doğrusal terim {p_cumle(pi)}; kare terim {p_cumle(p2)}, {anlamlilik_ifadesi(p2)})."
                )
                xmean = float(sonuc.X[x].mean())
                marj = bi + 2 * b2 * xmean
                if y_log:
                    yr.katsayi_satirlari.append(
                        f"Ortalama düzeyde ({_ad(x)} = {sayi(xmean, 2)}{xb}), {_ad(x)} 1{xb} artınca {_ad(y_ad)} yaklaşık {yuzde(abs(100 * (np.exp(marj) - 1)))} {'artıyor' if marj > 0 else 'azalıyor'}."
                    )
                else:
                    yr.katsayi_satirlari.append(
                        f"Ortalama düzeyde ({_ad(x)} = {sayi(xmean, 2)}{xb}), {_ad(x)} 1{xb} artınca {_ad(y_ad)} yaklaşık {sayi(abs(marj), 3)}{y_birim} {'artıyor' if marj > 0 else 'azalıyor'}."
                    )
                continue
            if y_log:
                etki = 100 * (np.exp(bi) - 1)
                cumle = f"{_ad(x)} 1{xb} artınca {_ad(y_ad)} ortalama {yuzde(abs(etki), 2)} {'artıyor' if etki > 0 else 'azalıyor'}"
            else:
                cumle = f"{_ad(x)} 1{xb} artınca {_ad(y_ad)} ortalama {sayi(abs(bi), 4)}{y_birim} {'artıyor' if bi > 0 else 'azalıyor'}"
            yr.katsayi_satirlari.append(f"{cumle} — {anl} ({p_cumle(pi)}).")
            continue

        yr.katsayi_satirlari.append(f"{_ad(ad)} katsayısı {sayi(bi, 4)} — {anl} ({p_cumle(pi)}).")

    # ---- notlar
    if y_log:
        yr.notlar.append("Bağımlı değişken logaritmik: katsayılar yüzde değişim olarak yorumlandı (%100·(eᵇ−1)).")
    if any(t.tur == "etkilesim" for t in sonuc._terimler):
        yr.notlar.append(
            "Modelde etkileşim terimi var: etkileşime giren değişkenlerin kendi katsayıları (ana etkiler), diğer değişken sıfırken ya da referans grupta geçerli olan etkiyi gösterir."
        )
    anlamsiz = [t.sutun for t in sonuc._terimler if t.tur not in ("sabit",) and p[t.sutun] >= 0.10]
    if anlamsiz:
        yr.notlar.append(
            "Anlamlı olmayan katsayılar sıfırdan ayırt edilemiyor; bu, 'etkisi yok' demek değil, 'veri bu etkiyi göstermeye yetmiyor' demektir: "
            + ", ".join(anlamsiz)
            + "."
        )
    if sonuc.kovaryans_turu == "klasik":
        yr.notlar.append("Standart hatalar klasik varsayımlara dayanıyor; varsayımları kontrol etmek için: model.varsayimlar()")
    yr.notlar.append(
        "Bu yorumlar ilişkiyi anlatır; nedensellik için araştırma tasarımının (deney, araç değişken vb.) buna izin vermesi gerekir."
    )
    return yr
