# Değişiklik günlüğü

Biçim [Keep a Changelog](https://keepachangelog.com/tr-TR/1.1.0/) önerisine, sürüm numaraları [Anlamsal Sürümleme](https://semver.org/lang/tr/) kurallarına uyar.

## [0.8.0] — 2026-11-13

### Eklendi
- **Grafikler:** `ek.kalinti_grafigi` (rvfplot), `normallik_grafigi` (qnorm), `katsayi_grafigi` (coefplot), `marjinal_grafik` (marginsplot), `zaman_grafigi` (tsline), `histogram`, `serpme` (scatter + lfit), `roc_grafigi` (lroc). Türkçe etiketler, ondalık virgül, 300 dpi; `Rapor.sekil_ekle()` ile Word / HTML / LaTeX raporuna numaralı şekil olarak girer
- **Tablo dışa aktarma:** `Karsilastirma.kaydet()` ve `Rapor.kaydet()` için .tex (booktabs) ve .xlsx; Stata `esttab m1 m2 using tablo.docx|.tex|.xlsx`
- **Marjlar:** `ek.marjlar()` — `margins, at()`, grup marjları, koşullu marjinal etki (`dydx() at()`), delta yöntemi
- **Logit/probit sonrası:** `roc()` ve AUC, `hosmer_lemeshow()` (estat gof), sınıflandırmada pozitif / negatif öngörü değeri
- **Seçim, sayım ve sansürlü modeller:** `ek.ologit`, `ek.oprobit`, `ek.mlogit`, `ek.poisson`, `ek.nbreg`, `ek.tobit`; odds / göreli risk / olay oranları, sonuç başına marjinal etkiler, Türkçe yorum ve tez raporu
- **VAR:** `ek.var`, `ek.var_gecikme_secimi` (varsoc), `kararlilik()` (varstable), `etki_tepki()` ve grafiği (irf)
- **Panel birim kök:** `ek.panel_birim_kok` — LLC, IPS (W-t-bar), Fisher-ADF (P, Z, L*, Pm); `ek.panel_duraganlik` düzey + fark tablosu
- **Stata komutları:** `ologit`, `oprobit`, `mlogit`, `poisson`, `nbreg`, `tobit`, `var`, `varsoc`, `varstable`, `vargranger`, `irf graph|table`, `ardl` + `estat ectest`, `vecrank`, `xtabond` + `estat abond|sargan`, `xtunitroot`, `margins … at()`, `estat classification|gof`, `lroc`, grafik komutları ve `graph export`, `esttab using`
- **Belge sitesi:** MkDocs ile (README, rehber, fonksiyon başvurusu, ders defterleri, doğrulama); GitHub Pages iş akışı
- 07 numaralı ders defteri: tez araç kutusu
- `predict` ologit / oprobit / mlogit (sonuç başına olasılık, `outcome()`, `xb`), poisson / nbreg (`n`), tobit (`xb`); `MLSonuc.tahminler()`
- README ve sitede grafik galerisi
- Bağımsız doğrulamaya 6 yeni grup (R MASS, nnet, AER, pROC, plm purtest): toplam 22 grup

### Düzeltildi
- `xtabond`: Sargan istatistiği yarı büyüklükte hesaplanıyordu (σ̂² paydası 2(n − k) olmalı); varsayılan standart hatalar artık Stata gibi klasik, `kovaryans="saglam"` ile sandviç. AR(1)/AR(2) testleri eklendi
- `qnorm` çizim konumları Stata gibi i / (N + 1)
- `nbreg`: aşırı yayılım yokken (α → 0) tahmin NaN veriyordu; artık α ≈ 0 ve LR p = 1 (Stata gibi), katsayılar Poisson ile aynı
- `poisson` / `nbreg` `exposure()` ile marjinal etkiler ve tahminler, ortalama yerine gözlem başına maruziyetle hesaplanıyor (Stata `margins` / `predict n`)
- Grafik eksenleri: çok küçük değerler `1,4×10⁻⁵` biçiminde, büyük sayılarda etiketler üst üste binmiyor
- Stata arayüzü: `if donem >= tq(2010q1)` / `2010q1` tarih koşulları (çeyreklik/aylık dönem sütunları) çalışıyor
- Stata arayüzü: `yorum`, `rapor`, `predict` desteklemeyen modellerden sonra Python hatası yerine Türkçe mesaj veriyor
- `marginsplot` yalnız faktör marjlarıyla (`margins bolge`) da çiziliyor

### Değişti
- statsmodels artık zorunlu bağımlılık (sıralı / çok terimli / sayım modellerinin optimizasyonu)

## [0.7.0] — 2026-11-12

### Eklendi
- **ARDL sınır testi:** `ek.ardl_sinir()`. Pesaran, Shin ve Smith (2001) kısıtsız hata düzeltme modeli, F sınırı ve uzun dönem çarpanları. Kritik değerler açıklayıcı sayısına (k) göre okunur
- **VAR Granger:** `ek.var_granger()`. Üç ve daha fazla seride, diğer değişkenler kontrol edildikten sonra tek denklem blok dışsallık F testi. Gecikme Lütkepohl AIC ile seçilebilir
- **Rapor dili ve tablo biçimi:** `rapor(dil="en")` İngilizce metin üretir: EKK, ARDL, VAR Granger, panel, logit/probit, 2SLS, ADF/KPSS durağanlık tablosu, Engle-Granger, hata düzeltme, iki değişkenli Granger, model karşılaştırma ve korelasyon. `bicim="apa"` üç çizgili tez tablosu, `bicim="izgara"` tüm kenarlıkları çizer
- **Johansen eşbütünleşme:** `ek.johansen()`. İz ve en büyük özdeğer istatistikleri, MacKinnon–Haug–Michelis (1999) kritik değerleri
- **Dinamik panel:** `ek.xtabond()`. Arellano-Bond tek adımlı fark GMM, Sargan aşırı tanımlama testi ve sağlam standart hatalar

## [0.6.0] — 2026-11-12

### Eklendi
- **Hata düzeltme modeli:** `ek.hata_duzeltme()` ve `EngleGrangerSonucu.hata_duzeltme()`. Uzun dönem EKK kalıntısının gecikmesi, kısa dönem farkları ve isteğe bağlı eşzamanlı fark; gecikme AIC/BIC ile seçilebilir. Eşbütünleşme yoksa hız yorumu yazılmaz
- **Granger nedensellik:** `ek.granger(y, x)`. İstenen yön ve ters yön; SSR F ve χ² testleri. Gecikme, ortak örneklemde AIC/BIC ile seçilebilir
- Her iki sonuç için Türkçe karar ve tez raporu (`rapor("dosya.docx")`)

## [0.5.0] — 2026-11-11

### Eklendi
- **Ağırlıklar:** `ek.ekk(..., agirlik=, agirlik_turu="aw"|"fw"|"pw")`, logit/probit (fw, pw), summarize; Stata `[aw=]` `[fw=]` `[pw=]` sözdizimi
- **Veri düzenleme (Stata):** merge, append, reshape long/wide, collapse, encode, decode, destring, label define/values, recode, duplicates, order
- **Klasik testler:** `ek.ttest`, `ek.ki_kare`, `ek.anova`, `ek.xttest0`; Stata `ttest`, `oneway`, `tabulate …, chi2`, `xttest0`
- **Zaman serisi tahmincileri:** `ek.prais` (Prais-Winsten / Cochrane-Orcutt), `kovaryans="newey"` (Stata newey); Stata `prais`, `newey`
- `yorumla()` değişken etiketlerini kullanır

### Düzeltildi
- Tekrarlanan satır indeksli veride (ör. birleştirme sonrası) gözlemler çoğalıyordu
- Kümelenmiş standart hatalı modelde raporun Yöntem cümlesi yanlış tahminciyi yazıyordu
- RESET testinde sayısal kararlılık (ŷ standartlaştırıldı)
- Metin değişkenlerinde `missing()` boş dizgiyi eksik saymıyordu

## [0.4.0] — 2026-11-11

### Eklendi
- **Kendi verin:** `ek.oku()` — Stata .dta (değişken ve değer etiketleri), Excel, CSV (ayırıcı, ondalık virgül, Windows-1254 otomatik), SPSS; `ek.kaydet()`, `ek.etiketle()`
- `ek.xtset()`, `ek.tsset()`, `ek.gecikme()`, `ek.fark()` (Stata L. / D. mantığı, zaman boşluklarında eksik)
- Değişken etiketleri tüm raporlarda değişken adının yerine kullanılır
- **Stata komutları:** `ek.Stata()` ve `.do()` — regress, logit, probit, xtreg fe/re, ivregress 2sls, dfuller, margins, estat …, hausman, test, predict, esttab, gen/replace/egen, summarize, tabulate, pwcorr ve diğerleri; faktör (`i.`, `ib#.`, `##`) ve zaman serisi (`L.`, `D.`, `F.`) değişken dili; Stata ifade kuralları (eksik değer +∞)
- Kümelenmiş standart hata: `ek.ekk(..., kume=)`, `ek.logit/probit(..., kume=)`, `ek.iv2sls(..., kume=)`
- `ek.breusch_pagan(degiskenler=, koenker=)`: Stata `estat hettest` varsayılanı ve `rhs` / `iid` seçenekleri
- 2SLS için Durbin (estat endogenous) ve Basmann (estat overid) testleri
- `ek.tanimlayici(detay=True)` (summarize, detail), `ek.korelasyon(eksik="cift")` (pwcorr)
- Tek panel modeli için `rapor()`; REHBER.md (Stata komut karşılıkları); örnek do dosyası; ders defteri 06

### Değişti (Stata ile aynı sonuç için)
- `ek.iv2sls()` varsayılanı Stata `ivregress 2sls` gibi: σ² = e'e/N, z ve Wald χ²; eski davranış `kucuk_orneklem=True` (Stata `small`)
- Rassal etkiler: z istatistiği ve Wald χ²; kümelenmiş standart hatada G − 1 serbestlik; R² ölçüleri Stata tanımıyla
- Logit/probit dayanıklı standart hata çarpanı N/(N−1); `margins` gibi ayrık fark yalnızca kategorik değişkenlerde
- `ek.reset()` varsayılanı ŷ², ŷ³, ŷ⁴ (estat ovtest); `ek.tanimlayici()` çarpıklık/basıklık Stata formülüyle
- Sonuç tablolarında okunur terim adları (`cinsiyet: Kadın`, `tecrube²`, `ln(ucret)`)

## [0.3.0] — 2026-11-11

### Eklendi
- `ek.karsilastir(m1, m2, ...)`: modelleri yan yana gösteren tez tablosu (katsayı + yıldız, parantezde standart hata, uyum istatistikleri) ve raporu
- `ek.korelasyon()` (Pearson / Spearman, anlamlılık yıldızlı alt üçgen tablo) ve `ek.tanimlayici()`
- Etkileşim terimlerinin (`x1:x2`, `x*grup`) tanınması ve Türkçe yorumu
- `ek.logit()`, `ek.probit()`: ortalama marjinal etkiler (delta yöntemi), odds oranları, sınıflandırma tablosu, yorum ve rapor
- `ek.adf()`, `ek.kpss()`, `ek.duraganlik()` (düzey + birinci fark tez tablosu, I(0)/I(1) kararı), `ek.engle_granger()`; MacKinnon tabloları pakete gömüldü
- `ek.panel()`, `ek.panel_analizi()`, `ek.hausman()`: havuzlanmış EKK, sabit etkiler (tek/iki yönlü), rassal etkiler (Swamy-Arora), F / Breusch-Pagan LM / Hausman ile model seçimi, kümelenmiş standart hata, tez bölümü raporu
- `ek.iv2sls()`: 2SLS, birinci aşama F ve kısmi R², Wu-Hausman ve Sargan testleri, EKK–2SLS karşılaştırmalı rapor
- Simüle veri setleri: `firma` (panel), `getiri` (araç değişken)
- Ders defterleri: 02 zaman serisi, 03 logit/probit, 04 panel veri, 05 araç değişken
- linearmodels ile karşılaştırmalı doğrulama testleri

## [0.2.0] — 2026-11-11

### Eklendi
- `model.rapor()`: tez / makale / ödev için Türkçe bulgu raporu — Yöntem bölümü (model denklemi, kaynak gösterimleri), tanımlayıcı istatistik ve frekans tabloları, tez formatında regresyon tablosu (β, SH, t, p, %95 GA), katsayıların akademik Türkçe yorumu, varsayım testleri tablosu ve metni, kaynakça
- Çıktı biçimleri: Word (`.docx`, APA tarzı tablolar), Markdown, HTML; Jupyter'de doğrudan görüntüleme
- Seçenekler: `etiketler` (değişken adları), `tablo_no`, `yontem`, `kaynakca`
- Örnek çıktı: `ornekler/bulgular.docx`

### Değişti
- `python-docx` bağımlılıklara eklendi

## [0.1.0] — 2026-11-11

### Eklendi
- `ek.ekk()`: formül arayüzlü EKK (OLS) regresyonu; Türkçe özet tablo (metin + Jupyter HTML)
- Standart hata türleri: klasik, `HC0`–`HC3`, `HAC` (Newey-West)
- `model.yorumla()`: düz/log/esneklik, kukla (referans grup) ve kare terim (dönüm noktası) için Türkçe yorum
- `model.varsayimlar()`: Breusch-Pagan, White, Durbin-Watson, Breusch-Godfrey, Jarque-Bera, VIF, Ramsey RESET — Türkçe karar ve öneri
- `model.wald_testi()`, `model.guven_araliklari()`, `model.tahmin()`
- Simüle veri setleri: `ucret`, `konut`, `sinav`, `ihracat`
- Ders defteri: 01 · Çoklu regresyon
- statsmodels ile birebir karşılaştırma yapan test paketi; GitHub Actions (Python 3.10–3.13)
