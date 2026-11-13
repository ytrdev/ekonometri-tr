# Stata kullanıcıları için rehber

`ekonometri-tr`'yi iki yoldan kullanabilirsin:

1. **Stata komutlarıyla.** Stata'da yazdığın komutları aynen yaz, do dosyalarını çalıştır.
2. **Python fonksiyonlarıyla.** `ek.ekk()`, `ek.panel()`, `ek.iv2sls()` gibi fonksiyonları doğrudan çağır.

İki yol da aynı hesaplama motorunu kullanır. Formüller Stata'nın belgelenmiş varsayılanlarına göre yazıldı:
- Standart hata çarpanları
- Serbestlik dereceleri
- z ya da t dağılımı seçimi
- R² tanımları

## 1. Kendi verini yükle

```python
import ekonometri as ek

veri = ek.oku("anket.dta")          # Stata: use anket.dta
veri = ek.oku("veri.xlsx", sayfa="2023")   # import excel veri.xlsx, sheet("2023") firstrow
veri = ek.oku("tuik.csv")           # import delimited tuik.csv
```

| Biçim | Ne korunur / ne yapılır |
|---|---|
| `.dta` | **Değişken etiketleri** raporlarda ad yerine kullanılır. **Değer etiketli** değişkenler kategoriye çevrilir; temel düzey Stata'daki gibi en küçük koddur. Kodlar saklanır, böylece `if cinsiyet == 2` çalışır. |
| `.xlsx` / `.xls` | İlk satır değişken adıdır (`firstrow`). Boşluklu ya da özel karakterli başlıklar formülde kullanılabilir hâle getirilir (`"Aylık Ücret"` → `aylik_ucret`); özgün başlık etiket olarak saklanır. |
| `.csv` / `.txt` | Ayırıcı (`,` `;` sekme), **ondalık virgül** (`1.234,56`) ve kodlama (UTF-8 / Windows-1254) otomatik bulunur. |
| `.sav` (SPSS) | `pip install pyreadstat` gerekir. |

Kaydetmek için `ek.kaydet(veri, "temiz.dta")` kullanılır. Değişken etiketleri dosyaya yazılır.

## 2. Stata komutlarıyla çalış

```python
s = ek.Stata()                      # ya da ek.Stata(veri)
s("use anket.dta")
s("gen lnw = ln(ucret)")
s("regress lnw egitim c.tecrube##c.tecrube i.cinsiyet, vce(robust)")
s("estat hettest")
s("rapor bulgular.docx")            # Türkçe tez raporu (Stata'da olmayan ek)

s.do("analiz.do")                   # hazır do dosyasını çalıştır
```

Do dosyasında yorumlar (`*`, `//`, `/* */`), satır devamı (`///`), `quietly` ve `capture` desteklenir. Örnek: [ornekler/ornek_analiz.do](ornekler/ornek_analiz.do).

### Desteklenen komutlar

| Grup | Komutlar |
|---|---|
| Veri | `use`, `sysuse` (hazır veri setleri), `import excel`, `import delimited`, `save`, `describe`, `list`, `count` |
| Değişken | `generate`, `replace`, `egen` (mean, total, sum, sd, min, max, count, median; `by()` / `bysort g:`), `drop`, `keep`, `rename`, `order`, `label variable` / `define` / `values`, `encode`, `decode`, `destring` (`dpcomma`, `ignore()`, `force`), `recode`, `duplicates report` / `drop`, `sort`, `tsset`, `xtset` |
| Birleştirme | `merge 1:1` / `m:1` / `1:m` (`keep()`, `nogenerate`, `keepusing()`, `_merge`), `append using`, `reshape long` / `wide`, `collapse` (mean, sum, count, sd, min, max, median, first, last; `by()`) |
| Tanımlayıcı | `summarize [, detail]`, `tabulate` (tek / iki yönlü), `correlate`, `pwcorr` |
| Tahmin | `regress`, `newey`, `prais` (`corc`), `logit`, `probit`, `ologit`, `oprobit`, `mlogit` (`baseoutcome()`, `rrr`), `poisson` / `nbreg` (`exposure()`, `irr`), `tobit` (`ll()`, `ul()`), `xtreg, fe` / `xtreg, re`, `xtabond` (`lags()`, `vce(robust)`), `ivregress 2sls`, `dfuller` |
| Zaman serisi | `var` (`lags(1/#)`), `varsoc` (`maxlag()`), `varstable`, `vargranger`, `irf graph oirf` / `irf table oirf` (`step()`; `irf create` gerekmez), `ardl` (`lags()`, `maxlags()`, `aic`) + `estat ectest`, `vecrank` (`lags()`, `trend()`) |
| Panel birim kök | `xtunitroot llc` / `ips` / `fisher …, dfuller` (`lags(#)` ya da `lags(aic #)`, `trend`, `noconstant`, `demean`) |
| Grafik | `rvfplot`, `qnorm [değişken]`, `histogram x [, normal bin()]`, `scatter y x`, `twoway (scatter y x) (lfit y x)`, `tsline`, `coefplot [m1 m2]`, `marginsplot`, `lroc`, `irf graph`, `graph export dosya.png [, replace]` (.png, .pdf, .svg) |
| Testler | `ttest` (tek örneklem, `by()`, `unequal`, eşleştirilmiş), `oneway`, `tabulate x y, chi2` (`V`, `lrchi2`), `xttest0` |
| Tahmin sonrası | `predict` (xb, residuals, olasılık), `margins [faktör] [, at(x=(8 12 16)) dydx()]` (numlist: `0(5)20`, `1/4`), `estat hettest / imtest, white / ovtest / vif / dwatson / bgodfrey / firststage / endogenous / overid / classification / gof, group(#) / abond / sargan / ectest`, `test`, `hausman`, `estimates store`, `estimates table` / `esttab [using dosya.docx \| .tex \| .xlsx \| .csv] [, replace]` / `etable`, `display` (`_b[x]`, `_se[x]`) |
| Türkçe ekler | `yorum`, `varsayimlar`, `rapor "dosya.docx"` |

**Değişken dili:**
- Faktör değişkenleri: `i.x`, `ib2.x`, `ib(first).x`, `ib(last).x`, `ib(freq).x`, `c.x`, `x#z`, `x##z`. `c.x#c.x` kare terim olarak tanınır ve yorumda dönüm noktası hesaplanır.
- Zaman serisi işlemcileri: `L.`, `L2.`, `D.`, `F.`, zincirli `L.D.y`, `L(1/3).x`.
- Değişken listeleri: `x*`, `x1-x5`, kısaltılmış adlar.
- Kısıtlar: `if`, `in`.

**Seçenekler:**
- `vce(robust | cluster değişken | hc2 | hc3)`, `robust`, `noconstant`
- Ağırlıklar: `[aw=x]`, `[fw=x]`, `[pw=x]` (regress, logit, probit; summarize için aw / fw)
- `small`, `first` (ivregress)
- `fe`, `re` (xtreg)
- `lags(#)`, `trend`, `detail`, `sigmamore`

**İfadeler:**
- Stata kuralları uygulanır: **eksik değer +∞ sayılır** (`if x > 100` eksikleri de seçer), sıfıra bölme eksik değer verir.
- İşlevler: `ln`, `log`, `exp`, `sqrt`, `abs`, `round`, `floor`, `ceil`, `int`, `missing`, `inlist`, `inrange`, `cond`, `max`, `min`, `normal`, `invnormal`, `_n`, `_N`, `x[_n-1]`.

`gen lnw = ln(ucret)` ve `gen x2 = x^2` ile üretilen değişkenler hatırlanır. Böylece yorum ve raporda katsayılar yüzde değişim ve dönüm noktası olarak doğru anlatılır.

Listede olmayan komutlar (ör. `xtgls`, `gmm`, `sem`) ve `merge m:m` desteklenmez; yazıldığında Türkçe hata mesajı verilir. Panel modellerinde (`xtreg`) ağırlık desteklenmez.

## 3. Komut karşılıkları ve Stata uyumu

| Stata | Python fonksiyonu | Stata ile aynı tutulan kurallar |
|---|---|---|
| `regress y x` | `ek.ekk("y ~ x", veri)` | — |
| `, vce(robust)` | `kovaryans="HC1"` | çarpan n/(n−k) |
| `, vce(hc2)` / `vce(hc3)` | `kovaryans="HC2"` / `"HC3"` | — |
| `, vce(cluster g)` | `kume="g"` | çarpan G/(G−1)·(n−1)/(n−k); t ve F için **G − 1** serbestlik derecesi |
| `estat hettest` | `ek.breusch_pagan(m, "tahmin", koenker=False)` | tahmin edilen değerler, normal hata, χ²(1) |
| `estat hettest, rhs iid` | `ek.breusch_pagan(m)` | Koenker n·R² (kütüphanenin varsayılanı) |
| `estat imtest, white` | `ek.white(m)` | — |
| `estat ovtest` | `ek.reset(m)` | ŷ², ŷ³, ŷ⁴ → F(3, n−k−3) |
| `estat vif` · `estat dwatson` · `estat bgodfrey, lags(p)` | `ek.vif` · `ek.durbin_watson` · `ek.breusch_godfrey(m, p)` | bgodfrey'de başlangıçtaki eksik gecikmeler 0 alınır |
| `logit` / `probit` | `ek.logit` / `ek.probit` | OIM standart hata |
| `, vce(robust)` (logit/probit) | `kovaryans="saglam"` | çarpan N/(N−1); başlıkta Wald χ² |
| `, vce(cluster g)` (logit/probit) | `kume="g"` | çarpan G/(G−1) |
| `margins, dydx(*)` | `m.marjinal_etkiler()` | delta yöntemi; ayrık fark yalnızca `i.` (kategorik) değişkenlerde |
| `xtset i t` | `ek.xtset(veri, "i", "t")` | — |
| `xtreg y x, fe` | `ek.panel(..., yontem="sabit")` | sabit terim; sd = n − N − K (K: sabit hariç açıklayıcı sayısı); R² içsel/arası/genel korelasyon kareleri |
| `xtreg y x, re` | `ek.panel(..., yontem="rassal")` | Swamy-Arora, **z** ve Wald χ² |
| `, vce(robust)` / `vce(cluster i)` (xtreg) | `kovaryans="kume"` | fe: G/(G−1)·(n−1)/(n−k) ve G − 1 sd; re: G/(G−1) |
| `hausman fe re` | `ek.hausman(fe, re, sigmamore=False)` | genelleştirilmiş ters; `sigmamore` seçeneği |
| `ivregress 2sls y x (z = w)` | `ek.iv2sls("y ~ x + [z ~ w]", veri)` | σ² = e'e/N, **z**, Wald χ² |
| `, small` (ivregress) | `kucuk_orneklem=True` | t, F, n − k |
| `estat firststage` | `m.birinci_asama` | — |
| `estat endogenous` | `m.tanilar()` | Durbin χ² ve Wu-Hausman F |
| `estat overid` | `m.tanilar()` | Sargan ve Basmann |
| `dfuller x, lags(p) trend` | `ek.adf(x, "trend", gecikme=p)` | Stata varsayılanı lags(0) |
| `summarize, detail` | `ek.tanimlayici(veri, detay=True)` | Stata yüzdelik tanımı; çarpıklık m₃/m₂^1,5, basıklık m₄/m₂² |
| `regress y x [aw=w]` / `[fw=w]` / `[pw=w]` | `ek.ekk(..., agirlik="w", agirlik_turu="aw")` | aw gözlem sayısına normalleştirilir; fw gözlem çoğaltmayla aynı (N = Σw); pw otomatik dayanıklı SH |
| `newey y x, lag(L)` | `kovaryans="newey", gecikme=L` | Bartlett çekirdeği, çarpan n/(n−k) |
| `prais y x` / `, corc` | `ek.prais(...)` / `corc=True` | yinelemeli; ρ: kalıntıların gecikmesine sabitsiz regresyon |
| `ttest x, by(g)` / `unequal` | `ek.ttest(veri, "x", by="g")` / `esit_varyans=False` | Satterthwaite serbestlik derecesi |
| `tabulate x y, chi2 V` | `ek.ki_kare(veri, "x", "y")` | Pearson χ², olabilirlik oranı, Cramér V (2×2'de işaretli) |
| `oneway y g` | `ek.anova(veri, "y", "g")` | Bartlett eşit varyans testi |
| `correlate` / `pwcorr, sig` | `ek.korelasyon(veri)` / `eksik="cift"` | liste / çift bazında eksik veri |
| `estimates table m1 m2` | `ek.karsilastir(m1, m2)` | — |
| `esttab m1 m2 using t.docx` / `.tex` / `.xlsx` | `ek.karsilastir(m1, m2).kaydet("t.tex")` | .tex: booktabs tablo parçası (`\input{}` ile eklenir) |
| `margins, at(x=(8 12 16))` / `margins g` | `ek.marjlar(m, at={"x": [8, 12, 16]}, grup="g")` | diğer değişkenler gözlenen değerinde (ortalama tahmin); EKK sonrası t, logit/probit sonrası z |
| `margins, dydx(x) at(z=(5 20))` | `ek.marjlar(m, dydx="x", at={"z": [5, 20]})` | delta yöntemi; kategorik `dydx` referans düzeye fark |
| `estat classification` / `lroc` / `estat gof, group(10)` | `m.siniflandirma()` / `m.roc()` / `m.hosmer_lemeshow(10)` | eşik 0,5; AUC Mann-Whitney formülü; HL grupları Stata `xtile` gibi, sd = g − 2 |
| `ologit` / `oprobit` | `ek.ologit` / `ek.oprobit` | P(y ≤ j) = F(kesim_j − xb), kesim noktaları `/cut1 …`; OIM SH |
| `mlogit y x` | `ek.mlogit("y ~ x", veri)` | temel kategori: en sık sonuç (`baseoutcome()` = `temel=`) |
| `poisson` / `nbreg` (`, exposure(e)`) | `ek.poisson` / `ek.nbreg` (`maruziyet="e"`) | nbreg: NB2, `/lnalpha` ve α; α = 0 LR testi χ̄²(01) (p yarıya bölünür) |
| `tobit y x, ll(0)` | `ek.tobit("y ~ x", veri, alt=0)` | `/sigma` ve var(e) raporlanır |
| `var y1 y2, lags(1/2)` | `ek.var(veri, ["y1", "y2"], gecikme=2)` | denklem başına EKK; Σ̂ = E'E/T (ML); z; Lütkepohl AIC/HQIC/SBIC |
| `varsoc, maxlag(4)` | `ek.var_gecikme_secimi(..., en_fazla_gecikme=4)` | tüm gecikmeler ortak örneklemde (T = n − p_max) |
| `irf graph oirf` | `v.etki_tepki_grafigi(10)` | Cholesky; değişken sırası önemlidir |
| `xtabond y x` / `, vce(robust)` | `ek.xtabond("y ~ x", veri)` / `kovaryans="saglam"` | tek adımlı; σ̂² = Δê'Δê / (2(n − k)); Sargan homoskedastik; AR(1)/AR(2) sağlam |
| `xtunitroot llc y` | `ek.panel_birim_kok(veri, "y", "llc")` | Bartlett bant round(3,21·T^{1/3}); μ*, σ* LLC (2002) Tablo 2'den T̃'ye göre ara değer |
| `xtunitroot ips y, lags(1)` | `ek.panel_birim_kok(veri, "y", "ips", gecikme=1)` | W-t-bar, IPS (2003) Tablo 3 momentleri |
| `xtunitroot fisher y, dfuller lags(1)` | `ek.panel_birim_kok(veri, "y", "fisher", gecikme=1)` | birim p-değerleri MacKinnon (1994) yaklaşık p (Stata `dfuller` ile aynı); P, Z, L*, Pm |

## 4. Doğrulama ve bilinen farklar

**Nasıl doğrulandı?**
- Stata'nın kendisi test ortamında çalıştırılamadığı için sonuçlar Stata çıktısıyla doğrudan karşılaştırılmadı.
- Bunun yerine her formül Stata'nın belgelenmiş tanımına göre yazıldı. Örnekler: kümelenmiş standart hata çarpanları, serbestlik dereceleri, z ya da t seçimi, `summarize, detail` yüzdelikleri, `xtreg` R² tanımları.
- Sonuçlar, Stata ile aynı sonucu verdiğini belgeleyen statsmodels ve linearmodels uygulamalarıyla test paketinde karşılaştırılıyor. Göreli fark genellikle 10⁻⁸'den küçük.
- Ayrıca kodumuzdan bağımsız ücretsiz yazılımlarla (R `estimatr` `se_type = "stata"`, `plm`, `AER`, `sandwich`, `lmtest`, `urca`, `MASS`, `nnet`, `pROC`; gretl; pyfixest) karşılaştırıldı; 22 grubun tamamı tutuyor: [dogrulama/SONUC.md](dogrulama/SONUC.md).
- Kendi Stata'nda bir model çalıştırıp farklılık görürsen lütfen [hata bildir](https://github.com/ytrdev/ekonometri-tr/issues).

**Bilinen farklar:**
- **`dfuller` kritik değerleri:** Stata, Fuller (1976) tablosundan ara değerleme yapar. Bu kütüphane MacKinnon (2010) kritik değerlerini kullanır; değerler arasındaki fark küçüktür. Test istatistiği ve MacKinnon yaklaşık p-değeri Stata ile aynıdır.
- **`xtreg, fe` ek satırları:** Stata'nın raporladığı `sigma_u`, `sigma_e`, `rho` ve `corr(u_i, Xb)` satırları gösterilmez. `xtreg, re` için σ²(birim), σ²(hata) ve θ gösterilir.
- **`xttest0`:** İstatistik Breusch-Pagan (1980) / Baltagi-Li formülüyle havuzlanmış EKK kalıntılarından hesaplanır, p-değeri tek yönlü χ̄²(01) olarak verilir (χ²(1) kuyruğunun yarısı). Stata'nın çıktısıyla karşılaştırılmadı. `ek.panel_analizi()` içindeki LM testi ise klasik iki yönlü biçimdir.
- **`prais`:** Yinelemeli Prais-Winsten / Cochrane-Orcutt. ρ, Stata varsayılanı `rhotype(regress)` gibi hesaplanır ve yakınsamaya kadar yinelenir. Yakınsama ölçütü ve raporlanan R² Stata ile karşılaştırılmadı.
- **Stata'da resmî olmayan testler:**
  - `ek.kpss()` statsmodels / Kwiatkowski vd. (1992) tanımını ve Hobijn vd. (1998) otomatik gecikmesini kullanır. Stata'daki kullanıcı yazımı `kpss` komutunun varsayılanları farklıdır.
  - `ek.engle_granger()` için Stata'da resmî bir komut yoktur.
  - Bu iki test Stata arayüzünde komut olarak sunulmaz.
- **`xtunitroot ips` (gecikmesiz):** Stata gecikme 0 iken dengeli panelde t̃-bar ve Z-t̃-bar raporlar. Bu kütüphane her durumda t-bar ve W-t-bar (IPS 2003, Tablo 3) verir; iki istatistik de asimptotik olarak N(0, 1)'dir ve sonuçlar genellikle aynı kararı verir, ancak sayısal değerler farklıdır.
- **`xtunitroot llc` küçük T:** LLC düzeltme tablosu T̃ = 25'ten başlar; daha kısa panelde ilk satır kullanılır ve uyarı verilir. Küçük T'de LLC boş hipotezi bir miktar fazla reddeder; IPS / Fisher ile birlikte raporla.
- **`nbreg` / sıralı modeller:** Optimizasyon statsmodels ile yapılır, standart hatalar Stata gibi gözlenen bilgi matrisinden (OIM) gelir; sayısal Hessian kullanıldığı için SH'ler beşinci-altıncı basamakta farklılaşabilir.
- **`vargranger`:** İki değişkenli VAR'da standart Granger F testi, üç ve daha fazla değişkende VAR içinde blok dışsallık F testi verilir. Stata `vargranger` Wald χ² raporlar; karar aynıdır, istatistik ölçeği farklıdır.
- **`ardl`:** Gecikmeler varsayılan olarak BIC ile (Stata `ardl` varsayılanı), `aic` seçeneğiyle AIC ile seçilir. Sınır testi Pesaran-Shin-Smith (2001) durum 3'tür.
- **Eksik değer kuralı:** Python fonksiyonlarında (`ek.ekk` vb.) pandas kuralları geçerlidir; eksik gözlemli satırlar modelden çıkarılır. Stata'nın "eksik değer +∞" kuralı yalnızca Stata arayüzündeki ifadelerde (`if`, `gen`, `replace`) uygulanır.
