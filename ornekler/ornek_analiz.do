* ============================================================================
* ekonometri-tr ile çalışan örnek do dosyası
*   Python'da:  import ekonometri as ek; ek.Stata().do("ornekler/ornek_analiz.do")
* Kendi verin için 'sysuse ucret' yerine:  use "verim.dta", clear
* ============================================================================

sysuse ucret

* --- değişkenler -------------------------------------------------------------
gen lnw = ln(ucret)
label variable lnw "Ücretin logaritması"
label variable egitim "Eğitim süresi (yıl)"
label variable tecrube "İş tecrübesi (yıl)"

summarize ucret egitim tecrube
tabulate cinsiyet

* --- modeller ----------------------------------------------------------------
quietly regress lnw egitim, vce(robust)
estimates store m1

quietly regress lnw egitim c.tecrube##c.tecrube, vce(robust)
estimates store m2

regress lnw egitim c.tecrube##c.tecrube i.cinsiyet i.sektor i.bolge, vce(robust)
estimates store m3

* --- tanı testleri (klasik standart hatalarla) --------------------------------
quietly regress lnw egitim c.tecrube##c.tecrube i.cinsiyet i.sektor i.bolge
estat hettest
estat ovtest
estat vif

* --- tablo ve Türkçe rapor ---------------------------------------------------
esttab m1 m2 m3
quietly regress lnw egitim c.tecrube##c.tecrube i.cinsiyet i.sektor i.bolge, vce(robust)
yorum
