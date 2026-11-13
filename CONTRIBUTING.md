# Katkı

Hata, bozuk Türkçe ya da eksik test görürsen issue aç. Kısa bir örnek koy ki aynı hatayı biz de görelim.

Yeni bir yöntem yazacaksan önce issue aç. README'deki yol haritasına bak; aynı iş iki kere yazılmasın.

Ders defteri ya da simüle veri ekleyebilirsin. Veri uydurmaysa bunu dosyada yaz. Gerçek veri koyacaksan kaynağını ve lisansını da yaz.

## Kurulum

```bash
git clone https://github.com/ytrdev/ekonometri-tr
cd ekonometri-tr
python -m venv .venv
```

Windows'ta `.venv\Scripts\activate`, macOS ve Linux'ta `source .venv/bin/activate`.

```bash
pip install -e ".[gelistirme]"
pytest
ruff check .
ruff format .
```

## Yazarken

Yeni bir tahmincinin sayıları testte bir referansla dursun: statsmodels, linearmodels ya da R. Mevcut testlerde göreli fark kabaca 10⁻⁸ civarında tutuluyor.

Ekrana çıkan metin Türkçe olsun. Değişken adının ardına kesme ile ek yapıştırma. `'Kamu'e` olmaz; "Kamu grubu", "değişkeni" yaz.

Fonksiyon ve değişken adları Türkçe, ASCII harfle: `kalintilar`, `varsayimlar`. Docstring de Türkçe.

## Pull request

1. GitHub'da Fork'a bas. Kendi kopyanda dal aç: `git checkout -b ozellik/kisa-ad`
2. Testi ekle. `pytest` ve `ruff check .` geçsin.
3. `CHANGELOG.md` içine ne değiştiğini bir satır yaz.
4. PR'da ne yaptığını Türkçe anlat.
