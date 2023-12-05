# Инструменты

Для форматирования и проверки качества кода были выбраны  инструметы:

- форматтер **black**
- линтер **Pylint**

## Formatter black

Использование: 

```
black .
```

**black** форматирует .py файлы в директории. Вывод команды:

```
reformatted C:\Users\Admin\aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\ml_engineering\utils.py
reformatted C:\Users\Admin\aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\ml_engineering\main.py

All done! ✨ 🍰 ✨
2 files reformatted.
```

К каким конкретно изменениям в файлах привел иструмент можно посмотреть в истории коммитов.

## Иструмент pylint

В качестве линтера был взят **pylint**. Он используется для анализа логики и стилистики кода. Пример для одного из файлов проекта без указания дополнительных опций.

```
pylint utils.py
```

```
************* Module utils
utils.py:1:0: C0114: Missing module docstring (missing-module-docstring)
utils.py:1:0: C0116: Missing function or method docstring (missing-function-docstring)
utils.py:2:13: R1734: Consider using [] instead of list() (use-list-literal)
utils.py:10:0: C0116: Missing function or method docstring (missing-function-docstring)
utils.py:12:8: C0200: Consider using enumerate instead of iterating with range and len (consider-using-enumerate)

-----------------------------------
Your code has been rated at 5.45/10

```

Замечания кроме докстрингов являются конструктивными и могут привести к улучшению восприятия логики программы.