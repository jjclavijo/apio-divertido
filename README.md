### el Etor

Esto es cada vez menos un fork.

Para poder comparar con hector, el modulo `control` es ahora un parser que
convierte archivos .ctl a .toml y .toml a .ctl+.cli (con la opción `-I`)

La idea es:

1. Generar un `xxxx.toml` válido
2. Convertirlo con `python -m eletor.control -I xxxx.toml`
3. Correr hector con `simulatenoise -i xxxx.ctl < xxxx.cli`
4. Ver cómo comparar el archivo .mom

Para comparar la salida de hector tenemos que dejar el writer de archivos .mom
en eletor, pero el camino es eliminar todo el resto de observations.py

### el Etor

Es un fork parcial de hectorp con muchas modificaciones para hacer que la simulación sea razonablemente funcional.

Algunas cosas raras aparecen como en control.py y observations.py que son por compatibilidad con hectorp para poder hacer algunos tests de que la salida es la misma que da el programa original de Machiel Bos.

Los test que estan en la carpeta test corren con pytest. basicamente son heredados de algunas pruebas que hice para garantizar que la equivalencia entre versiones (mía y la original).

En los tests el test `test_and...` genera un conjunto de Inputs / Outputs en una carpeta test/bmks. Tener en cuenta que el ruido usado para generar las muestras es el que está en el archivo `test/10000Random.dat` Este archivo es directamente una cadena de bytes que representan float64. En python se lee con:

```python
from array import array

ar = array('d')
with open('....','rb') as f:
    ar.fromfile(f,10000)
    #10000 porque son 10000 floats
```

No se si puede haber algún tipo de variación en la representación del float pero debe estar bien.

Los Outputs para benchmark son el mismo tipo de cadenas de bytes.
Los Inputs son json.
