
# vigas_app (Web local con Flask) — versión corregida

Aplicación web local para ingresar datos de una viga, calcular reacciones y generar diagramas y un **PDF** con el informe.

## Requisitos
- Python 3.9+ recomendado (funciona con 3.7+ si tenés Anaconda)
- `pip install -r requirements.txt`

## Ejecutar
```bash
python app.py
```
Abrí `http://127.0.0.1:5000` en el navegador.

## Notas
- El formulario admite **coma o punto** decimal; internamente se normaliza a punto.
- **Arreglo aplicado**: se procesan bien cargas **point** aunque el campo "Posición final x" esté deshabilitado.
- Reacciones mostradas **positivas hacia arriba**.
