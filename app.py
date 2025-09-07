# app.py
from flask import Flask, render_template, request, Response, redirect, url_for, send_file, jsonify
import json
import subprocess
import io
import zipfile
import uuid
from pathlib import Path
import traceback

# --- Matplotlib en modo headless ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from beam_model import (
    BeamAnalyzer,
    PointLoad,
    DistributedLinearLoad,
    AppliedMoment,
    Support,
)

app = Flask(__name__)

# Carpeta para imágenes generadas
BASE_DIR = Path(__file__).resolve().parent
GEN_IMG = BASE_DIR / "generated" / "images"
GEN_IMG.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Utilidades
# -----------------------------
def f2(s):
    """Convierte string con coma o punto decimal en float."""
    if s is None or s == "":
        return None
    try:
        return float(str(s).replace(",", ".").strip())
    except (ValueError, AttributeError) as e:
        print(f"[WARN] No se pudo convertir a float: {s} - Error: {e}")
        return None

def loads_json(form, key, default="[]"):
    """
    Lee un campo JSON del form de forma robusta.
    Devuelve lista (o default parseado) y nunca lanza excepción al llamador.
    Soporta default vacío ("") -> [].
    """
    raw = form.get(key, None)
    if raw is None or raw.strip() == "":
        if default is None or (isinstance(default, str) and default.strip() == ""):
            return []
        try:
            return json.loads(default)
        except json.JSONDecodeError as e:
            print(f"[WARN] JSON inválido en default '{key}': {e}")
            return []
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON inválido en '{key}': {e}")
        if default is None or (isinstance(default, str) and default.strip() == ""):
            return []
        try:
            return json.loads(default)
        except json.JSONDecodeError:
            return []

def wants_json_response():
    """True si el cliente espera JSON (query ?as=json o cabecera Accept)."""
    return (request.args.get("as") == "json") or ("application/json" in (request.headers.get("Accept", "")))

def render_beam_plots(beam, units):
    """
    Genera y guarda las gráficas de V(x) y M(x) con Matplotlib.
    Retorna rutas absolutas (Path) a los archivos de imagen.
    """
    if not hasattr(beam, "x") or not hasattr(beam, "V") or not hasattr(beam, "M"):
        raise RuntimeError("El modelo no ha generado arreglos. Asegúrate de llamar generate_arrays().")

    x = beam.x
    V = beam.V
    M = beam.M
    if len(x) == 0 or len(V) == 0 or len(M) == 0:
        raise RuntimeError("Los arreglos están vacíos. Verifica el modelo de viga.")

    fig_paths = []

    # --- V(x) ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x, V, linewidth=2)
    ax1.axhline(0, linewidth=0.8, linestyle='--', alpha=0.7)
    ax1.set_xlabel(f"x ({units['x']})")
    ax1.set_ylabel(f"V ({units['V']})")
    ax1.set_title("Diagrama de cortante V(x)")
    ax1.grid(True, alpha=0.3)
    f1 = GEN_IMG / f"V_{uuid.uuid4().hex}.png"
    fig1.savefig(f1, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    fig_paths.append(f1)

    # --- M(x) ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(x, M, linewidth=2)
    ax2.axhline(0, linewidth=0.8, linestyle='--', alpha=0.7)
    ax2.set_xlabel(f"x ({units['x']})")
    ax2.set_ylabel(f"M ({units['M']})")
    ax2.set_title("Diagrama de momento M(x)")
    ax2.grid(True, alpha=0.3)
    f2 = GEN_IMG / f"M_{uuid.uuid4().hex}.png"
    fig2.savefig(f2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    fig_paths.append(f2)

    return fig_paths

def _ensure_package(tex, pkg):
    """Inserta \\usepackage{pkg} si no existe ya en el preámbulo."""
    needle = f"\\usepackage{{{pkg}}}"
    if needle in tex:
        return tex
    idx = tex.find("\\documentclass")
    if idx == -1:
        return needle + "\n" + tex
    line_end = tex.find("\n", idx)
    if line_end == -1:
        line_end = len(tex)
    return tex[:line_end+1] + needle + "\n" + tex[line_end+1:]

def _append_figures(tex, figV_name, figM_name, uV, uM):
    """Agrega sección de figuras antes de \\end{document}. Usa \\detokenize en las rutas."""
    block = rf"""
\section*{{Diagramas}}
\begin{{figure}}[h]
  \centering
  \includegraphics[width=\linewidth]{{\detokenize{{{figV_name}}}}}
  \caption{{Diagrama de cortante \(V(x)\) en \(\SI{{}}{{{uV}}}\).}}
\end{{figure}}

\begin{{figure}}[h]
  \centering
  \includegraphics[width=\linewidth]{{\detokenize{{{figM_name}}}}}
  \caption{{Diagrama de momento \(M(x)\) en \(\SI{{}}{{{uM}}}\).}}
\end{{figure}}
"""
    if "\\end{document}" in tex:
        return tex.replace("\\end{document}", block + "\n\\end{document}")
    else:
        return tex + "\n" + block

def _inject_graphics_packages(tex):
    """Asegura paquetes para incluir imágenes y formato."""
    tex = _ensure_package(tex, "graphicx")
    tex = _ensure_package(tex, "caption")
    if "\\usepackage{siunitx}" not in tex:
        tex = _ensure_package(tex, "siunitx")
        if "output-decimal-marker" not in tex and "\\begin{document}" in tex:
            tex = tex.replace("\\begin{document}",
                              "\\sisetup{output-decimal-marker = {,}}\n\\begin{document}")
    return tex

def build_tex_with_figs(beam):
    """
    Construye el .tex base con beam.render_latex_document(),
    asegura paquetes de imágenes y agrega las figuras V/M.
    Inserta rutas por nombre de archivo (para usarse con ZIP).
    """
    base_tex = beam.render_latex_document()
    units = beam.units if hasattr(beam, "units") else {"x": "m", "V": "kN", "M": "kN·m"}
    figV_path, figM_path = [Path(p) for p in render_beam_plots(beam, units)]
    tex = _inject_graphics_packages(base_tex)
    tex = _append_figures(tex, figV_path.name, figM_path.name, units["V"], units["M"])
    return tex, [figV_path, figM_path]

# -----------------------------
# Rutas
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # --- leer “raw” para mantener consistencia en descargas ---
        length_raw = request.form.get("length", "").strip()
        ux = request.form.get("ux", "m")
        uw = request.form.get("uw", "kN/m")
        uV = request.form.get("uV", "kN")
        uM = request.form.get("uM", "kN.m")
        point_loads_raw       = (request.form.get("point_loads") or "[]").strip()
        distributed_loads_raw = (request.form.get("distributed_loads") or "[]").strip()
        moments_raw           = (request.form.get("moments") or "[]").strip()
        supports_raw          = (request.form.get("supports") or "").strip()
        if not supports_raw:
            supports_raw = f'[\n  {{"x":"0,0","type":"pinned"}},\n  {{"x":"{length_raw if length_raw else "0,0"}","type":"roller"}}\n]'

        # --- resolver modelo ---
        beam = build_beam_from_form(request.form)
        points = beam.critical_points_summary()
        ext    = beam.extrema()

        # --- contexto para HTML ---
        ctx = {
            "length": beam.length,
            "length_raw": length_raw,
            "units": {"x": ux, "w": uw, "V": uV, "M": uM},
            "supports": beam.supports,
            "reactions": beam.reactions,
            "fix_moment": beam.fix_moment,
            "points": points,
            "extrema": ext,
            "supports_raw": supports_raw,
            "point_loads_raw": point_loads_raw,
            "distributed_loads_raw": distributed_loads_raw,
            "moments_raw": moments_raw,
        }

        # --- si piden JSON, respondemos JSON liviano ---
        if wants_json_response():
            sumFy, sumM0 = beam._equivalents_about(0.0)
            sumFy_total = sumFy + sum(beam.reactions.values())
            sumM0_total = sumM0 + sum(xR * R for xR, R in beam.reactions.items()) + sum(beam.fix_moment.values())
            return jsonify(ok=True, sumFy_total=sumFy_total, sumM0_total=sumM0_total)

        # --- camino HTML tradicional ---
        return render_template("result.html", **ctx)

    except Exception as e:
        err = str(e)
        print(f"[ERROR] Error en análisis: {err}")
        print(traceback.format_exc())
        if wants_json_response():
            return jsonify(ok=False, error=err), 400
        return f"Error en el análisis: {err}", 400

@app.route("/download-tex", methods=["GET", "POST"])
def download_tex():
    """Genera y descarga el archivo LaTeX .tex con imágenes referenciadas por nombre."""
    if request.method == "GET":
        return redirect(url_for("index"))
    try:
        beam = build_beam_from_form(request.form)
        tex_str, _ = build_tex_with_figs(beam)
        return Response(
            tex_str,
            mimetype="application/x-tex",
            headers={"Content-Disposition": 'attachment; filename="reporte_viga.tex"'}
        )
    except Exception as e:
        print(f"[ERROR] Error generando archivo TEX: {e}")
        print(traceback.format_exc())
        return f"Error generando archivo TEX: {str(e)}", 400

@app.route("/download-zip", methods=["POST"])
def download_zip():
    """Descarga un ZIP con: reporte_viga.tex + imágenes generadas."""
    try:
        beam = build_beam_from_form(request.form)
        tex_str, fig_paths = build_tex_with_figs(beam)

        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("reporte_viga.tex", tex_str)
            for p in fig_paths:
                p = Path(p)
                if p.exists():
                    zf.write(p, arcname=p.name)
                else:
                    print(f"[WARN] Archivo de imagen no encontrado: {p}")
        mem.seek(0)
        return send_file(mem, mimetype="application/zip", as_attachment=True,
                         download_name="reporte_viga_tex_y_figuras.zip")
    except Exception as e:
        print(f"[ERROR] Error generando ZIP: {e}")
        print(traceback.format_exc())
        return f"Error generando ZIP: {str(e)}", 400

# -----------------------------
# Construcción del modelo
# -----------------------------
def build_beam_from_form(form):
    """Construye y resuelve el modelo de viga a partir del form."""
    try:
        length = f2(form.get("length"))
        if length is None or length <= 0:
            raise ValueError("La longitud de la viga debe ser un número positivo.")

        units = {
            "x": form.get("ux", "m"),
            "w": form.get("uw", "kN/m"),
            "V": form.get("uV", "kN"),
            "M": form.get("uM", "kN·m"),
        }

        beam = BeamAnalyzer(length=length, units=units)

        # Apoyos
        supports_data = loads_json(form, "supports", default="")
        if (not supports_data) and (form.get("supports") is None or form.get("supports").strip() == ""):
            supports_data = [
                {"x": "0,0", "type": "pinned"},
                {"x": str(length).replace(".", ","), "type": "roller"},
            ]
        supports = []
        for s in supports_data:
            if not isinstance(s, dict):
                continue
            xs = f2(s.get("x"))
            st = (s.get("type") or "").strip().lower()
            if xs is None:
                continue
            if not (0 <= xs <= length):
                continue
            if st not in {"pinned", "roller", "fixed"}:
                continue
            supports.append(Support(position=xs, kind=st))
        if not supports:
            raise ValueError("No hay apoyos válidos. Verifica el JSON 'supports'.")
        beam.set_supports(supports)

        # Cargas puntuales
        for item in loads_json(form, "point_loads", default="[]"):
            if not isinstance(item, dict):
                continue
            x_val = f2(item.get("x"))
            P_val = f2(item.get("P"))
            if x_val is None or P_val is None:
                continue
            if 0 <= x_val <= length:
                beam.add_point_load(PointLoad(position=x_val, magnitude=P_val))

        # Cargas distribuidas lineales
        for item in loads_json(form, "distributed_loads", default="[]"):
            if not isinstance(item, dict):
                continue
            xa = f2(item.get("xa")); xb = f2(item.get("xb"))
            wa = f2(item.get("wa")); wb = f2(item.get("wb"))
            if None in (xa, xb, wa, wb):
                continue
            if not (0 <= xa < xb <= length):
                continue
            beam.add_distributed_load(
                DistributedLinearLoad(x_start=xa, x_end=xb, w_start=wa, w_end=wb)
            )

        # Momentos aplicados (CCW +)
        for item in loads_json(form, "moments", default="[]"):
            if not isinstance(item, dict):
                continue
            x_val = f2(item.get("x"))
            M_val = f2(item.get("M"))
            if x_val is None or M_val is None:
                continue
            if 0 <= x_val <= length:
                beam.add_moment(AppliedMoment(position=x_val, magnitude=M_val))

        beam.solve_reactions()
        beam.generate_arrays()
        return beam

    except Exception as e:
        print(f"[ERROR] Error construyendo beam: {e}")
        print(traceback.format_exc())
        raise

# -----------------------------
# Manejo de errores (sin templates)
# -----------------------------
@app.errorhandler(404)
def not_found(error):
    if wants_json_response():
        return jsonify(ok=False, error="Página no encontrada"), 404
    return "Página no encontrada", 404

@app.errorhandler(500)
def internal_error(error):
    msg = getattr(error, "description", str(error))
    if wants_json_response():
        return jsonify(ok=False, error=msg), 500
    return f"Error interno del servidor: {msg}", 500

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"[ERROR] Excepción no manejada: {e}")
    print(traceback.format_exc())
    if wants_json_response():
        return jsonify(ok=False, error=str(e)), 500
    return f"Error inesperado: {str(e)}", 500

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    try:
        subprocess.run(["pdflatex", "--version"],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True)
        print("pdflatex encontrado - funcionalidad PDF disponible.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("pdflatex no encontrado - se descarga .tex igualmente.")

    print(f"Directorio de imágenes: {GEN_IMG}")
    # debug=False para que el debugger de Werkzeug no reemplace nuestras respuestas JSON
    app.run(host="127.0.0.1", port=5000, debug=False)
