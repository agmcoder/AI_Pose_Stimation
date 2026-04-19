#!/usr/bin/env python3
"""
Script para unir múltiples archivos CSV con el mismo esquema en un único archivo.

Uso:
    python merge_csvs.py --output_file training/data/merged.csv
    python merge_csvs.py --output_dir training/data/merged_results --add_source_file
"""

# EJEMPLOS DE USO DESDE TERMINAL:
# ---------------------------------------------------------------------------
# 1. Unir todos los CSV terminados en *cleaned.csv de training/data y guardar en archivo específico:
#    python merge_csvs.py --output_file training/data/merged_squats.csv
#
# 2. Unir y añadir columna source_file con nombre del archivo original:
#    python merge_csvs.py --output_file training/data/merged.csv --add_source_file
#
# 3. Crear carpeta de resultados y guardar merged.csv dentro:
#    python merge_csvs.py --output_dir training/data/merged_results
#
# 4. Especificar carpeta de entrada personalizada:
#    python merge_csvs.py --input_folder otra/carpeta --output_file salida.csv
#
# 5. Combinar opciones (carpeta personalizada, source_file, directorio de salida):
#    python merge_csvs.py --input_folder datos/procesados --output_dir resultados --add_source_file
# ---------------------------------------------------------------------------

import argparse
import sys
from pathlib import Path
import traceback

# Verificar dependencias necesarias
try:
    import pandas as pd
except ImportError:
    print("❌ Error: La biblioteca 'pandas' no está instalada.")
    print("   Instálala con: pip install pandas")
    sys.exit(1)


def merge_csv_files(input_folder: Path, output_file: Path, add_source_file: bool = False) -> bool:
    """
    Busca archivos CSV terminados en '*cleaned.csv' en la carpeta de entrada,
    los concatena y guarda el resultado.

    Args:
        input_folder: Directorio donde buscar los archivos CSV.
        output_file: Ruta donde guardar el CSV unificado.
        add_source_file: Si True, añade columna 'source_file' con nombre del archivo original.

    Returns:
        True si la operación fue exitosa, False si hubo errores fatales.

    Ejemplo de uso desde otro script Python:
        >>> from pathlib import Path
        >>> from merge_csvs import merge_csv_files
        >>>
        >>> input_folder = Path("training/data")
        >>> output_file = Path("training/data/merged.csv")
        >>> success = merge_csv_files(input_folder, output_file, add_source_file=True)
        >>> if success:
        >>>     print("CSV unificado creado exitosamente")
    """
    # Verificar que la carpeta de entrada existe
    if not input_folder.is_dir():
        print(f"❌ Error: La carpeta de entrada '{input_folder}' no existe o no es un directorio.")
        return False

    # Buscar archivos CSV que terminen en 'cleaned.csv'
    # Usamos glob para encontrar cualquier CSV, luego filtramos por nombre
    csv_files = list(input_folder.glob("*.csv"))
    # Filtrar aquellos cuyo nombre termina en 'cleaned.csv' (no sensible a mayúsculas)
    cleaned_csvs = [f for f in csv_files if f.name.lower().endswith("cleaned.csv")]

    if not cleaned_csvs:
        print(f"⚠️  No se encontraron archivos CSV que terminen en 'cleaned.csv' en '{input_folder}'.")
        print(f"   Archivos CSV encontrados: {[f.name for f in csv_files]}")
        return False

    print(f"📂 Encontrados {len(cleaned_csvs)} archivos 'cleaned.csv':")
    for f in cleaned_csvs:
        print(f"   • {f.name}")

    # Leer y concatenar todos los CSV
    dataframes = []
    columns_order = None  # Para mantener el orden de columnas del primer archivo válido

    for csv_file in cleaned_csvs:
        try:
            print(f"📄 Leyendo {csv_file.name}...", end=" ", flush=True)
            df = pd.read_csv(csv_file, encoding='utf-8')

            if df.empty:
                print("(vacío, se omite)")
                continue

            # Si es el primer DataFrame no vacío, guardamos el orden de columnas
            if columns_order is None:
                columns_order = df.columns.tolist()

            # Añadir columna source_file si se solicita
            if add_source_file:
                df = df.copy()  # Evitar SettingWithCopyWarning
                df['source_file'] = csv_file.name

            dataframes.append(df)
            print(f"✅ {len(df)} filas")

        except pd.errors.EmptyDataError:
            print(f"⚠️  (vacío, se omite)")
        except Exception as e:
            print(f"❌ Error leyendo {csv_file.name}: {e}")
            print("   Continuando con los demás archivos...")
            # Opcional: mostrar traza detallada para depuración
            # traceback.print_exc()

    if not dataframes:
        print("❌ No se pudo leer ningún archivo CSV con datos válidos.")
        return False

    # Concatenar todos los DataFrames
    print("\n🔗 Concatenando DataFrames...")
    try:
        merged_df = pd.concat(dataframes, ignore_index=True, sort=False)

        # Asegurar el orden de columnas original
        if columns_order is not None:
            # Si añadimos source_file, la ponemos al final
            if add_source_file and 'source_file' not in columns_order:
                columns_order = columns_order + ['source_file']
            # Reordenar columnas según el orden original
            merged_df = merged_df.reindex(columns=columns_order)

        # Resetear índice (ignore_index ya lo hace, pero por si acaso)
        merged_df.reset_index(drop=True, inplace=True)

    except Exception as e:
        print(f"❌ Error al concatenar DataFrames: {e}")
        traceback.print_exc()
        return False

    # Guardar el CSV final
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"💾 CSV unificado guardado en: {output_file}")
        print(f"📊 Estadísticas finales:")
        print(f"   • Total de filas: {len(merged_df):,}")
        print(f"   • Total de columnas: {len(merged_df.columns)}")
        print(f"   • Columnas: {', '.join(merged_df.columns.tolist())}")
        return True

    except Exception as e:
        print(f"❌ Error al guardar el archivo de salida: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Unir múltiples archivos CSV (terminados en *cleaned.csv) en un único archivo.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Usando --output_file (ruta completa al archivo)
  %(prog)s --output_file training/data/merged.csv
  %(prog)s --output_file training/data/merged.csv --add_source_file

  # Usando --output_dir (crea merged.csv dentro de la carpeta)
  %(prog)s --output_dir training/data/merged_results
  %(prog)s --output_dir training/data/merged_results --add_source_file

  # Especificando input_folder personalizado
  %(prog)s --input_folder otra/carpeta --output_file salida.csv
        """
    )

    parser.add_argument(
        '--input_folder',
        default='training/data',
        help='Carpeta donde buscar archivos CSV terminados en *cleaned.csv (default: training/data)'
    )
    parser.add_argument(
        '--output_file',
        help='Ruta completa del archivo CSV de salida unificado (ej: training/data/merged.csv)'
    )
    parser.add_argument(
        '--output_dir',
        help='Carpeta donde guardar el archivo merged.csv (se crea el archivo dentro)'
    )
    parser.add_argument(
        '--add_source_file',
        action='store_true',
        help='Añadir columna "source_file" con el nombre del archivo original'
    )

    args = parser.parse_args()

    # Convertir a Path objetos
    input_folder = Path(args.input_folder).resolve()

    # Determinar ruta de salida
    output_file_path = None

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        # Crear directorio si no existe
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / "merged.csv"
        print(f"📁 Usando directorio de salida: {output_dir}")
    elif args.output_file:
        output_file_path = Path(args.output_file).resolve()
        # Crear directorio padre si no existe
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        print("❌ Error: Debes especificar --output_file o --output_dir")
        sys.exit(1)

    print("=" * 60)
    print("🔄 MERGE CSVs - Unificación de archivos CSV")
    print("=" * 60)
    print(f"Carpeta de entrada: {input_folder}")
    print(f"Archivo de salida:  {output_file_path}")
    print(f"Añadir source_file: {'Sí' if args.add_source_file else 'No'}")
    print("-" * 60)

    success = merge_csv_files(input_folder, output_file_path, args.add_source_file)

    if success:
        print("\n✅ Proceso completado exitosamente.")
        sys.exit(0)
    else:
        print("\n❌ Proceso finalizado con errores.")
        sys.exit(1)


if __name__ == "__main__":
    main()