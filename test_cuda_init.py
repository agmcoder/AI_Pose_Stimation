#!/usr/bin/env python3
"""
Test de inicialización CUDA detallado.
"""
import torch
import os
import sys

print("=== Test de inicialización CUDA ===")
print(f"PyTorch versión: {torch.__version__}")
print(f"CUDA build: {torch.version.cuda}")

# Verificar variables de entorno
print("\n--- Variables de entorno ---")
env_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'LD_LIBRARY_PATH']
for var in env_vars:
    value = os.environ.get(var)
    print(f"{var}: {value if value else '(no definida)'}")

# Intentar inicialización manual
print("\n--- Inicialización CUDA ---")
try:
    # Esto es lo que hace torch.cuda.is_available() internamente
    count = torch._C._cuda_getDeviceCount()
    print(f"Device count (low-level): {count}")
except Exception as e:
    print(f"Error en _cuda_getDeviceCount: {e}")

# Intentar torch.cuda.init()
print("\n--- torch.cuda.init() ---")
try:
    torch.cuda.init()
    print("✅ torch.cuda.init() exitoso")
except Exception as e:
    print(f"❌ torch.cuda.init() falló: {e}")

# Verificar disponibilidad
print("\n--- torch.cuda.is_available() ---")
try:
    available = torch.cuda.is_available()
    print(f"Resultado: {available}")
except Exception as e:
    print(f"Error: {e}")

# Si está disponible, mostrar info
if torch.cuda.is_available():
    print(f"\n--- Información GPU ---")
    print(f"Número de GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memoria total: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
else:
    print("\n--- Diagnóstico adicional ---")
    # Verificar si hay bibliotecas CUDA
    import ctypes
    import ctypes.util

    # Buscar libcuda.so
    libcuda_path = ctypes.util.find_library('cuda')
    print(f"libcuda.so encontrado: {libcuda_path if libcuda_path else 'No'}")

    # Buscar libcudart.so
    libcudart_path = ctypes.util.find_library('cudart')
    print(f"libcudart.so encontrado: {libcudart_path if libcudart_path else 'No'}")

    # Intentar cargar libcuda
    if libcuda_path:
        try:
            libcuda = ctypes.CDLL(libcuda_path)
            print("✅ libcuda.so se puede cargar")
        except Exception as e:
            print(f"❌ Error cargando libcuda.so: {e}")

print("\n=== Fin del test ===")