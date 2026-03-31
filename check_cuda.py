#!/usr/bin/env python3
"""
Diagnóstico de compatibilidad CUDA para el proyecto de pose tracking.
"""
import subprocess
import sys
import os

def run_command(cmd):
    """Ejecutar comando y retornar salida."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def check_nvidia_driver():
    """Verificar driver NVIDIA."""
    print("🔍 Verificando driver NVIDIA...")
    code, out, err = run_command("nvidia-smi")
    if code == 0:
        print("✅ NVIDIA driver encontrado")
        # Extraer versión CUDA
        for line in out.split('\n'):
            if 'CUDA Version' in line:
                print(f"   Versión CUDA driver: {line.split(':')[-1].strip()}")
        return True
    else:
        print("❌ NVIDIA driver no encontrado o nvidia-smi no disponible")
        print(f"   Error: {err}")
        return False

def check_cuda_toolkit():
    """Verificar CUDA Toolkit instalado."""
    print("\n🔍 Verificando CUDA Toolkit...")
    code, out, err = run_command("nvcc --version")
    if code == 0:
        print("✅ CUDA Toolkit encontrado")
        lines = out.split('\n')
        for line in lines:
            if 'release' in line.lower():
                print(f"   {line.strip()}")
        return True
    else:
        print("❌ CUDA Toolkit no encontrado (nvcc)")
        return False

def check_pytorch():
    """Verificar PyTorch instalación y compatibilidad CUDA."""
    print("\n🔍 Verificando PyTorch...")
    try:
        import torch
        print(f"✅ PyTorch versión: {torch.__version__}")

        # Check CUDA build
        if hasattr(torch.version, 'cuda'):
            print(f"   Compilado con CUDA: {torch.version.cuda}")
        else:
            print("   ⚠️  PyTorch compilado sin soporte CUDA")

        # Check CUDA available
        if torch.cuda.is_available():
            print(f"   ✅ CUDA disponible")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Versión CUDA runtime: {torch.version.cuda}")
        else:
            print("   ❌ CUDA no disponible en PyTorch")
            # Try to get error
            try:
                torch.cuda.init()
            except Exception as e:
                print(f"   Error inicialización: {e}")

        return True
    except ImportError:
        print("❌ PyTorch no instalado")
        return False

def check_environment_vars():
    """Verificar variables de entorno CUDA."""
    print("\n🔍 Verificando variables de entorno CUDA...")
    env_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'LD_LIBRARY_PATH', 'PATH']
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"   {var}: {value}")
        else:
            print(f"   {var}: (no definida)")

def check_conda_env():
    """Verificar si estamos en entorno conda."""
    print("\n🔍 Verificando entorno...")
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        print(f"✅ Entorno Conda: {conda_prefix}")
    elif os.path.exists('.venv') or os.path.exists('venv'):
        print("✅ Entorno virtual Python detectado")
    else:
        print("⚠️  No se detectó entorno virtual/conda")

def main():
    print("=" * 60)
    print("DIAGNÓSTICO CUDA - Pose Tracking Project")
    print("=" * 60)

    check_conda_env()
    check_nvidia_driver()
    check_cuda_toolkit()
    check_pytorch()
    check_environment_vars()

    print("\n" + "=" * 60)
    print("RECOMENDACIONES:")
    print("-" * 60)

    # Proporcionar recomendaciones basadas en diagnóstico
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  CUDA no está disponible para PyTorch.")
            print("   Soluciones posibles:")
            print("   1. Instalar PyTorch con CUDA 11.8 compatible:")
            print("      pip uninstall torch torchvision torchaudio")
            print("      pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118")
            print("   2. Usar entorno Conda con environment_linux_cuda.yml actualizado")
            print("   3. Verificar que CUDA Toolkit 11.8+ esté instalado")
    except ImportError:
        print("❌ PyTorch no instalado. Instalar con soporte CUDA.")

    print("=" * 60)

if __name__ == "__main__":
    main()