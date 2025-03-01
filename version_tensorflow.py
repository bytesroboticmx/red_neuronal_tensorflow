try:
    import tensorflow as tf
    print("¡TensorFlow está instalado correctamente!")
    print(f"Versión de TensorFlow: {tf.__version__}")
except ImportError:
    print("Error: TensorFlow no está instalado en tu sistema.")