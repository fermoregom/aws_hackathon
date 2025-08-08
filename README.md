# AutoPartes AI - Recomendador de Piezas

## Descripción del Proyecto

AutoPartes AI es una aplicación web inteligente desarrollada con Streamlit que ayuda a los usuarios a encontrar piezas de automóviles de manera eficiente. La aplicación combina tradición y tecnología para ofrecer una experiencia de usuario moderna en la búsqueda de autopartes.

### Características principales:
- **Interfaz intuitiva** con navegación por pestañas
- **Asistente AI** para consultas conversacionales sobre piezas
- **Catálogo completo** con filtros avanzados por marca, modelo y tipo de pieza
- **Resultados de análisis** personalizados
- **Base de datos** cargada desde archivo CSV

## Cómo Desplegar

### Requisitos previos
- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone <url-del-repositorio>
   cd aws_hackathon
   ```

2. **Instalar dependencias:**
   ```bash
   pip install streamlit pandas
   ```

3. **Ejecutar la aplicación:**
   ```bash
   streamlit run main.py
   ```

4. **Acceder a la aplicación:**
   - La aplicación se abrirá automáticamente en tu navegador
   - Si no se abre, ve a: `http://localhost:8501`

### Archivos necesarios
- `main.py` - Aplicación principal
- `base_autopartes_dummy.csv` - Base de datos de piezas
- `AutoPartes AI_ Tradición y Tecnología.png` - Imagen de inicio

### Estructura del proyecto
```
aws_hackathon/
├── main.py                                    # Aplicación principal
├── base_autopartes_dummy.csv                  # Base de datos
├── AutoPartes AI_ Tradición y Tecnología.png  # Imagen de inicio
├── src/                                       # Código fuente adicional
└── README.md                                  # Este archivo
```
