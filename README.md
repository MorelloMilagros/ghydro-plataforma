# ghydro-plataforma
Este repositorio contendrá todo el software que corre en la nube - servidor y página web-. La estructura será simil monorepo, una donde ambos proyectos conviven de forma ordenada. Aquí vive la plataforma web del proyecto.
backend/: Contiene el código del servidor.

fuente/api/: Las rutas de la API (ej. /datos, /alertas).

fuente/servicios/: La lógica de negocio (cómo se procesan los datos).

fuente/modelos/: La estructura de los datos para la base de datos.

frontend/: Contiene el código de la interfaz web que verá el usuario.

fuente/componentes/: Piezas reutilizables como botones o gráficos.

fuente/paginas/: Las diferentes pantallas de la aplicación (el dashboard, la configuración, etc.).

fuente/recursos/: Para guardar imágenes, logos y otros archivos estáticos.

instalar las librerias:
pip install paho-mqtt
pip install influxdb-client
pip install python-dotenv