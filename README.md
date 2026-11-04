## Despliegue Automatizado con Databricks Asset Bundles (DABs)

El proyecto utiliza Databricks Asset Bundles para garantizar reproducibilidad exacta de extremo a extremo mediante Infrastructure as Code (IaC).

### Requisitos Previos
* Databricks CLI (v0.205 o superior) instalado y autenticado en su workspace. Con:
```bash
  databricks auth login --host https://<tu-workspace-url>
```
ignorando cualquier ruta posterior como /?o=

### Pasos de Ejecucion
1. Clone este repositorio localmente:
 ```bash  
   git clone https://github.com/TU_USUARIO/cement-strength-prediction.git
   cd cement-strength-prediction
```
2. Valide y despliegue el bundle hacia su workspace:
```bash
   databricks bundle deploy
```
3. Ejecute el pipeline orquestado:
```bash
   databricks bundle run gcc_pipeline
```
El comando creara de manera autonoma el Job con sus 6 tareas enlazadas en Databricks Workflows, aprovisionara el catalogo, instalara dependencias de PyPI y ejecutara el entrenamiento de los modelos.
