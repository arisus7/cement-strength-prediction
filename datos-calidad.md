# Datos y Control de Calidad en la Producción de Cemento y Concreto

## Fase 1: Materias Primas Básicas (Química Cruda)
* **Datos Generados (Variables de Entrada / Features - $X$):**
  * **Fluorescencia de Rayos X (XRF):** Análisis químico continuo de la harina cruda.
  * **Composición de Óxidos (% en peso):** $\text{CaO}$, $\text{SiO}_2$, $\text{Al}_2\text{O}_3$, y $\text{Fe}_2\text{O}_3$.
  * **Módulos Químicos Calculados:** 
    * Factor de Saturación de Cal ($LSF$)
    * Módulo de Sílice ($SM$)
    * Módulo de Alúmina ($AM$)
* **Pruebas de Control de Calidad:**
  * **Titulación y Balance Químico:** Monitoreo constante para evitar un $LSF$ desbalanceado que genere exceso de cal libre o problemas de fundición en el horno.

## Fase 2: Clinkerización (Horno Rotatorio)
* **Datos Generados (Variables de Entrada / Features - $X$):**
  * **Sensores SCADA / IoT Industrial (Series de Tiempo):**
    * Temperatura en la zona de quema ($1400^\circ\text{C} - 1650^\circ\text{C}$)
    * Flujo y tipo de combustible
    * Velocidad de rotación del horno (RPM)
    * Velocidad de flujo de aire en enfriadores
  * **Fases Químicas del Clinker (Ecuaciones de Bogue):**
    * Silicato Tricálcico ($\text{C}_3\text{S}$)
    * Silicato Dicálcico ($\text{C}_2\text{S}$)
    * Aluminato Tricálcico ($\text{C}_3\text{A}$)
    * Ferroaluminato Tetracálcico ($\text{C}_4\text{AF}$)
* **Pruebas de Control de Calidad:**
  * **Ensayo de Cal Libre ($\text{CaO}$ Libre):** Cuantifica el calcio no reaccionado. Una alta cal libre indica sub-cocción o mala homogenización del crudo, lo que provoca inestabilidad volumétrica y pérdida de resistencia mecánica.

## Fase 3: Enfriamiento y Molienda (Producto: Cemento)
* **Datos Generados (Variables de Entrada / Features - $X$):**
  * **Parámetros del Molino:** Consumo de energía ($kWh/t$), velocidad del separador de aire y tasa de alimentación.
  * **Dosificación de Aditivos:** Porcentaje exacto de yeso ($\text{CaSO}_4 \cdot 2\text{H}_2\text{O}$) intermolido con el clinker.
* **Pruebas de Control de Calidad:**
  * **Finura Blaine (Superficie Específica, $\text{cm}^2/\text{g}$):** Mide la finura del polvo. A mayor finura, mayor área de contacto con el agua y mayor velocidad de hidratación del $\text{C}_3\text{S}$.
  * **Tiempos de Fraguado (Agujas de Vicat):** Determina los tiempos de fraguado inicial y final, asegurando que la dosificación de yeso controle la hidratación violenta del $\text{C}_3\text{A}$.

## Fase 4: Aplicación e Hidratación (Control de Calidad Final)
* **Datos Generados (Variables Objetivo / Target - $Y$):**
  * **Registros de Ensayos Destructivos de Laboratorio:** Es el objetivo principal a predecir ($Y$).
  * **Variables:** Resistencia a la compresión medida en $MPa$ o $PSI$ a diferentes edades del espécimen (3, 7 y 28 días).
* **Pruebas de Control de Calidad:**
  * **Ruptura en Prensa Hidráulica:** Ensayos de compresión sobre cubos o probetas de mortero/concreto curados en agua.
  * **Resistencia Temprana (3 y 7 Días):** Mide la velocidad de desarrollo mecánico inicial, dominada por la hidratación rápida del $\text{C}_3\text{S}$.
  * **Resistencia Estructural Oficial (28 Días):** Punto de referencia normativo de calidad, impulsado por la hidratación continua y tardía del $\text{C}_2\text{S}$.
