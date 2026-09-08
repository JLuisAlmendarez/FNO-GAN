# GA-FNO — Generative Adversarial Fourier Neural Operator para CFD

Modelo surrogate para turbulencia 2D de Kolmogorov, combinando un Fourier
Neural Operator (FNO) con un esquema de entrenamiento adversarial (GAN).
Proyecto de investigación desarrollado en ITESO.

**Autores:** José Luis Almendarez González, José Iván Sandoval Ruiz Velasco,
Maximiliano Aguayo Villanueva.

📄 Reporte completo: [`GAN.pdf`](./GAN.pdf)

## Resumen

El modelo (GA-FNO) usa dos discriminadores especializados durante el
entrenamiento: uno evalúa la coherencia estadística de las trayectorias
generadas, y el otro evalúa qué tan bien se respetan los residuos de
Navier-Stokes entre frames consecutivos.

Se exploran además tres extensiones de post-procesamiento sobre el modelo
ya entrenado:

- **Generalización zero-shot a dominios extendidos**, vía descomposición de
  dominio con mezcla coseno entre parches (`zeroshot_domain_decomp.gif`).
- **Súper-resolución** espacial de los campos generados (`super_resolution.gif`).
- **Método de fronteras inmersas (IBM)** para simular objetos dentro de los
  campos de flujo generados (`ibm_zeroshot.gif`).

El reporte también documenta limitaciones autorregresivas del modelo,
en particular la aparición de atractores espurios en rollouts de horizonte
largo — un hallazgo relevante para futuras mejoras de arquitectura y
entrenamiento.

## Estructura del repositorio

```
analysis/
├── Chosen dataset Physics Patterns.ipynb   <- Exploración de patrones físicos en el dataset elegido
├── Data Analysis (Exploratory).ipynb       <- Análisis exploratorio de datos
└── utilities.py

tasks/
├── Design.ipynb              <- Diseño de la arquitectura GA-FNO
├── Experiments.ipynb         <- Experimentos de entrenamiento
├── Preprocess Pipe.ipynb     <- Pipeline de preprocesamiento
├── SIMs.ipynb                <- Simulaciones / generación de resultados
├── media/                    <- GIFs y capturas de simulaciones calibradas
├── training_logs/            <- Diagnósticos por época: espectros de energía,
│                                enstrofía, palinstrofía, correlación espectral
│                                y curvas de pérdida a lo largo del entrenamiento
└── *.png                     <- Diagnósticos finales (atractores, dominio
                                  extendido, super-resolución, PCA/t-SNE)

GAN.pdf                       <- Reporte completo del proyecto
ibm_zeroshot.gif              <- Demo: método de fronteras inmersas
super_resolution.gif          <- Demo: súper-resolución
zeroshot_domain_decomp.gif    <- Demo: generalización a dominio extendido
```

## Cómo navegar el proyecto

1. Empieza por `GAN.pdf` para el contexto completo: arquitectura, metodología
   y resultados.
2. `analysis/` — cómo se entendió y preparó el dataset antes de entrenar.
3. `tasks/Design.ipynb` y `tasks/Experiments.ipynb` — diseño del modelo y
   entrenamiento.
4. `tasks/training_logs/` — evolución del entrenamiento época por época
   (espectros de energía y curvas de pérdida).
5. Los GIFs en la raíz muestran las tres extensiones de post-procesamiento
   (zero-shot, súper-resolución, IBM).
